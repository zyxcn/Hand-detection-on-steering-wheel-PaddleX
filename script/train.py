import paddlex as pdx
from paddlex import transforms as T

# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
import matplotlib
matplotlib.use('Agg') 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

train_transforms = T.Compose([
    T.MixupImage(mixup_epoch=250),
    T.RandomDistort(),
    T.RandomExpand(),
    T.RandomCrop(),
    T.Resize(target_size=608, interp='RANDOM'),
    T.RandomHorizontalFlip(),
    T.Normalize()

])

eval_transforms = T.Compose([
    T.Resize(target_size=608, interp='CUBIC'),
    T.Normalize()
])

train_dataset = pdx.datasets.VOCDetection(
                        data_dir='../data/HandsOnSteeringWheel',
                        file_list='../data/HandsOnSteeringWheel/train_list.txt',
                        label_list='../data/HandsOnSteeringWheel/labels.txt',
                        transforms=train_transforms)
                        
eval_dataset = pdx.datasets.VOCDetection(
                        data_dir='../data/HandsOnSteeringWheel',
                        file_list='../data/HandsOnSteeringWheel/val_list.txt',
                        label_list='../data/HandsOnSteeringWheel/labels.txt',
                        transforms=eval_transforms)

# 定义模型
num_classes = len(train_dataset.labels)
model = pdx.det.YOLOv3(num_classes=num_classes, backbone='MobileNetV3')

#开始训练
model.train(
    num_epochs=300,
    train_dataset=train_dataset,
    train_batch_size=32,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    lr_decay_epochs=[210, 240],
    save_interval_epochs=20,
    save_dir='../output/MobileNetV3',
    use_vdl=True)