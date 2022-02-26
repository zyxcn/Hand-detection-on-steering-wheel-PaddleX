# Hand-detection-on-steering-wheel-PaddleX
基于yolov3，可以检测方向盘上的手

## 效果展示

| ![](https://ai-studio-static-online.cdn.bcebos.com/a912b1fb03bb40a2bad169a5d463e0cb4eb12639e8a74dc4854ca4c8c3c4f42a) | ![](https://ai-studio-static-online.cdn.bcebos.com/61ff8f319560483d979b448fe26f753e8f44958fcb134691a9455d8b869e47bf) |
| -------- | -------- |
| ![](https://ai-studio-static-online.cdn.bcebos.com/24d718e68b38458a8ba993dead586931e9aea20cd60e47cdb3ca9ccddb927c16)     | ![](https://ai-studio-static-online.cdn.bcebos.com/f39bbe0d596b4b78917986d702dc0e28ea4b848bb0004f5a89189bf0e2e31b28)     |

## 一、项目背景介绍
在公共交通工具如客车、出租车等车辆驾驶的过程中，驾驶员的对交通工具的控制将会影响到乘客及驾驶员自身的生命安全。 在驾驶过程中，驾驶员手中的方向盘是乘客可以触及的，一旦不理智的乘客使用影响驾驶员其控制车辆（如抢夺方向盘，阻挡驾驶员视线等），其后果不堪设想。 另外，如果驾驶员自身出现意外，无法控制车辆，同意可能产生意外。 好在一般的公共交通工具驾驶位基本都存在监控设备，使通过机器视觉来分辨驾驶过程中的紧急情况可行。
​
本项目旨在完成对方向盘附近手的识别这一步骤,可以称本项目为《保卫方向盘》。
​
本项目使用PaddleX进行训练，网络选择了MobileNetV3，代码简单，操作方便。

## 二、数据介绍


样例图片：

| ![](https://ai-studio-static-online.cdn.bcebos.com/b76435c52ede46468cf99462d9781c65c0d961cdc4f24fe2a76ac3753e67a775) | ![](https://ai-studio-static-online.cdn.bcebos.com/b561df53150745d88c980891eabcbd2242b65135aaa44925bd34fa18fa3a24dc) |
| -------- | -------- |

**数据集文件名称：** HandsOnSteeringWheel.zip

包含三个子文件夹：pos、posGt、posVOC，

对应标准VOC数据集：

├── pos ==> JPEGImages

├── posGt

└── posVOC ==> Annotations


其中pos文件夹中为图片：共有5500张png图片，每张图片的名称形如11_0000513_0_0_0_0.png。

posGt、posVOC中分别为为两种格式的标注文件

## 三、快速上手训练

安装PaddleX
``` 
pip install paddlex > /dev/null
```

克隆本项目
```
git clone https://github.com/zyxcn/Hand-detection-on-steering-wheel-PaddleX.git
cd Hand-detection-on-steering-wheel-PaddleX
```

下载数据集
``` 
wget -O data/dataset.zip https://bj.bcebos.com/v1/ai-studio-online/0fc5cfcf94ee41a0849ab3c0088128d579ffbf12044e41f38147d55319aefd03?responseContentDisposition=attachment%3B%20filename%3DHandsOnSteeringWheel.zip&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2021-01-31T05%3A00%3A28Z%2F-1%2F%2F5155de08d97f25b22beb307ca722ec45737791921315db28c6541f8af7e17dc4
```
解压数据集

``` 
unzip -oq data/dataset.zip -d data/HandsOnSteeringWheel
```

PaddleX PascalVOC数据集标准化
``` 
mv data/HandsOnSteeringWheel/pos data/HandsOnSteeringWheel/JPEGImages
mv data/HandsOnSteeringWheel/posVOC data/HandsOnSteeringWheel/Annotations
rm -rf data/HandsOnSteeringWheel/posGt
```

数据集划分
``` 
paddlex --split_dataset --format VOC --dataset_dir data/HandsOnSteeringWheel --val_value 0.2 --test_value 0.1
```

开始训练
```
python script/train.py
```

## 四、模型预测

```python
import paddlex as pdx
from matplotlib import pyplot as plt 

model = pdx.load_model('output/MobileNetV3/best_model')
image_name = 'data/HandsOnSteeringWheel/JPEGImages/5L_0053479_I_3_0_3.png'
result = model.predict(image_name)
pdx.det.visualize(image_name, result, threshold=0.5, save_dir='./output/visualize')
plt.imshow(plt.imread('./output/visualize/visualize_'+image_name[image_name.rfind('/')+1:]))
```

