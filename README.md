# ViServer - 视觉服务器
---
## 简介
ViServer是一款小型简易视觉服务器，供其他程序调用。


## 支持
- Yolov5。

## 使用方法
- 安装Python，推荐版本3.9。
- 安装核心程序依赖库。
```bash
pip install -r ./Conf/requirements.txt
```
- 根据所需使用的算法安装对应的依赖，例如Yolov5：
```bash
pip install -r ./Lib/Yolov5/requirements.txt
```
- 启动程序：
```bash
python VS.py \
    --weights your_weight.pt \
    --source rtsp://... \
```
- 通过POST方法获取视觉结果：
```Python
import requests
response=requests.post(
    "http://127.0.0.1:5005/Control",
    {"Command":"getBox"})
jdict=response.json()
"Return" in jdict
"Boxes" in jdict
"Image" in jdict
for box in jdict["Boxes"]:
    box[0]:int  # 标签类别。
    box[1]:float  # 置信度。
    box[2:6]  # xywh浮点坐标，注意xy为检测框中心。
```

## 启动参数
- weights：要加载的权重文件。
- source：检测源，可用网络流协议地址。留空为通过post上传图片进行检测。
- view：是否实时显示检测画面，推荐调试使用。默认False。
- port：ViServer服务器的运行端口，默认5005。
- method：使用的算法，默认"Yolov5"。
- 。。。

## POST命令
通过POST发送的字典中Command字段支持以下命令：
- detect：检测单张图片，需要有名为Image的文件域或者以Source字段发送文件路径。返回检测框列表。
- getBox：获取已检测的检测框列表。同时会以Image字段返回base64编码的当前图片。
