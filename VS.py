import random
import sys, os
import threading
import time

import flask as fl
import torch
import argparse
import cv2
import numpy as np
from pathlib import Path
import logging
import base64

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = str(ROOT)
sys.path.append(ROOT + "\\Lib")


def image_to_base64(image_np):

    image = cv2.imencode(".jpg", image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]

    return image_code


class ViServerCore:
    def __init__(self) -> None:
        self.Option = vars(self.parseOpt())
        self.Method = self.Option["method"]
        self.Source = self.Option["source"]
        self.Weights = self.Option["weights"]
        self.Confidence = self.Option["conf_thres"]
        self.IOU = self.Option["iou_thres"]
        self.MaxDet = self.Option["max_det"]
        self.Device = self.Option["device"]
        self.FlagRun = True
        self.Boxes = list()
        self.ImgSize = [640, 640]
        self.ImageTemp = None
        self.FlagWarm = False
        self.DictColor = dict()
        return

    @staticmethod
    def parseOpt():
        parser = argparse.ArgumentParser()

        parser.add_argument("--port", type=int, default=5005, help="ViServer port.")
        parser.add_argument("--view", type=bool, default=True)
        parser.add_argument(
            "--method", type=str, default="Yolov5", help="MYolo server port."
        )

        parser.add_argument(
            "--weights",
            nargs="+",
            type=str,
            default="../TowerDriverGradeSys/Res/Pt/Changde2.pt",
            # default="Lib/Yolov5/Pt/best.pt",
            help="model path(s)",
        )
        parser.add_argument(
            "--source",
            type=str,
            default=r"test.jpg",
            # default=r"rtsp://192.168.82.191:554/main_stream",
            help="file/URL",
        )

        parser.add_argument(
            "--imgsz",
            "--img",
            "--img-size",
            nargs="+",
            type=int,
            default=[640],
            help="inference size h,w",
        )
        parser.add_argument(
            "--conf-thres", type=float, default=0.25, help="confidence threshold"
        )
        parser.add_argument(
            "--iou-thres", type=float, default=0.45, help="NMS IoU threshold"
        )
        parser.add_argument(
            "--max-det", type=int, default=100, help="maximum detections per image"
        )
        parser.add_argument(
            "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
        )

        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        # print_args(FILE.stem, opt)
        return opt

    def initServer(self):

        def threadServer():
            self.WebServer.run("127.0.0.1", port=self.Option["port"])
            return

        self.WebServer = fl.Flask(__name__)
        werkzeuglog = logging.getLogger("werkzeug")
        werkzeuglog.setLevel(logging.ERROR)

        @self.WebServer.route("/Command", methods=["POST"])
        def command():
            return self.respCommand()

        self.ThdServer = threading.Thread(target=threadServer, daemon=True)
        self.ThdServer.start()
        return

    def respCommand(self):
        ret = {"Return": 0}

        form = fl.request.form
        if "Command" not in form:
            ret["Return"] = 500
            return fl.jsonify(ret)

        cmd = form["Command"]
        if cmd == "detect":
            file = fl.request.files.get("Image")
            if file is not None:
                image_np = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            else:
                res = self.detect(form["Source"])
                ret["Boxes"] = res
        elif cmd == "getBox":
            ret["Boxes"] = self.Boxes
            if self.ImageTemp is not None:
                ret["Image"] = image_to_base64(self.ImageTemp)
        return fl.jsonify(ret)

    def loadModel(self):

        res = False
        self.FlagStream = self.Source.lower().startswith(
            ("rtsp://", "rtmp://", "http://", "https://")
        )
        print(self.Source)
        if self.Method == "Yolov5":
            try:
                self.loadYolov5()
                res = True
            except BaseException as e:
                print(e)
        return res

    def loadYolov5(self):
        sys.path.append(ROOT + "\\Lib\\Yolov5")
        # print(sys.path)
        from Yolov5.models.common import DetectMultiBackend
        from Yolov5.utils.torch_utils import select_device
        from Yolov5.utils.general import check_img_size

        imgsz = [640]
        imgsz *= 2 if len(imgsz) == 1 else 1

        weights = self.Weights
        self.Device = select_device(self.Device)

        stride = 64
        names = [f"class{i}" for i in range(100)]  # assign defaults
        model = DetectMultiBackend(weights, device=self.Device)
        stride, names, pt = model.stride, model.names, model.pt

        # model = attempt_load(weights, map_location=self.Device)
        # stride = int(model.stride.max())  # model stride
        names = (
            model.module.names if hasattr(model, "module") else model.names
        )  # get class names

        imgsz = check_img_size(imgsz, s=stride)  # check image size

        self.Stride = stride
        self.Model = model
        self.ImgSize = imgsz
        self.Names = names
        return True

    def detect(self, src=""):
        ret = None
        if self.Method == "Yolov5":
            ret = self.detectYolov5(src)
        return ret

    def detectYolov5(self, source=""):
        from Yolov5.utils.general import non_max_suppression
        from Yolov5.utils.general import scale_boxes, xyxy2xywh
        from Yolov5.utils.dataloaders import LoadImages, LoadStreams

        if source == "":
            source = self.Source

        if not self.FlagStream:
            dataset = LoadImages(source, img_size=self.ImgSize, stride=self.Stride)
            print("Use image.")
        else:
            dataset = LoadStreams(source, img_size=self.ImgSize, stride=self.Stride)
            print("Use stream.")

        self.FlagWarm = False
        if not self.FlagWarm:
            self.Model.warmup(imgsz=(1, 3, *self.ImgSize))  # warmup
            self.FlagWarm = True
        seen = 0
        for path, img, im0s, vid_cap, s in dataset:
            boxes = []
            img = torch.from_numpy(img).to(self.Device)
            img = img.float() / 255.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

            # Inference
            if True:
                pred = self.Model(img, augment=False, visualize=False)

            # NMS
            pred = non_max_suppression(
                pred, self.Confidence, self.IOU, None, False, max_det=self.MaxDet
            )

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if self.FlagStream:
                    im0s = im0s[0]
                im0 = im0s.copy()
                self.ImageTemp = im0s

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if len(det):
                    det[:, :4] = scale_boxes(
                        img.shape[2:], det[:, :4], im0.shape
                    ).round()

                    for *xyxy, conf, cls in reversed(det):
                        xywh = (
                            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                            .view(-1)
                            .tolist()
                        )
                        boxes.append([int(cls), int(conf), *xywh])

                if self.Option["view"]:
                    imgv = im0s.copy()
                    for box in boxes:
                        if box[0] not in self.DictColor:
                            self.DictColor[box[0]] = [
                                random.randint(0, 255),
                                random.randint(0, 255),
                                random.randint(0, 255),
                            ]
                        pt1 = [
                            int((box[2] - box[4] / 2) * imgv.shape[1]),
                            int((box[3] - box[5] / 2) * imgv.shape[0]),
                        ]
                        pt2 = [
                            int((box[2] + box[4] / 2) * imgv.shape[1]),
                            int((box[3] + box[5] / 2) * imgv.shape[0]),
                        ]
                        imgv = cv2.rectangle(imgv, pt1, pt2, self.DictColor[box[0]], 2)
                    cv2.imshow("", imgv)

            self.Boxes = boxes
            # time.sleep(0.01)
        if not self.FlagStream:
            self.Source = ""
        return boxes

    def run(self):
        try:
            self.initServer()
            flag = self.loadModel()
            if not flag:
                print("Load failed.")
            else:
                print("Load finished.")
            # self.FlagStream = True
            if self.FlagStream and flag:
                thd = threading.Thread(target=self.detect)
                thd.start()

        except BaseException as e:
            self.FlagRun = False
            print(e)
        while self.FlagRun:
            time.sleep(1)
            continue
        return

    pass


if __name__ == "__main__":
    VSC = ViServerCore()
    VSC.run()
