import argparse
from pathlib import Path

import cv2
import torch
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging

from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

class Detector():
    def __init__(self, weights = 'yolov7-ContextNav.pt', img_size = 320, trace = True, 
                 augment = False, conf_thres = 0.25, iou_thres = 0.45,
                 classes = None, agnostic_nms = False):
        self.weights, self.imgsz, self.trace =  weights, img_size, trace
        self.augment, self.conf_thres, self.iou_thres = augment, conf_thres, iou_thres
        self.classes, self.agnostic_nms = classes, agnostic_nms
        # Initialize
        set_logging()
        self.device = select_device(opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size

        if self.trace:
            self.model = TracedModel(self.model, self.device, self.imgsz)

        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1



    def detect(self, img_path):
   
        path = Path(img_path) # Remove after ROS implementation

        for image in path.iterdir(): # Remove after ROS implementation
            # TODO change to read images from ROS2
            im0s = cv2.imread(str(image))  # BGR # Remove after ROS implementation
            assert im0s is not None, 'Image Not Found ' + path

            img = letterbox(im0s, self.imgsz, stride=self.stride)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
                self.old_img_b = img.shape[0]
                self.old_img_h = img.shape[2]
                self.old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=self.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=self.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
            
                s, im0 = '', im0s

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf)
                        output = (('%g ' * len(line)).rstrip() % line).split(' ')
                        
                        ID = output[0]
                        x = output[1]
                        y = output[2]
                        w = output[3]
                        h = output[4]
                        confi = output[5]

                        print(f"Class = {ID}, x = {x}, y = {y}, width = {w}, height = {h}, conf = {confi}") # Remove after ROS implementation

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')




if __name__ == '__main__':

    with torch.no_grad():
        detector = Detector()
        img_path = input("Put image path:\n") # Remove after ROS implementation
        while img_path: # Remove after ROS implementation - change to callback func
            detector.detect(img_path)
            img_path = input("Put image path:\n") # Remove after ROS implementation
