import os
import cv2
import numpy as np
import time
import torch
from face_detect.yolov3 import Yolov3, Yolov3Tiny
from face_detect.utils.torch_utils import select_device
from face_detect.utils.utils import process_data, plot_one_box, non_max_suppression, scale_coords


# 人脸检测过程的实现
class yolo_v3_face_model(object):
    # 初始化
    def __init__(self,
                 model_path='./components/face_detect/weights/face_yolo_416-20210418.pt',
                 model_arch='yolov3',
                 yolo_anchor_scale=1.,
                 img_size=416,
                 conf_thres=0.4,
                 nms_thres=0.4, ):
        """
        :param model_path: 模型权重存储位置
        :param model_arch: 模型类别：yolo yolo-tiny
        :param yolo_anchor_scale: anchor的比例
        :param img_size: 送入网络中图像的大小
        :param conf_thres: 置信度阈值
        :param nms_thres: nms iou的阈值
        """
        pass

    # 模型预测
    def predict(self, img_, vis):
        pass