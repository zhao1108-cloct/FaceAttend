import argparse
import time
import os
import torch
from utils.datasets import *
from utils.utils import *
from utils.parse_config import parse_data_cfg
from yolov3 import Yolov3, Yolov3Tiny
from utils.torch_utils import select_device


# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# 图像预处理
def process_data(img, img_size=416):
    img, _, _, _ = letterbox(img, height=img_size)
    # 通道转换 BGR to RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)
    # 类型转换 uint8 to float32
    img = np.ascontiguousarray(img, dtype=np.float32)
    # 归一化 0 - 255 to 0.0 - 1.0
    img /= 255.0
    return img


def detect(model_path, cfg, data_cfg, img_size=416, conf_thres=0.5, nms_thres=0.5, video_path=0):
    """

    :param model_path: 模型路径
    :param cfg: 配置信息
    :param data_cfg: 数据配置信息
    :param img_size: 图像的大小
    :param conf_thres: 置信度阈值
    :param nms_thres: NMS阈值
    :param video_path: 要处理的视频路径
    :return:
    """
    # 获取检测的类别信息
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    num_classes = len(classes)
    # 1.模型加载


    # 2.数据加载


    # 3.遍历帧图像进行处理


# 4.模型使用
if __name__ == '__main__':
    pass
