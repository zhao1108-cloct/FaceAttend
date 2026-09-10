# coding:utf-8
import os
from yolov3 import Yolov3, Yolov3Tiny
from utils.parse_config import parse_data_cfg
from utils.torch_utils import select_device
import torch
from torch.utils.data import DataLoader
from utils.datasets import LoadImagesAndLabels
from utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import time


def train(data_cfg='cfg/face.data'):
    pass
    # 1.配置文件解析

    # 2.模型加载

    # 3.数据加载

    # 4.模型训练


# -------------------------------------------------------------------------------
if __name__ == '__main__':
    train(data_cfg="cfg/face.data")
    print('well done ~ ')
