import glob
import math
import os
import random
import shutil
from pathlib import Path
from PIL import Image
# import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.utils import letterbox, random_affine, xywh2xyxy, xyxy2xywh


# import utils.letterbox as letterbox
# import utils.random_affine as random_affine
# import utils.xyxy2xywh as xyxy2xywh

# 获取数据：图像数据和标签数据
class LoadImagesAndLabels(Dataset):
    # 2.1 初始化处理
    def __init__(self, path, batch_size, img_size=416, augment=False, multi_scale=False, root_path=os.path.curdir):
        pass
    # 2.2 数据量
    def __len__(self):
        pass

    # 2.3 图像读取与增强
    def __getitem__(self, index):
        pass

    # 2.4 获取batch数据
    @staticmethod
    def collate_fn(batch):
        pass


# 3 数据获取测试
if __name__ == "__main__":
    pass