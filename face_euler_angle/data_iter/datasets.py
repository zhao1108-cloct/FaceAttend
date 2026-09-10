import glob
import math
import os
import random
import shutil
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_iter.data_agu import *
import json


# 数据集加载
class LoadImagesAndLabels(Dataset):
    # 2.1 初始化处理
    def __init__(self, ops, img_size=(224, 224), flag_agu=False):
        # 初始化：图像文件列表，bboxes_list列表，角度列表
        file_list = []
        bboxes_list = []
        angles_list = []
        # 计数
        idx = 0
        # 获取图像路径
        images_path = ops.train_path + "images/"
        # 遍历每一个图像
        for f_ in os.listdir(images_path):
            # 获取对应的label路径
            label_path = (images_path + f_).replace("images", "labels").replace(".jpg", ".json")
            # 读取 json文件
            f = open(label_path, encoding='utf-8')
            # 加载标注数据
            dict = json.load(f)
            # 关闭文件流
            f.close()
            # 获取角度和bbox
            angle = dict["euler_angle"]
            bbox = dict["bbox"]
            idx += 1
            print("  images : {}".format(idx), end="\r")
            # 将图像文件，bbox,角度添加到list列表中
            file_list.append(images_path + f_)
            bboxes_list.append((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
            angles_list.append((angle["yaw"], angle["pitch"], angle["roll"]))
        # 属性赋值
        self.files = file_list
        self.bboxes = bboxes_list
        self.angles = angles_list
        self.img_size = img_size
        self.flag_agu = flag_agu

    # 2.2 数据量
    def __len__(self):
        return len(self.files)

    # 2.3 图像读取和增强
    # 获取每一条训练数据
    def __getitem__(self, index):
        img_path = self.files[index]
        box = self.bboxes[index]
        yaw, pitch, roll = self.angles[index]
        img = cv2.imread(img_path)
        face_w = box[2] - box[0]
        face_h = box[3] - box[1]
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        xmin = int(xmin - random.randint(-6, int(face_w * 2 / 3)))
        ymin = int(ymin - random.randint(-6, int(face_h * 2 / 3)))
        xmax = int(xmax + random.randint(-6, int(face_w * 2 / 3)))
        ymax = int(ymax + random.randint(-12, int(face_h * 2 / 3)))
        xmin = np.clip(xmin, 0, img.shape[1] - 1)
        xmax = np.clip(xmax, 0, img.shape[1] - 1)
        ymin = np.clip(ymin, 0, img.shape[0] - 1)
        ymax = np.clip(ymax, 0, img.shape[0] - 1)
        try:
            face_crop = img[ymin:ymax, xmin:xmax, :]
        except:
            face_crop = img[box[1]:box[3], box[0]:box[2], :]
        # 几何
        if random.random() >= 0.5:
            face_crop = cv2.flip(face_crop, 1)
            yaw = -yaw
            roll = -roll
        img_ = cv2.resize(face_crop, self.img_size, interpolation=0)
        # 颜色
        if self.flag_agu == True:
            if random.random() > 0.7:
                img_hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)
                s_v = random.randint(-10, 10)
                img_hsv[:, :, 1] = img_hsv[:, :, 1] + s_v
                img_hsv[:, :, 1] = np.maximum(img_hsv[:, :, 1], 0)
                img_hsv[:, :, 1] = np.minimum(img_hsv[:, :, 1], 255)
                img_ = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        img_ = img_.astype(np.float32)[:, :, ::-1]
        img_ = (img_ - 128.) / 256.
        img_ = img_.transpose(2, 0, 1)

        yaw = yaw / 90.
        pitch = pitch / 90.
        roll = roll / 90.
        angle_ = np.array([yaw, pitch, roll]).ravel()
        return img_, angle_


# 3 数据读取测试
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=' Project Face Euler Angle Train')
    #  yaw,pitch,roll
    parser.add_argument('--num_classes', type=int, default=3,
                        help='num_classes')
    # 训练集标注信息
    parser.add_argument('--train_path', type=str,
                        default='/Users/mac/Documents/02.计算机视觉/03.人脸支付/02.code/datasets/face_euler_angle_datasets/',
                        help='train_path')
    # 训练每批次图像数量
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size')
    # 训练线程数
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num_workers')
    # 输入模型图片尺寸
    parser.add_argument('--img_size', type=tuple, default=(256, 256),
                        help='img_size')
    # 是否进行数据增强
    parser.add_argument('--flag_agu', type=bool, default=True,
                        help='data_augmentation')

    # --------------------------------------------------------------------------
    ops = parser.parse_args()  # 解析添加参数
    dataset = LoadImagesAndLabels(ops, img_size=ops.img_size, flag_agu=ops.flag_agu)
    # img,angel = dataset.__getitem__(10)
    # print(angel)
    # # print(dataset.__getitem__(10))
    # cv2.imshow('result', np.uint8(img.transpose(1, 2, 0) * 256.0 + 128.0)[:, :, ::-1])
    # cv2.waitKey(0)
    dataloader = DataLoader(dataset, batch_size=ops.batch_size, num_workers=ops.num_workers, shuffle=True)
    for i,(imgs,angles) in enumerate(dataloader):
        print(angles)
        for j in range(ops.batch_size):
            cv2.imshow('results',np.uint8(imgs[j].permute(1,2,0)*256.+128.)[:,:,::-1])
            cv2.waitKey(0)
    cv2.destroyAllWindows()
