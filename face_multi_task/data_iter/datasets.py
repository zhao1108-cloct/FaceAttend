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
from data_iter.data_agu import *
# from data_agu import *
from utils.draw_utils import draw_global_contour
import json


# 数据集加载
class LoadImagesAndLabels(Dataset):
    # 2.1 初始化处理
    def __init__(self, ops, img_size=(224, 224), flag_agu=False):
        # 参数的初始化
        # 年龄的最大值在0以上
        max_age = 0
        # 年龄的最小值在65535以下
        min_age = 65535.
        # 存储图像文件的list
        file_list = []
        # 关键点list
        landmarks_list = []
        # 年龄list
        age_list = []
        # 性别list
        gender_list = []
        # 图像计数
        idx = 0
        # 遍历所有的图像
        for f_ in os.listdir(ops.train_path):
            # 读取json文件
            f = open(ops.train_path + f_, encoding='utf-8')
            # 将json文件中的内容写入dict中
            dict = json.load(f)
            # 关闭文件流
            f.close()
            # 若年龄超出1-100之间，则循环下条数据
            if dict["age"] > 100. or dict["age"] < 1.:
                continue
            idx += 1
            # 获取标注json文件对应的图像文件
            img_path_ = (ops.train_path + f_).replace("label_new", "image").replace(".json", ".jpg")
            # 读取图像数据
            img = cv2.imread(img_path_)
            # 将路径保存在file_list中
            file_list.append(img_path_)
            # 关键点list
            pts = []
            # 遍历所有的关键点
            for pt_ in dict["landmarks"]:
                # 获取关键点坐标
                x, y = pt_
                # 将其保存在list中
                pts.append([x, y])
            # 将关键点保存下来
            landmarks_list.append(pts)
            # 创建性别的目标值
            if dict["gender"] == "male":
                gender_list.append(1)
            else:
                gender_list.append(0)
            # 将目标值存放在list中
            age_list.append(dict["age"])
            # 更新年龄的极值
            if max_age < dict["age"]:
                max_age = dict["age"]
            if min_age > dict["age"]:
                min_age = dict["age"]
        # 属性赋值
        self.files = file_list
        self.landmarks = landmarks_list
        self.ages = age_list
        self.genders = gender_list
        self.img_size = img_size
        self.flag_agu = flag_agu

    # 2.2 数据量
    def __len__(self):
        # 获取图像的个数
        return len(self.files)

    # 2.3 图像读取和增强
    def __getitem__(self, index):
        # 第一步：通过init中赋值的属性获取图像路经，关键点，性别和年龄
        # 获取图像路径
        img_path = self.files[index]
        # 关键点
        pts = self.landmarks[index]
        # 性别
        gender = self.genders[index]
        # 年龄
        age = self.ages[index]
        # 第二步：读入图像
        # 读取图像 -BGR
        img = cv2.imread(img_path)
        # 若进行图像增强,进行图像旋转
        if self.flag_agu and random.random() > 0.35:
            # 图像旋转 65%
            # 获取左眼和右眼的关键点的均值,用于计算旋转中心
            left_eye = np.average(pts[60:68], axis=0)
            right_eye = np.average(pts[68:76], axis=0)
            # 随机生成旋转角度
            angle_random = random.randint(-33, 33)
            # 返回旋转后的crop图和归一化的关键点:
            img_, landmarks_ = face_random_rotate(img, pts, angle_random, left_eye, right_eye, img_size=self.img_size)
        else:
            # 对人脸区域进行裁剪，并进行归一化
            x_max = -65535
            y_max = -65535
            x_min = 65535
            y_min = 65535
            # 遍历所有的关键点
            for pt_ in pts:
                # 获取x,y坐标
                x_, y_ = int(pt_[0]), int(pt_[1])
                # 获取关键点区域的左上角坐标和右下角坐标
                x_min = x_ if x_min > x_ else x_min
                y_min = y_ if y_min > y_ else y_min
                x_max = x_ if x_max < x_ else x_max
                y_max = y_ if y_max < y_ else y_max

            # 获取人脸区域的宽高
            face_w = x_max - x_min
            face_h = y_max - y_min
            # 对人脸区域进行随机的扩展
            x_min = int(x_min - random.randint(-6, int(face_w / 10)))
            y_min = int(y_min - random.randint(-6, int(face_h / 10)))
            x_max = int(x_max + random.randint(-6, int(face_w / 10)))
            y_max = int(y_max + random.randint(-6, int(face_h / 10)))
            # 确保坐标在图像范围内
            x_min = np.clip(x_min, 0, img.shape[1] - 1)
            x_max = np.clip(x_max, 0, img.shape[1] - 1)
            y_min = np.clip(y_min, 0, img.shape[0] - 1)
            y_max = np.clip(y_max, 0, img.shape[0] - 1)
            # 获取人脸区域的宽高
            face_w = x_max - x_min
            face_h = y_max - y_min
            # 裁剪人脸区域
            face_crop = img[y_min:y_max, x_min:x_max, :]
            # 关键点
            landmarks_ = []
            # 遍历所有的关键点
            for pt_ in pts:
                # 获取关键点相对于裁剪后人脸的坐标值
                x_, y_ = int(pt_[0]) - x_min, int(pt_[1]) - y_min
                # 将关键点进行归一化
                landmarks_.append([float(x_) / float(face_w), float(y_) / float(face_h)])
                # 将图像进行缩放
            img_ = cv2.resize(face_crop, self.img_size, interpolation=random.randint(0, 4))
        # 第三步：图像增强
        # 颜色增强
        if self.flag_agu:
            # 颜色增强 70%的概率
            if random.random() > 0.7:
                # 颜色空间转换
                img_hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)
                hue_x = random.randint(-10, 10)
                # 对H通道进行增强
                img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_x)
                # 对取值进行修正
                img_hsv[:, :, 0] = np.maximum(img_hsv[:, :, 0], 0)
                img_hsv[:, :, 0] = np.minimum(img_hsv[:, :, 0], 180)
                # 将色彩空间转换为BGR
                img_ = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        # 第四步：对数据进行归一化，类型等处理
        # 类型转换,BGR->RGB
        img_ = img_.astype(np.float32)[:, :, ::-1]
        # 归一化处理
        img_ = (img_ - 128.) / 256.
        # CHW->HWC
        img_ = img_.transpose(2, 0, 1)
        # 关键点的扁平化处理 [98，2]->[1,196]
        landmarks_ = np.array(landmarks_).ravel()
        # 年龄，去中心化，归一化
        age = np.expand_dims(np.array(((age - 50.) / 100.)), axis=0)
        return img_, landmarks_, gender, age


# 3 数据获取测试
if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description=' Project Multi Task Train')
    # 训练集标注信息
    parser.add_argument('--train_path', type=str,
                        default='/Users/mac/Documents/02.计算机视觉/03.人脸支付/02.code/facetoPay_edit/face_multi_task/multi_task_samples/label_new/',
                        help='train_path')
    parser.add_argument('--img_size', type=tuple, default=(256, 256),
                        help='img_size')  # 输入模型图片尺寸
    parser.add_argument('--flag_agu', type=bool, default=False,
                        help='data_augmentation')  # 训练数据生成器是否进行数据扩增
    ops = parser.parse_args()  # 解析添加参数

    dataset = LoadImagesAndLabels(ops=ops, img_size=ops.img_size, flag_agu=ops.flag_agu)
    print('len train datasets : %s' % (dataset.__len__()))
    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=2,
                            num_workers=4,
                            shuffle=True)
    for (imgs_, pts_, gender_, age_) in dataloader:
        print(gender_)
        print(age_ * 100.0 + 50.0)
        for j in range(2):
            # 对图像进行处理:反归一化，表示形式CHW->HWC,类型转换，RGB->BGR
            img = np.uint8(imgs_[j].permute(1, 2, 0) * 256.0 + 128.0)[:, :, ::-1]
            # 转换为CV2中的图像格式
            img = cv2.UMat(img).get()
            # 将年龄绘制在图像上
            cv2.putText(img, 'age:{:.2f}'.format(age_[j][0] * 100.0 + 50.0), (2, 20), cv2.FONT_HERSHEY_COMPLEX, 0.8,(0, 255, 0), 2)
            # 将性别绘制在图像上
            if gender_[j] == 1.:
                cv2.putText(img, 'gender:{}'.format("male"), (2, 40),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(img, 'gender:{}'.format("female"), (2, 40),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
            # 将扁平化处理的关键点reshape成相应的位置信息
            pts = pts_[j].reshape((-1,2))
            for pt_ in pts:
                # 获取关键点坐标
                x_, y_ = int(pt_[0]*256), int(pt_[1]*256)
                # 绘制关键点
                cv2.circle(img, (x_, y_), 2, (0, 255, 0), -1)
            # 结果展示
            cv2.imshow('result', img)
            cv2.waitKey(0)
    cv2.destroyAllWindows()

