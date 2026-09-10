import os
import torch
import cv2
import torch.nn.functional as F

from face_euler_angle.network.resnet import resnet18
from face_euler_angle.utils.common_utils import *

import numpy as np

# 人脸姿态检测
class FaceAngle_Model(object):
    # 初始化
    def __init__(self,
                 model_path='resnet_18_imgsize_256-epoch-225.pth',
                 img_size=256,
                 num_classes=3,  # yaw,pitch,roll
                 ):
        """
        :param model_path: 训练好的模型路径
        :param img_size: 图像大小
        :param num_classes: 三个角度信息
        """
        # 设备信息获取
        use_cuda = torch.cuda.is_available()
        self.use_cuda = use_cuda
        self.device = torch.device("cuda:0" if use_cuda else "cpu")  # 可选的设备类型及序号
        # 图像大小
        self.img_size = img_size
        # 模型实例化
        model_ = resnet18(num_classes=num_classes, img_size=img_size)
        # 加载预训练模型参数
        chkpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        model_.load_state_dict(chkpt)
        # 设置为eval模式
        model_.eval()
        # 写入设备中
        self.model_ = model_.to(self.device)

    # 预测过程
    def predict(self, img):
        with torch.no_grad():
            # 图像数据写入设备中
            img_ = torch.from_numpy(img)
            if self.use_cuda:
                img_ = img_.cuda()  # (bs, 3, h, w)
            # 模型预测
            output_ = self.model_(img_.float())
            # 获取预测结果
            output_ = output_.cpu().detach().numpy()
            # 反归一化处理
            output_ = output_ * 90.
        # 返回结果
        return output_
