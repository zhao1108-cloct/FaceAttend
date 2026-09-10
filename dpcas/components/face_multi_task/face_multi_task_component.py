import os
import torch
import cv2
import torch.nn.functional as F

from face_multi_task.network.resnet import resnet50, resnet34, resnet18
from face_multi_task.utils.common_utils import *
import numpy as np


# 人脸多任务实现
class FaceMuitiTask_Model(object):
    def __init__(self,
                 model_path='./components/face_multi_task/weights_multask/resnet_50_imgsize-256-20210411.pth',
                 img_size=256,
                 num_classes=196,
                 model_arch="resnet50",  # 模型结构
                 ):
        """
        :param model_path: 模型参数存储路径
        :param img_size: 图像大小
        :param num_classes: # 人脸关键点*2
        :param model_arch: 模型类型
        """
        # 设备信息的获取
        use_cuda = torch.cuda.is_available()
        self.use_cuda = use_cuda
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        # 图像大小
        self.img_size = img_size
        # 模型实例化
        if model_arch == "resnet50":
            face_multi_model = resnet50(landmarks_num=num_classes, img_size=img_size)
        elif model_arch == "resnet34":
            face_multi_model = resnet34(landmarks_num=num_classes, img_size=img_size)
        # 加载模型参数
        chkpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        face_multi_model.load_state_dict(chkpt)
        # 设置为eval模式
        face_multi_model.eval()
        # 写入设备中
        self.face_multi_model = face_multi_model.to(self.device)
    # 模型预测
    def predict(self, img):
        with torch.no_grad():
            # 图像数据
            img_ = torch.from_numpy(img)
            if self.use_cuda:
                img_ = img_.cuda()  # (bs, 3, h, w)
            # 模型预测
            output_landmarks, output_gender, output_age = self.face_multi_model(img_.float())
            # 关键点信息
            output_landmarks = output_landmarks.cpu().detach().numpy()
            # 性别信息
            output_gender = output_gender.cpu().detach().numpy()
            output_gender = np.array(output_gender)
            # 年龄
            output_age = output_age.cpu().detach().numpy()
            output_age = (output_age * 100. + 50.)
        # 返回结果
        return output_landmarks, output_gender, output_age
