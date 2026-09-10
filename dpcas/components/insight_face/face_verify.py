# -*-coding:utf-8-*-
# date:2021-04-16
# Author: Eric.Lee
# function: face verify

import warnings

warnings.filterwarnings("ignore")
import os
import torch
from insight_face.model import Backbone
from pathlib import Path
from PIL import Image
import cv2
from torchvision import transforms as trans
import torch
from insight_face.model import l2_norm
import pdb
import cv2
import numpy as np


# 加载pth,npy文件中存储的特征
def load_facebank(facebank_path):
    embeddings = torch.load(facebank_path + '/facebank.pth',
                            map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    names = np.load(facebank_path + '/names.npy')
    return embeddings, names


def infer(model, device, faces, target_embs, threshold=1.2, tta=False):
    '''
    :param model: 进行预测的模型
    :param device: 设备信息
    :param faces: 要处理的人脸图像
    :param target_embs: 数据库中的人脸特征
    :param threshold: 阈值
    :param tta: 进行水平翻转的增强
    :return:
    '''
    # 将类型转换和标准化合并在一起
    test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 特征向量
    embs = []
    # 遍历人脸图像
    for img in faces:
        # 若进行翻转
        if tta:
            # 镜像翻转
            mirror = trans.functional.hflip(img)
            # 模型预测
            emb = model(test_transform(img).to(device).unsqueeze(0))
            emb_mirror = model(test_transform(mirror).to(device).unsqueeze(0))
            # 获取最终的特征向量
            embs.append(l2_norm(emb + emb_mirror))
        else:
            with torch.no_grad():
                # 未进行翻转时，进行预测
                embs.append(model(test_transform(img).to(device).unsqueeze(0)))
    # 将特征拼接在一起
    source_embs = torch.cat(embs)
    # 计算要检测的图像特征与目标特征之间的差异
    diff = source_embs.unsqueeze(-1) - target_embs.transpose(1, 0).unsqueeze(0)
    dist = torch.sum(torch.pow(diff, 2), dim=1)
    # 获取差异最小值及对应的索引
    minimum, min_idx = torch.min(dist, dim=1)
    # 若没有匹配成功，将索引设置为-1
    min_idx[minimum > threshold] = -1
    return min_idx, minimum


class insight_face_model(object):
    def __init__(self,
                 net_mode="ir_se",  # [ir, ir_se]
                 net_depth=50,  # [50,100,152]
                 backbone_model_path="./components/insight_face/weights/model_ir_se-50.pth",
                 facebank_path="./components/insight_face/facebank",  # 人脸库
                 tta=False,
                 threshold=1.2
                 ):
        '''
        :param net_mode: 模型类型
        :param net_depth: 网络深度
        :param backbone_model_path: backbone的权重
        :param facebank_path: 人脸库
        :param tta: 是否进行翻转
        :param threshold: 阈值
        '''
        # 属性赋值
        self.threshold = threshold
        self.tta = tta
        # 设置选择
        device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 实例化backbone
        model_ = Backbone(net_depth, 1., net_mode).to(device_)
        # 加载预训练模型
        if os.access(backbone_model_path, os.F_OK):
            model_.load_state_dict(torch.load(backbone_model_path, map_location=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")))
        # 设置为eval模式
        model_.eval()
        # 属性赋值
        self.model_ = model_
        self.device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 加载人脸库，获取特征向量和人名
        targets, names = load_facebank(facebank_path)
        self.face_targets = targets
        self.face_names = names

    # 模型预测
    def predict(self, faces_identify):
        with torch.no_grad():
            # 模型预测
            results, face_dst = infer(self.model_, self.device_, faces_identify, self.face_targets,
                                      threshold=self.threshold, tta=self.tta)
        # 返回人脸识别结果
        return results, face_dst
