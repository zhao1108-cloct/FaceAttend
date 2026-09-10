
import os
import argparse
import torch
import torch.nn as nn
import numpy as np

import math
import cv2
import torch.nn.functional as F

from models.resnet import resnet50, resnet34, resnet18
from utils.common_utils import *

if __name__ == "__main__":
    # 1.配置信息解析
    parser = argparse.ArgumentParser(description=' Project face euler angle Test')
    # 训练好的模型路径
    parser.add_argument('--test_model', type=str,
                        default='./model_exp/2021-05-13_07-07-30/resnet_18_imgsize_256-epoch-5.pth',
                        help='test_model')
    # 模型类型
    parser.add_argument('--model', type=str, default='resnet_18',
                        help='model : resnet_x')
    # 分类类别个数
    parser.add_argument('--num_classes', type=int, default=3,
                        help='num_classes')
    # GPU选择
    parser.add_argument('--GPUS', type=str, default='0',
                        help='GPUS')
    # 测试集路径
    parser.add_argument('--test_path', type=str, default='./samples/',
                        help='test_path')
    # 输入模型图片尺寸
    parser.add_argument('--img_size', type=tuple, default=(256, 256),
                        help='img_size')
    # 是否可视化图片
    parser.add_argument('--vis', type=bool, default=True,
                        help='vis')
    print('\n/******************* {} ******************/\n'.format(parser.description))
    # --------------------------------------------------------------------------
    # 解析添加参数
    ops = parser.parse_args()
    # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    unparsed = vars(ops)
    # 打印参数配置信息
    for key in unparsed.keys():
        print('{} : {}'.format(key, unparsed[key]))
    # 设备设置
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS
    # 测试图片文件夹路径
    test_path = ops.test_path
    # 2.模型加载
    # 第一步：构建模型
    model = resnet18(pretrained=False,num_classes=ops.num_classes,img_size = ops.img_size[0])
    # 第二步：获取设备信息
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    # 第三步：加载预训练模型
    if os.access(ops.test_model,os.F_OK):
        model.load_state_dict(torch.load(ops.test_model,map_location=device))
    # 3.数据加载
    for file in os.listdir(ops.test_path):
        if '.jpg' not in file:
            continue
        img = cv2.imread(ops.test_path+file)
        img_ = cv2.resize(img,(ops.img_size[1],ops.img_size[0]))
        img_ = (img_.astype(np.float32)-128.)/256.
        img_ = img_[:,:,::-1]
        img_ = img_.transpose(2,0,1)
        print(img_.shape)
        img_ =torch.from_numpy(img_.copy())
        img_=img_.unsqueeze_(0)

        # 4.模型预测
        angle = model(img_.float())
        angle=angle.cpu().detach().numpy()
        yaw,pitch,roll=np.squeeze(angle)
        yaw *=90
        pitch *=90
        roll *=90
        # 将检测结果绘制在图像上
        cv2.putText(img, "ypr:{:.1f},{:.1f},{:.1f}".format(yaw, pitch, roll), (1, 80), cv2.FONT_HERSHEY_DUPLEX, 2,
                    (55, 0, 220), 5)
        cv2.putText(img, "ypr:{:.1f},{:.1f},{:.1f}".format(yaw, pitch, roll), (1, 80), cv2.FONT_HERSHEY_DUPLEX, 2,
                    (255, 50, 50), 2)
        cv2.imshow('result',img)
        cv2.waitKey(0)
