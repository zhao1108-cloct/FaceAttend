# -*-coding:utf-8-*-
# date:2021-04-16
# function: train

import os
import warnings

warnings.filterwarnings("ignore")

from config import get_config
import argparse
# --------------------------------
from data.data_pipe import de_preprocess, get_train_loader
from model import Backbone, Arcface, l2_norm
import torch
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
from utils.utils import separate_bn_paras,schedule_lr
from PIL import Image
from torchvision import transforms as trans
import math
import time

def trainer(conf):
    # 加载训练集数据
    data_loader, class_num, datasets_len = get_train_loader(conf)

    # 模型选择：backbone
    model_ = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
    print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
    # 加载预训练模型:backbone
    if os.access(conf.finetune_backbone_model, os.F_OK):
        model_.load_state_dict(torch.load(conf.finetune_backbone_model))

    # 加载head模型，并添加预训练模型
    head_ = Arcface(embedding_size=conf.embedding_size, classnum=class_num).to(conf.device)
    if os.access(conf.finetune_head_model, os.F_OK):
        head_.load_state_dict(torch.load(conf.finetune_head_model))
        print("-------->>>   load head : {}".format(conf.finetune_head_model))

    # 获取网络中的不同层，用于进行网络优化
    paras_only_bn, paras_wo_bn = separate_bn_paras(model_)
    # 优化方法：weight_decay是正则化参数，将BN层分离出来不进行正则化
    optimizer = optim.SGD([
        {'params': paras_wo_bn + [head_.kernel], 'weight_decay': 5e-4},
        {'params': paras_only_bn}
    ], lr=conf.lr, momentum=conf.momentum)
    # 模型训练
    model_.train()
    # 迭代次数计数
    step_ = 0
    # 用来存放loss进行绘图
    loss_list = []
    # 遍历每一个epoch
    for e in range(conf.epochs):
        # 学习率衰减策略，变为原来的0.1倍
        print("  epoch < {} >".format(e))
        if e == conf.milestones[0]:
            schedule_lr(optimizer)
        if e == conf.milestones[1]:
            schedule_lr(optimizer)
        if e == conf.milestones[2]:
            schedule_lr(optimizer)
        # 遍历每一副图像
        for i, (imgs, labels) in enumerate(data_loader):
            if i >5:
                break
            # 将数据写入设备中
            imgs = imgs.to(conf.device)
            labels = labels.to(conf.device)
            optimizer.zero_grad()
            # 使用backbone模型获取特征向量
            embeddings = model_(imgs)
            # 使用head获取网络输出
            thetas = head_(embeddings, labels)
            # 计算损失
            loss = conf.ce_loss(thetas, labels)
            # 将损失放入list中，绘图
            loss_list.append(loss.detach().numpy())
            # 反向传播
            loss.backward()
            optimizer.step()
            # 每10个迭代次数打印信息
            if i % 1 == 0:
                print(
                    "  epoch - < {}/{} >, [{}/{}], loss: {:.6f} , bs: {}".format(
                        e, conf.epochs, i, int(datasets_len / conf.batch_size), loss.item(),
                        conf.batch_size))
            # 每100个迭代次数保存checkpoint
            if step_ % 1 == 0:
                # 保存路径
                save_path = conf.save_path
                # 若不存在，则创建该路径
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                # 获取当前时刻
                time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                # 保存backbone的结果
                torch.save(
                    model_.state_dict(), save_path +
                                         ('/model_{}_step_{}.pth'.format(time_str, step_)))
                # 保存head部分的结果
                torch.save(
                    head_.state_dict(), save_path +
                                        ('/head_{}_step_{}.pth'.format(time_str, step_)))
            # 迭代次数加1
            step_ += 1

    # 导入制图工具包
    import matplotlib.pyplot as plt
    # 创建第一张画布
    plt.figure(0)
    # 绘制总损失曲线 , 颜色为蓝色
    plt.plot(loss_list, color="blue", label="Loss")
    # 曲线说明在左上方
    plt.legend(loc='upper left')
    # 保存图片
    plt.savefig("./loss.png")


if __name__ == '__main__':
    # 获取配置信息
    conf = get_config()
    # print(conf.epochs)
    # 模型训练
    trainer(conf)
