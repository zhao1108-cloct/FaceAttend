import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from utils.model_utils import *
from utils.common_utils import *
from data_iter.datasets import *

from models.resnet import resnet50, resnet34, resnet18
from loss.loss import got_total_wing_loss, wing_loss
import cv2
import time
import json
from datetime import datetime
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def trainer(ops):
    # 设备信息
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS
    #  构建模型
    # 2.模型加载
    # 第一步：模型结构初始化
    # 模型加载
    if ops.model == 'resnet_50':
        model_ = resnet50(pretrained=False, landmarks_num=ops.num_classes, img_size=ops.img_size[0],
                          dropout_factor=ops.dropout)
    elif ops.model == 'resnet_34':
        model_ = resnet34(pretrained=False, landmarks_num=ops.num_classes, img_size=ops.img_size[0],
                          dropout_factor=ops.dropout)
    elif ops.model == 'resnet_18':
        model_ = resnet18(pretrained=False, landmarks_num=ops.num_classes, img_size=ops.img_size[0],
                          dropout_factor=ops.dropout)
    # 若有GPU使用GPU进行训练
    use_cuda = torch.cuda.is_available()
    # 否则使用CPU
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # 将网络写入设备中
    model_ = model_.to(device)
    # 第二步：加载预训练模型
    # 加载预训练模型
    if os.access(ops.fintune_model, os.F_OK):  # checkpoint
        chkpt = torch.load(ops.fintune_model, map_location=device)
        model_.load_state_dict(chkpt)
        print('load fintune model : {}'.format(ops.fintune_model))
    # 3.数据加载
    dataset = LoadImagesAndLabels(ops=ops, img_size=ops.img_size, flag_agu=ops.flag_agu)
    print('len train datasets : %s' % (dataset.__len__()))
    # Dataloader获取batchsize的数据
    dataloader = DataLoader(dataset,
                            batch_size=ops.batch_size,
                            num_workers=ops.num_workers,
                            shuffle=True)

    # 4.模型训练
    # 第一步：相关参数设置
    # 优化器设计
    optimizer = torch.optim.Adam(model_.parameters(), lr=ops.init_lr, betas=(0.9, 0.99),
                                 weight_decay=ops.weight_decay)

    # 损失函数：用于计算年龄和关键点
    if ops.loss_define != 'wing_loss':
        criterion = nn.MSELoss(reduce=True, reduction='mean')
    # 交叉熵损失函数：softmax+损失的组合
    criterion_gender = nn.CrossEntropyLoss()
    # 学习率
    init_lr = ops.init_lr
    # 初始化损失，将损失添加到列表中用于绘制训练曲线
    pts_loss = []
    gender_loss = []
    age_loss = []
    sum_loss = []

    # 第二步：遍历每个epoch开始进行训练
    # 遍历每个epoch进行训练
    for epoch in range(0, ops.epochs):
        # 模型训练开始
        model_.train()
        # 损失均值
        loss_mean = 0.
        # 损失计算计数器
        loss_idx = 0.
        # 第三步：遍历batch中的数据，进行预测
        # 遍历每个batch中的数据
        for i, (imgs_, pts_, gender_, age_) in enumerate(dataloader):
            # 将数据写入设备中
            if use_cuda:
                imgs_ = imgs_.cuda()
                pts_ = pts_.cuda()
                gender_ = gender_.cuda()
                age_ = age_.cuda()
            # 将图像送入网络中，进行预测
            output_landmarks, output_gender, output_age = model_(imgs_.float())
            # 计算年龄和关键点的损失
            if ops.loss_define == 'wing_loss':
                loss_pts = got_total_wing_loss(output_landmarks, pts_.float())
                loss_age = got_total_wing_loss(output_age, age_.float())
            else:
                loss_pts = criterion(output_landmarks, pts_.float())
                loss_age = criterion(output_age, age_.float())
            # 计算性别的损失
            loss_gender = criterion_gender(output_gender, gender_)
            pts_loss.append(loss_pts.item())
            age_loss.append(loss_age.item())
            gender_loss.append(loss_gender.item())
            # 多任务损失:不同任务的损失的权重是不一样的，相对较难的任务权重较大
            loss = loss_pts + 0.3 * loss_age + 0.25 * loss_gender
            sum_loss.append(loss.item())
            # 求损失均值
            loss_mean += loss.item()
            # 计数加1
            loss_idx += 1.
            # 每10个batch打印一次结果
            if i % 1 == 0:
                loc_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print('  %s - %s - epoch [%s/%s] (%s/%s):' % (
                    loc_time, ops.model, epoch, ops.epochs, i, int(dataset.__len__() / ops.batch_size)), \
                      'Mean Loss : %.6f - Loss: %.6f' % (loss_mean / loss_idx, loss.item()), \
                      " loss_pts:{:.4f},loss_age:{:.4f},loss_gender:{:.4f}".format(loss_pts.item(), loss_age.item(),
                                                                                   loss_gender.item()), \
                      ' lr : %.5f' % init_lr, ' bs:', ops.batch_size, \
                      ' img_size: %s x %s' % (ops.img_size[0], ops.img_size[1]))
            # 计算梯度
            loss.backward()
            # 优化器对模型参数更新
            optimizer.step()
            # 优化器梯度清零
            optimizer.zero_grad()
            break

        # 第四步：保存ckpt
        # 每3个epoch保存一次训练结果
        if epoch % 3 == 0:
            torch.save(model_.state_dict(), ops.model_exp + '{}_epoch-{}.pth'.format(ops.model, epoch))

    # 第五步：损失变化的曲线
    # 创建第一张画布
    plt.figure(0)
    # 绘制pts损失曲线
    plt.plot(pts_loss, label="pts Loss")
    # 绘制性别损失曲线 , 颜色为红色
    plt.plot(gender_loss, color="red", label="gender Loss")
    # 绘制年龄损失曲线 , 颜色为绿色
    plt.plot(age_loss, color="green", label="age Loss")
    # 绘制总损失曲线 , 颜色为蓝色
    plt.plot(sum_loss, color="blue", label="sum Loss")
    # 曲线说明在左上方
    plt.legend(loc='upper left')
    # 保存图片
    plt.savefig("./loss.png")


# 1、配置信息设置
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=' Project Multi Task Train')
    # 模型输出文件夹
    parser.add_argument('--model_exp', type=str, default='./model_exp',
                        help='model_exp')
    # 模型类型
    parser.add_argument('--model', type=str, default='resnet_34',
                        help='model : resnet_34')
    # landmarks 个数*2（每个关键点有x,y两个坐标）
    parser.add_argument('--num_classes', type=int, default=196,
                        help='num_classes')
    # GPU选择
    parser.add_argument('--GPUS', type=str, default='0',
                        help='GPUS')
    # 训练集标注信息
    parser.add_argument('--train_path', type=str,
                        default='/Users/mac/Documents/02.计算机视觉/03.人脸支付/02.code/facetoPay_edit/face_multi_task/multi_task_samples/label_new/',
                        help='train_path')
    # 初始化学习率
    parser.add_argument('--pretrained', type=bool, default=True,
                        help='imageNet_Pretrain')
    # 模型微调
    parser.add_argument('--fintune_model', type=str,
                        default='none',
                        help='fintune_model')
    # 损失函数定义
    parser.add_argument('--loss_define', type=str, default='wing_loss',
                        help='define_loss')
    # 初始化学习率
    parser.add_argument('--init_lr', type=float, default=2e-4,
                        help='init_learningRate')
    # 学习率权重衰减率
    parser.add_argument('--lr_decay', type=float, default=0.5,
                        help='learningRate_decay')
    # 优化器正则损失权重
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight_decay')
    # 训练每批次图像数量
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch_size')
    # dropout
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout')
    # 训练周期
    parser.add_argument('--epochs', type=int, default=10,
                        help='epochs')
    # 训练数据生成器线程数
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num_workers')
    # 输入模型图片尺寸
    parser.add_argument('--img_size', type=tuple, default=(256, 256),
                        help='img_size')
    # 训练数据生成器是否进行数据扩增
    parser.add_argument('--flag_agu', type=bool, default=True,
                        help='data_augmentation')
    # 模型输出文件夹是否进行清除
    parser.add_argument('--clear_model_exp', type=bool, default=False,
                        help='clear_model_exp')

    # --------------------------------------------------------------------------
    args = parser.parse_args()  # 解析添加参数
    # --------------------------------------------------------------------------
    # 根据配置信息创建训练结果保存的根目录
    # mkdir_的功能是：
    # 存在路径时
    # 若flag_rm = True，则删除文件重新创建
    # 否则不修改
    # 若不存在路劲，则创建路径即可
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)
    loc_time = time.localtime()
    args.model_exp = args.model_exp + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", loc_time) + '/'
    # 根据训练时间创建保存结果的路经
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)
    # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    unparsed = vars(args)
    # 打印参数结果
    for key in unparsed.keys():
        print('{} : {}'.format(key, unparsed[key]))
    # 当前时间
    unparsed['time'] = time.strftime("%Y-%m-%d %H:%M:%S", loc_time)
    # 将配置信息写入到文件照中
    fs = open(args.model_exp + 'train_ops.json', "w", encoding='utf-8')
    # 将配置信息写入到json文件中
    json.dump(unparsed, fs, ensure_ascii=False, indent=1)
    fs.close()
    # 模型训练
    trainer(ops=args)
    print('well done : {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

