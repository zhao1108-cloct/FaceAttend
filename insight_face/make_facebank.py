#-*-coding:utf-8-*-
# date:2021-04-16
# Author: Eric.Lee
# function: make facebank

import warnings
warnings.filterwarnings("ignore")
import os
import torch
from model import Backbone
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import io
from torchvision import transforms as trans
import torch
from model import l2_norm

def prepare_facebank(path_images, facebank_path, model, device, tta=True):
    '''
    :param path_images:图像路径
    :param facebank_path:保存的位置
    :param model: 人脸特征提取使用的模型
    :param device: 设备信息
    :param tta: 是否获取镜像的特征
    :return:
    embeddings : torch.floattensor类型
        n*512大小
    names : list
        n,保存每个人的姓名.
    '''
    # 将类型转换为tensor和标准化整合在一起进行
    test_transform_ = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # 模型只进行前向传播
    model.eval()
    # 用来存储每个人提取的特征
    embeddings = []
    # 用来存储每个人的名称，不存在的对象名称为Unknown
    names = ['Unknown']
    # 记录人（文件夹）的个数
    idx = 0
    # 遍历每个人的文件夹
    for path in path_images.iterdir():
        # 若是文件，则进行下一次的遍历
        if path.is_file():
            continue
        else:
            # 计数加1
            idx += 1
            # 存放某个对象的特征
            embs = []
            # 遍历文件夹中的所有图像
            for file in path.iterdir():
                # 若不是图像文件，则进行下一次的循环
                if not file.is_file():
                    continue
                else:
                    try:
                        # 读取图像文件
                        img = Image.open(file)
                        print(" {}) {}".format(idx + 1, file))
                    except:
                        continue
                    # 若图像尺寸不是112,则进行尺度的调整
                    if img.size != (112, 112):
                        try:
                            img = img.resize((112, 112))
                        except:
                            continue
                    # 模型预测
                    with torch.no_grad():
                        if tta:
                            # 水平翻转
                            mirror = trans.functional.hflip(img)
                            # 原图像的预测
                            emb = model(test_transform_(img).to(device).unsqueeze(0))
                            # 镜像后图像的预测
                            emb_mirror = model(test_transform_(mirror).to(device).unsqueeze(0))
                            # 将特征求平均后放入embs中
                            embs.append(l2_norm(emb + emb_mirror))
                        else:
                            # 获取原图像的特征，存入embs中
                            embs.append(model(test_transform_(img).to(device).unsqueeze(0)))
        if len(embs) == 0:
            continue
        # 对多幅图像的特征求平均
        embedding = torch.cat(embs).mean(0, keepdim=True)
        # 储存特征
        embeddings.append(embedding)
        # 储存对应的名称
        names.append(path.name)
    # 将特征添加到embeddings中
    embeddings = torch.cat(embeddings)
    # 姓名存储到names中
    names = np.array(names)
    # 将特征保存到pth文件中，将名称保存在names.npy文件中
    torch.save(embeddings, facebank_path + '/facebank.pth')
    np.save(facebank_path + '/names', names)
    return embeddings, names



if __name__ == '__main__':
    # 参数配置
    parser = argparse.ArgumentParser(description='make facebank')
    # 模型
    parser.add_argument("--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
    # 网络深度
    parser.add_argument("--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    # 预训练模型
    parser.add_argument("--finetune_backbone_model", help="finetune_backbone_model", default="/Users/mac/Desktop/AI16计算机视觉/02.人脸支付/02.code/facetoPay_edit/insight_face/model_weights/face_verify-model_ir_se-50.pth", type=str)
    # 人脸仓库中的人脸图像
    parser.add_argument("--facebank_images_path", help="facebank_images_path", default="./facebank_images/", type=str)
    # 人脸仓库
    parser.add_argument("--facebank_path", help="facebank_path", default="./facebank/", type=str)
    # 是否翻转
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",default=False,type=bool)

    args = parser.parse_args()
    # 设备信息
    device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 模型选择
    model_ = Backbone(args.net_depth, 1., args.net_mode).to(device_)
    print('{}_{} model generated'.format(args.net_mode, args.net_depth))
    # 加载预训练模型
    if os.access(args.finetune_backbone_model,os.F_OK):
        model_.load_state_dict(torch.load(args.finetune_backbone_model,map_location='cpu'))
        print("-------->>>   load model : {}".format(args.finetune_backbone_model))
    # 模型预测
    model_.eval()
    # 创建模型仓库
    targets, names = prepare_facebank(Path(args.facebank_images_path), args.facebank_path,model_ ,device_, tta = args.tta)
    print(targets[0].size())
    print(len(targets))
    print(names)
