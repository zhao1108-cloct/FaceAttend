from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import pickle
import torch
from tqdm import tqdm


# 反归一化
def de_preprocess(tensor):
    return tensor * 0.5 + 0.5

# 获取训练集数据
def get_train_dataset(imgs_folder):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    dataset=ImageFolder(imgs_folder,transform=transform)
    return dataset


# 加载训练集
def get_train_loader(conf):
    dataset = get_train_dataset(conf.datasets_train_path + '/imgs')
    dataloader=DataLoader(dataset=dataset,batch_size=conf.batch_size,shuffle=True)
    return dataloader,len(dataset.classes),dataset.__len__()


if __name__ == "__main__":
    from config import get_config

    # 获取参数配置信息
    conf = get_config()
    # 设置数据的路径
    conf.datasets_train_path = "/Users/mac/Desktop/AI13期人脸支付项目/02.code/datasets/insight_face"
    # 获取送入网络中的数据
    data_loader, class_num, datasets_len = get_train_loader(conf)
    # 打印数据数量和类别个数
    print("train datasets len : {}".format(datasets_len))
    print(" class_num:{} ".format(class_num))
    # 遍历数据进行展示
    for i, (imgs, labels) in enumerate(data_loader):
        # 遍历每个batch中的每一副图像进行展示
        for j in range(conf.batch_size):
            # 展示
            cv2.imshow('results', np.uint8(de_preprocess(imgs[j].permute(1, 2, 0)) * 255.0)[:, :, ::-1])
            cv2.waitKey(0)
            # 打印相应的目标值
            print(labels[j])
    cv2.destroyAllWindows()


