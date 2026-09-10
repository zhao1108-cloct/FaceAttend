import os
import numpy as np
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torch.nn as nn

# reference:
# https://github.com/ultralytics/yolov3/blob/master/models.py
# https://github.com/TencentYoutuResearch/ObjectDetection-OneStageDet/blob/master/yolo/vedanet/network/backbone/brick/darknet53.py
# True 查看相关的网络结构
flag_yolo_structure = False


# 构建CBL模块
class Conv2dBatchLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, leaky_slope=0.1):
        '''
        :param in_channels: 输入特征图的通道数
        :param out_channels: 输出特征图的通道数，即卷积核个数
        :param kernel_size: 卷积核大小
        :param stride: 步长
        :param leaky_slope: leak_relu的系数
        '''
        super(Conv2dBatchLeaky, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii / 2) for ii in kernel_size]
            if flag_yolo_structure:
                print('------------------->>>> Conv2dBatchLeaky isinstance')
        else:
            self.padding = int(kernel_size / 2)

        self.leaky_slope = leaky_slope
        # Layer
        # LeakyReLU : y = max(0, x) + leaky_slope*min(0,x)
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(self.leaky_slope, inplace=True)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


# 构建Resunit模块
class ResBlockSum(nn.Module):
    def __init__(self, nchannels):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dBatchLeaky(nchannels, int(nchannels / 2), 1, 1),
            Conv2dBatchLeaky(int(nchannels / 2), nchannels, 3, 1)
        )

    def forward(self, x):
        return x + self.block(x)


# 构建头部分
class HeadBody(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HeadBody, self).__init__()

        self.layer = nn.Sequential(
            Conv2dBatchLeaky(in_channels, out_channels, 1, 1),
            Conv2dBatchLeaky(out_channels, out_channels * 2, 3, 1),
            Conv2dBatchLeaky(out_channels * 2, out_channels, 1, 1),
            Conv2dBatchLeaky(out_channels, out_channels * 2, 3, 1),
            Conv2dBatchLeaky(out_channels * 2, out_channels, 1, 1)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


# 上采样
class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


# 网络的输出层，若进行预测返回预测结果
# default anchors=[(10,13), (16,30), (33,23), (30,61), (62,45), (59,119), (116,90), (156,198), (373,326)]
class YOLOLayer(nn.Module):
    def __init__(self, anchors, nC):
        """
        :param anchors:
        :param nC:
        """
        super(YOLOLayer, self).__init__()

        self.anchors = torch.FloatTensor(anchors)
        self.nA = len(anchors)  # number of anchors (3)
        self.nC = nC  # number of classes
        self.img_size = 0
        if flag_yolo_structure:
            print('init YOLOLayer ------ >>> ')
            print('anchors  : ', self.anchors)
            print('nA       : ', self.nA)
            print('nC       : ', self.nC)
            print('img_size : ', self.img_size)

    def forward(self, p, img_size, var=None):  # p : feature map
        bs, nG = p.shape[0], p.shape[-1]  # batch_size , grid
        if flag_yolo_structure:
            print('bs, nG --->>> ', bs, nG)
        if self.img_size != img_size:
            create_grids(self, img_size, nG, p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, xywh + confidence + classes)
        p = p.view(bs, self.nA, self.nC + 5, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p
        else:  # inference
            io = p.clone()  # inference output
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            io[..., 4:] = torch.sigmoid(io[..., 4:])  # p_conf, p_cls
            io[..., :4] *= self.stride
            if self.nC == 1:
                io[..., 5] = 1  # single-class model
            # flatten prediction, reshape from [bs, nA, nG, nG, nC] to [bs, nA * nG * nG, nC]
            return io.view(bs, -1, 5 + self.nC), p


# 若图像尺寸不是416，调整anchor的生成，输出特征图
def create_grids(self, img_size, nG, device='cpu'):
    # self.nA : len(anchors)  # number of anchors (3)
    # self.nC : nC  # number of classes
    # nG : 输出特征图的大小，与输入图像大小有关
    self.img_size = img_size
    self.stride = img_size / nG
    if flag_yolo_structure:
        print('create_grids stride : ', self.stride)

    # build xy offsets
    grid_x = torch.arange(nG).repeat((nG, 1)).view((1, 1, nG, nG)).float()
    grid_y = grid_x.permute(0, 1, 3, 2)
    self.grid_xy = torch.stack((grid_x, grid_y), 4).to(device)
    if flag_yolo_structure:
        print('grid_x : ', grid_x.size(), grid_x)
        print('grid_y : ', grid_y.size(), grid_y)
        print('grid_xy : ', self.grid_xy.size(), self.grid_xy)

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride  # 基于 stride 的归一化
    # print('self.anchor_vecself.anchor_vecself.anchor_vec:',self.anchor_vec)
    self.anchor_wh = self.anchor_vec.view(1, self.nA, 1, 1, 2).to(device)
    self.nG = torch.FloatTensor([nG]).to(device)


def get_yolo_layer_index(module_list):
    yolo_layer_index = []
    for index, l in enumerate(module_list):
        try:
            a = l[0].img_size and l[0].nG  # only yolo layer need img_size and nG
            yolo_layer_index.append(index)
        except:
            pass
    assert len(yolo_layer_index) > 0, "can not find yolo layer"
    return yolo_layer_index


# ----------------------yolov3------------------------
# 1.1 模型构建
class Yolov3(nn.Module):
    def __init__(self, num_classes=80,anchors=[(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198),(373, 326)]):
        super().__init__()
        pass

    def forward(self, x):
        pass








# ----------------------yolov3 tiny------------------------
# 短连接，不对数据进行任何处理
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Yolov3Tiny(nn.Module):
    def __init__(self, num_classes=80, anchors=[(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)]):
        super(Yolov3Tiny, self).__init__()
        # 6个anchor,输出尺度有两种，还是32倍下采样
        # anchor_mask1是13*13 大物体
        # anchor_mask2是26*26 中等物体
        anchor_mask1 = [i for i in range(len(anchors) // 2, len(anchors), 1)]  # [3, 4, 5]
        anchor_mask2 = [i for i in range(0, len(anchors) // 2, 1)]  # [0, 1, 2]
        # 网络构建，所有的网络层都存放在layerlist中，
        # OrderedDict 是 dict 的子类，其最大特征是可以保持添加的key-valu对的顺序
        # 直接按照网络的构成顺序构建网络
        layer_list = []
        layer_list.append(OrderedDict([
            # layer 0
            ("conv_0", nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_0", nn.BatchNorm2d(16)),
            ("leaky_0", nn.LeakyReLU(0.1)),
            # layer 1
            ("maxpool_1", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
            # layer 2
            ("conv_2", nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_2", nn.BatchNorm2d(32)),
            ("leaky_2", nn.LeakyReLU(0.1)),
            # layer 3
            ("maxpool_3", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
            # layer 4
            ("conv_4", nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_4", nn.BatchNorm2d(64)),
            ("leaky_4", nn.LeakyReLU(0.1)),
            # layer 5
            ("maxpool_5", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
            # layer 6
            ("conv_6", nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_6", nn.BatchNorm2d(128)),
            ("leaky_6", nn.LeakyReLU(0.1)),
            # layer 7
            ("maxpool_7", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
            # layer 8
            ("conv_8", nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_8", nn.BatchNorm2d(256)),
            ("leaky_8", nn.LeakyReLU(0.1)),
        ]))

        layer_list.append(OrderedDict([
            # layer 9
            ("maxpool_9", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
            # layer 10
            ("conv_10", nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_10", nn.BatchNorm2d(512)),
            ("leaky_10", nn.LeakyReLU(0.1)),
            # layer 11
            ('_debug_padding_11', nn.ZeroPad2d((0, 1, 0, 1))),
            ("maxpool_11", nn.MaxPool2d(kernel_size=2, stride=1, padding=0)),
            # layer 12
            ("conv_12", nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_12", nn.BatchNorm2d(1024)),
            ("leaky_12", nn.LeakyReLU(0.1)),
            # layer 13
            ("conv_13", nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)),
            ("batch_norm_13", nn.BatchNorm2d(256)),
            ("leaky_13", nn.LeakyReLU(0.1)),
        ]))

        layer_list.append(OrderedDict([
            # layer 14
            ("conv_14", nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_14", nn.BatchNorm2d(512)),
            ("leaky_14", nn.LeakyReLU(0.1)),
            # layer 15
            ("conv_15",
             nn.Conv2d(in_channels=512, out_channels=len(anchor_mask1) * (num_classes + 5), kernel_size=1, stride=1,
                       padding=0, bias=True)),
        ]))

        # layer 16 13*13特征图检测的结果
        anchor_tmp1 = [anchors[i] for i in anchor_mask1]
        layer_list.append(OrderedDict([("yolo_16", YOLOLayer(anchor_tmp1, num_classes))]))

        # layer 17
        layer_list.append(OrderedDict([("route_17", EmptyLayer())]))

        layer_list.append(OrderedDict([
            # layer 18
            ("conv_18", nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)),
            ("batch_norm_18", nn.BatchNorm2d(128)),
            ("leaky_18", nn.LeakyReLU(0.1)),
            # layer 19
            ("upsample_19", Upsample(scale_factor=2)),
        ]))

        # layer 20
        layer_list.append(OrderedDict([('route_20', EmptyLayer())]))

        layer_list.append(OrderedDict([
            # layer 21
            ("conv_21", nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_21", nn.BatchNorm2d(256)),
            ("leaky_21", nn.LeakyReLU(0.1)),
            # layer 22
            ("conv_22",
             nn.Conv2d(in_channels=256, out_channels=len(anchor_mask2) * (num_classes + 5), kernel_size=1, stride=1,
                       padding=0, bias=True)),
        ]))

        # layer 23 26*26特征图的输出结果
        anchor_tmp2 = [anchors[i] for i in anchor_mask2]
        layer_list.append(OrderedDict([("yolo_23", YOLOLayer(anchor_tmp2, num_classes))]))
        # nn.ModuleList类似于pytho中的list类型，只是将一系列层装入列表
        self.module_list = nn.ModuleList([nn.Sequential(layer) for layer in layer_list])
        # 网络的输出层
        self.yolo_layer_index = get_yolo_layer_index(self.module_list)

    def forward(self, x):
        # 前向传播过程
        img_size = x.shape[-1]
        output = []
        # layer0 to layer8
        x = self.module_list[0](x)
        x_route8 = x
        # layer9 to layer13
        x = self.module_list[1](x)
        x_route13 = x
        # layer14, layer15
        x = self.module_list[2](x)
        # yolo_16 13*13特征图的输出
        x = self.module_list[3][0](x, img_size)
        output.append(x)
        # layer18, layer19
        x = self.module_list[5](x_route13)
        # 特征融合
        x = torch.cat([x, x_route8], 1)
        # layer21, layer22
        x = self.module_list[7](x)
        # yolo_23 26*26特征图的输出
        x = self.module_list[8][0](x, img_size)
        output.append(x)
        # 训练是直接输出两个尺度的结果，
        # train_out: torch.Size([5, 3, 13, 13, 85])
        # train_out: torch.Size([5, 3, 26, 26, 85])
        # 预测时进行拼接
        # inference_out: torch.Size([5, 2535, 85])
        if self.training:
            return output
        else:
            io, p = list(zip(*output))  # inference output, training output
            return torch.cat(io, 1), p


# 1.2 模型测试
if __name__ == "__main__":
    # 定义模型输入，实例化
    input = torch.Tensor(5,3,608,608)
    model = Yolov3Tiny(num_classes=20)
    # 训练阶段
    model.train()
    reses = model(input)
    for res in reses:
        print(np.shape(res))
    # 预测阶段
    model.eval()
    infer_res,train_res = model(input)
    print('预测',np.shape(infer_res))
    for res in train_res:
        print("训练",np.shape(res))