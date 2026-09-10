from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
import pdb


##################################  Original Arcface Model #############################################################

# 展平
class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# l2 范数
def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

# SE模块构建
class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

# resnet中残差块的包含3个BN层
class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        """
        :param in_channel: 输入通道数
        :param depth: 输出通道数
        :param stride: 步长
        """
        super(bottleneck_IR, self).__init__()
        # 短连接部分
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        # 残差部分构成
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))
    # 前向传播过程
    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

# 瓶颈模块添加SE模块
class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        # 短连接部分
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        # 残差部分加入se模块
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )
    # 前向传播过程
    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

# 定义元组的子类，可以对元素指明名称，在这里指明瓶颈模块的包括：输入通道数，输出通道数，步长
class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''

# 残差网络中残差模块的构成：第一个残差块进行降采样，步长为2，其他残差块步长为1
def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]

# 设置不同层resnet网络的结构：残差模块的构成
def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks

# 构建残差网络
class Backbone(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir'):
        """
        :param num_layers: 网络层数
        :param drop_ratio: 随机失活比例
        :param mode: 是否添加se模块
        """
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        # 获取网络的残差模块
        blocks = get_blocks(num_layers)
        # reset中残差块设计
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        # 网络输入层
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        # 网络输出层
        self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512))
        # 残差模块部分
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
    # 前向传播
    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)


##################################  Arcface head #############################################################

class Arcface(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1704.06369
    def __init__(self, embedding_size=512, classnum=51332, s=64., m=0.5):
        """
        :param embedding_size: 人脸图像的特征向量
        :param classnum: 人脸分类数，人的个数
        :param s: 半径
        :param m: 夹角差值
        """
        super(Arcface, self).__init__()
        # 类别个数
        self.classnum = classnum
        # 初始化
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        # 夹角差值，默认是0，5
        self.m = m
        # 半径，默认是64
        self.s = s
        # 夹角差值的cos和sin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        #
        self.mm = self.sin_m * m  # issue 1
        # 阈值，避免theta + m >= pi
        self.threshold = math.cos(math.pi - m)

    def forward(self, embbedings, label):
        # 权重的规范化
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # 将特征向量与权重相乘，获取cos值
        cos_theta = torch.mm(embbedings, kernel_norm)
        # 将数值固定在[-1,1]之间，是稳定性更高
        cos_theta = cos_theta.clamp(-1, 1)
        # 求平方
        cos_theta_2 = torch.pow(cos_theta, 2)
        # 获取sin值
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        # cos(theta+t)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # cos(theta)-t
        cond_v = cos_theta - self.threshold
        # 获取cos(theta)-t小于0的位置，mask设为1
        cond_mask = cond_v <= 0
        # 对于cos(theta)-t小于0的位置，也就是theta不在（0，pi）之间，使用下式来替代
        keep_val = (cos_theta - self.mm)  # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0  # a little bit hacky way to prevent in_place operation on cos_theta
        # 获取类别的索引值
        idx_ = torch.arange(0, nB, dtype=torch.long)
        # label是真实值
        # 对于正确类别（1*phi）即公式中的cos(theta + m)，对于错误的类别（1*cosine）即公式中的cos(theta）
        # 这样对于每一个样本，比如[0,0,0,1,0,0]属于第四类，则最终结果为[cosine, cosine, cosine, phi, cosine, cosine]
        # 再乘以半径，经过交叉熵，正好是ArcFace的公式
        output[idx_, label] = cos_theta_m[idx_, label]
        # 乘以s
        output *= self.s
        return output
