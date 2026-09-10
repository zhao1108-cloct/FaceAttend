
import cv2
import numpy as np
import random


# 非形变处理:将图像按长宽比resize，然后进行pad
def letterbox(img_, img_size=256, mean_rgb=(128, 128, 128)):
    shape_ = img_.shape[:2]  # shape = [height, width]
    ratio = float(img_size) / max(shape_)  # ratio  = old / new
    new_shape_ = (round(shape_[1] * ratio), round(shape_[0] * ratio))
    dw_ = (img_size - new_shape_[0]) / 2  # width padding
    dh_ = (img_size - new_shape_[1]) / 2  # height padding
    top_, bottom_ = round(dh_ - 0.1), round(dh_ + 0.1)
    left_, right_ = round(dw_ - 0.1), round(dw_ + 0.1)
    # resize img
    img_a = cv2.resize(img_, new_shape_, interpolation=cv2.INTER_LINEAR)

    img_a = cv2.copyMakeBorder(img_a, top_, bottom_, left_, right_, cv2.BORDER_CONSTANT,
                               value=mean_rgb)  # padded square

    return img_a


def img_agu_channel_same(img_):
    """
    将RGB图像转换为灰度图后，将灰度图的结果赋值给每一通道
    :param img_:
    :return:
    """
    img_a = np.zeros(img_.shape, dtype=np.uint8)
    gray = cv2.cvtColor(img_, cv2.COLOR_RGB2GRAY)
    img_a[:, :, 0] = gray
    img_a[:, :, 1] = gray
    img_a[:, :, 2] = gray

    return img_a


# 图像白化
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


# 图像亮度、对比度增强
def contrast_img(img, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols, channels = img.shape
    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels], img.dtype)
    dst = cv2.addWeighted(img, c, blank, 1 - c, b)
    return dst
