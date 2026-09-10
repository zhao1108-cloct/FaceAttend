import os
import os
import numpy as np
import cv2
import torch
from PIL import Image
import random
"""Parses the data configuration file"""
def parse_data_cfg(path):
    print('data_cfg ： ',path)
    options = dict()
    with open(path, 'r',encoding='UTF-8') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


def compute_face_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        #return (intersect / (sum_area - intersect))*1.0
        return (intersect / (S_rec1 + 1e-6))*1.0


def compute_iou(rec1, rec2):
    """
    计算IOU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        # return (intersect / (sum_area - intersect))*1.0
        return (intersect / (S_rec1 + 1e-6)) * 1.0


# 绘制人脸关键点 98个关键点
def draw_landmarks(img, output, face_w, face_h, x0, y0, vis=False):
    """
    :param img:  图像数据
    :param output: 关键点信息
    :param face_w: 人脸宽
    :param face_h: 人脸高
    :param x0: 左上角x
    :param y0: 左上角y
    :param vis: 是否进行显示
    :return:
    """
    img_width = img.shape[1]
    img_height = img.shape[0]
    # 关键点
    dict_landmarks = {}
    # 人眼
    eyes_center = []
    x_list = []
    y_list = []
    for i in range(int(output.shape[0] / 2)):
        # 获取关键点在图像中的位置
        x = output[i * 2 + 0] * float(face_w) + x0
        y = output[i * 2 + 1] * float(face_h) + y0
        #
        x_list.append(x)
        y_list.append(y)

        if 41 >= i >= 33:
            if 'left_eyebrow' not in dict_landmarks.keys():
                dict_landmarks['left_eyebrow'] = []
            dict_landmarks['left_eyebrow'].append([int(x), int(y), (0, 255, 0)])
            if vis:
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
        elif 50 >= i >= 42:
            if 'right_eyebrow' not in dict_landmarks.keys():
                dict_landmarks['right_eyebrow'] = []
            dict_landmarks['right_eyebrow'].append([int(x), int(y), (0, 255, 0)])
            if vis:
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
        elif 67 >= i >= 60:
            if 'left_eye' not in dict_landmarks.keys():
                dict_landmarks['left_eye'] = []
            dict_landmarks['left_eye'].append([int(x), int(y), (255, 55, 255)])
            if vis:
                cv2.circle(img, (int(x), int(y)), 2, (255, 0, 255), -1)
        elif 75 >= i >= 68:
            if 'right_eye' not in dict_landmarks.keys():
                dict_landmarks['right_eye'] = []
            dict_landmarks['right_eye'].append([int(x), int(y), (255, 55, 255)])
            if vis:
                cv2.circle(img, (int(x), int(y)), 2, (255, 0, 255), -1)
        elif 97 >= i >= 96:
            eyes_center.append((x, y))
            if vis:
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
        elif 54 >= i >= 51:
            if 'bridge_nose' not in dict_landmarks.keys():
                dict_landmarks['bridge_nose'] = []
            dict_landmarks['bridge_nose'].append([int(x), int(y), (0, 170, 255)])
            if vis:
                cv2.circle(img, (int(x), int(y)), 2, (0, 170, 255), -1)
        elif 32 >= i >= 0:
            if 'basin' not in dict_landmarks.keys():
                dict_landmarks['basin'] = []
            dict_landmarks['basin'].append([int(x), int(y), (255, 30, 30)])
            if vis:
                cv2.circle(img, (int(x), int(y)), 2, (255, 30, 30), -1)
        elif 59 >= i >= 55:
            if 'wing_nose' not in dict_landmarks.keys():
                dict_landmarks['wing_nose'] = []
            dict_landmarks['wing_nose'].append([int(x), int(y), (0, 255, 255)])
            if vis:
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 255), -1)
        elif 87 >= i >= 76:
            if 'out_lip' not in dict_landmarks.keys():
                dict_landmarks['out_lip'] = []
            dict_landmarks['out_lip'].append([int(x), int(y), (255, 255, 0)])
            if vis:
                cv2.circle(img, (int(x), int(y)), 2, (255, 255, 0), -1)
        elif 95 >= i >= 88:
            if 'in_lip' not in dict_landmarks.keys():
                dict_landmarks['in_lip'] = []
            dict_landmarks['in_lip'].append([int(x), int(y), (50, 220, 255)])
            if vis:
                cv2.circle(img, (int(x), int(y)), 2, (50, 220, 255), -1)
        else:
            if vis:
                cv2.circle(img, (int(x), int(y)), 2, (255, 0, 255), -1)
    # 人脸面积
    face_area = (max(x_list) - min(x_list)) * (max(y_list) - min(y_list))
    return dict_landmarks, eyes_center, face_area


# 绘制眼睛，面颊和鼻子
def draw_contour(image, dict, vis=False):
    x0 = 0  # 偏置
    y0 = 0

    for key in dict.keys():
        # print(key)
        _, _, color = dict[key][0]

        if 'left_eye' == key:
            eye_x = np.mean([dict[key][i][0] + x0 for i in range(len(dict[key]))])
            eye_y = np.mean([dict[key][i][1] + y0 for i in range(len(dict[key]))])
            if vis:
                cv2.circle(image, (int(eye_x), int(eye_y)), 3, (255, 255, 55), -1)
        if 'right_eye' == key:
            eye_x = np.mean([dict[key][i][0] + x0 for i in range(len(dict[key]))])
            eye_y = np.mean([dict[key][i][1] + y0 for i in range(len(dict[key]))])
            if vis:
                cv2.circle(image, (int(eye_x), int(eye_y)), 3, (255, 215, 25), -1)

        if 'basin' == key or 'wing_nose' == key:
            pts = np.array([[dict[key][i][0] + x0, dict[key][i][1] + y0] for i in range(len(dict[key]))], np.int32)
            if vis:
                cv2.polylines(image, [pts], False, color, thickness=2)

        else:
            points_array = np.zeros((1, len(dict[key]), 2), dtype=np.int32)
            for i in range(len(dict[key])):
                x, y, _ = dict[key][i]
                points_array[0, i, 0] = x + x0
                points_array[0, i, 1] = y + y0

            # cv2.fillPoly(image, points_array, color)
            if vis:
                cv2.drawContours(image, points_array, -1, color, thickness=2)


def plot_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 2)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [185, 195, 190], thickness=tf, lineType=cv2.LINE_AA)
