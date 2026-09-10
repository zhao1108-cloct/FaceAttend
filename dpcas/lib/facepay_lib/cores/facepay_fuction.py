# -*-coding:utf-8-*-
import os
import numpy as np
import cv2
import torch
from PIL import Image
import random
from lib.facepay_lib.utils.utils import draw_contour,draw_landmarks,plot_box

# 人脸对齐过程
def face_alignment(imgn, eye_left_n, eye_right_n, \
                   desiredLeftEye=(0.34, 0.42), desiredFaceWidth=256, desiredFaceHeight=None):
    """
    :param imgn: 要进行校正的人脸图像文件
    :param eye_left_n: 左眼中心
    :param eye_right_n: 右眼中心
    :param desiredLeftEye:
    :param desiredFaceWidth: 人脸宽度
    :param desiredFaceHeight: 人脸高度
    :return:
    """
    # 若人脸高度为空时，将其设置为人脸宽的大小
    if desiredFaceHeight is None:
        desiredFaceHeight = desiredFaceWidth
    # 左右眼中心
    leftEyeCenter = eye_left_n
    rightEyeCenter = eye_right_n
    # 计算两眼之间的距离
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    # 计算人眼的倾斜角度
    angle = np.degrees(np.arctan2(dY, dX))

    # 计算目标图像中的右眼的坐标
    desiredRightEyeX = 1.0 - desiredLeftEye[0]
    # 计算两眼之间的距离
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    # 计算目标图像中左眼和右眼的距离
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    # 获取尺度变化
    scale = desiredDist / dist
    # 计算原图中两眼的中心位置
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) / 2, (leftEyeCenter[1] + rightEyeCenter[1]) / 2)
    # 计算以两眼中心为旋转中心的旋转矩阵
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
    # 计算矩阵中的平移项
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])
    # 获取仿射变换后的宽高
    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    # 仿射变换获取人脸对齐后的结果
    output = cv2.warpAffine(imgn, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return output


# 扩展人脸框
def refine_face_bbox(bbox, img_shape):
    # 获取图像的高，宽
    height, width, _ = img_shape
    # 获取bbox的坐标
    x1, y1, x2, y2 = bbox
    # 获取宽高
    expand_w = (x2 - x1)
    expand_h = (y2 - y1)
    # 对人脸区域进行扩展
    x1 -= expand_w * 0.12
    y1 -= expand_h * 0.12
    x2 += expand_w * 0.12
    y2 += expand_h * 0.12
    # 量化
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # 对超出图像区域的进行裁剪
    x1 = np.clip(x1, 0, width - 1)
    y1 = np.clip(y1, 0, height - 1)
    x2 = np.clip(x2, 0, width - 1)
    y2 = np.clip(y2, 0, height - 1)
    # 返回扩展后的结果
    return (x1, y1, x2, y2)


def get_faces_batch_attribute(face_multitask_model, face_euler_model, dets, img_raw, use_cuda, face_size=256,
                              vis=False):
    """
    :param face_multitask_model: 多任务模型
    :param face_euler_model: 人脸姿态模型
    :param dets: 人脸检测框
    :param img_raw: 原始图像
    :param use_cuda: 是否使用硬件conda
    :param face_size: 人脸尺寸
    :param vis: 是否进行显示
    :return:
    """
    face_map = np.zeros([112 * 3, 112 * 3, 3]).astype(np.uint8)
    face_map[:, :, 0].fill(205)
    face_map[:, :, 1].fill(205)
    face_map[:, :, 2].fill(205)
    if len(dets) == 0:
        return [], [], [], face_map
    img_align = img_raw.copy()
    # 绘制图像
    image_batch = None
    # 存放人脸框的位置
    r_bboxes = []
    # 存放裁剪的人脸区域
    imgs_crop = []
    # 遍历检测框
    for b in dets:
        # 构建list
        b = list(map(int, b))
        # 获取修正后的结果
        r_bbox = refine_face_bbox((b[0], b[1], b[2], b[3]), img_raw.shape)
        # 将人脸框存入列表中
        r_bboxes.append(r_bbox)
        # 裁剪人脸区域
        img_crop = img_raw[r_bbox[1]:r_bbox[3], r_bbox[0]:r_bbox[2]]
        # 将人脸区域存入列表中
        imgs_crop.append(img_crop)
        # 将face的区域resize成256*256
        img_ = cv2.resize(img_crop, (face_size, face_size), interpolation=cv2.INTER_LINEAR)  # INTER_LINEAR INTER_CUBIC
        # 类型转换
        img_ = img_.astype(np.float32)
        # 归一化
        img_ = (img_ - 128.) / 256.
        # 将通道维放到前面
        img_ = img_.transpose(2, 0, 1)
        # 扩展图像数量维
        img_ = np.expand_dims(img_, 0)
        # 将图像添加到image_batch中
        if image_batch is None:
            image_batch = img_
        else:
            image_batch = np.concatenate((image_batch, img_), axis=0)
    # 多任务：获取关键点，性别，年龄
    landmarks_pre, gender_pre, age_pre = face_multitask_model.predict(image_batch)
    # 获取人脸姿态
    euler_angles = face_euler_model.predict(image_batch)
    # 符合要求，需要识别的人脸图像
    faces_identify = []
    # 符合要求，需要识别的人脸边界框
    faces_identify_bboxes = []
    # 符合要求，需要识别的人脸索引计数器
    faceid_idx = 0
    # 遍历所有检测框的索引
    for i in range(len(dets)):
        # 获取左上角坐标
        x0, y0 = r_bboxes[i][0], r_bboxes[i][1]
        # 获取人脸的宽高
        face_w = r_bboxes[i][2] - r_bboxes[i][0]
        face_h = r_bboxes[i][3] - r_bboxes[i][1]
        # 绘制人脸关键点，获取关键点字典，左右人眼中心，人脸面积
        dict_landmarks, eyes_center, face_area = draw_landmarks(img_raw, landmarks_pre[i], face_w, face_h, x0, y0,
                                                                vis=False)
        # 概率最大类别索引
        gender_max_index = np.argmax(gender_pre[i])
        # 获取人脸姿态的角度
        yaw, pitch, roll = euler_angles[i]
        # 将角度绘制在图像上
        cv2.putText(img_raw, "yaw:{:.1f},pitch:{:.1f},roll:{:.1f}".format(yaw, pitch, roll),
                    (int(r_bboxes[i][0] - 20), int(r_bboxes[i][1] - 30)), cv2.FONT_HERSHEY_DUPLEX, 0.65, (253, 139, 54),
                    5)
        cv2.putText(img_raw, "yaw:{:.1f},pitch:{:.1f},roll:{:.1f}".format(yaw, pitch, roll),
                    (int(r_bboxes[i][0] - 20), int(r_bboxes[i][1] - 30)), cv2.FONT_HERSHEY_DUPLEX, 0.65, (20, 185, 255),
                    1)
        # 将人脸面积绘制在图像上
        cv2.putText(img_raw, "{}".format(int(face_area)), (int(r_bboxes[i][0] - 1), int(r_bboxes[i][3] - 3)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (253, 39, 54), 5)  # face_area
        cv2.putText(img_raw, "{}".format(int(face_area)), (int(r_bboxes[i][0] - 1), int(r_bboxes[i][3] - 3)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (20, 185, 255), 1)
        # 若性别索引为1，则为男性，否则为女性
        if gender_max_index == 1.:
            gender_str = "male"
        else:
            gender_str = "female"
        # 绘制性别，年龄
        plot_box(r_bboxes[i][0:4], img_raw, label="{}, age: {:.1f}".format(gender_str, age_pre[i][0]),
                 color=(255, 90, 90), line_thickness=2)

        # 人脸对齐
        if abs(yaw) < 45.:
            face_align_output = face_alignment(img_align, eyes_center[0], eyes_center[1], desiredLeftEye=(0.365, 0.38),
                                               desiredFaceWidth=112, desiredFaceHeight=None)
        # 当人脸姿态是正面并且人脸面积大于60*60时，说明检测到人脸
        if abs(yaw) < 36. and abs(pitch) < 36. and (face_area > (60 * 60)):
            if vis:
                draw_contour(img_raw, dict_landmarks, vis=True)
            # 获取识别的人脸和区域
            faces_identify.append(Image.fromarray(face_align_output))
            faces_identify_bboxes.append(r_bboxes[i][0:4])
            # 一共可展示九张人脸
            if faceid_idx < 9:
                # 获取每张图像在展示区域的位置
                y1_map, y2_map = int(faceid_idx / 3) * 112, (int(faceid_idx / 3) + 1) * 112
                x1_map, x2_map = int(faceid_idx % 3) * 112, (int(faceid_idx % 3) + 1) * 112
                # 将图像数据填充到face_map中
                face_map[y1_map:y2_map, x1_map:x2_map, :] = face_align_output
                # 绘制矩形框
                cv2.rectangle(face_map, (int(x1_map), int(y1_map)), (int(x2_map), int(y2_map)), (55, 255, 255), 2)
                faceid_idx += 1
        # 若不满足条件则未检测到人脸
        else:
            cv2.putText(img_raw, "bad for face reco", (int(r_bboxes[i][0] - 1), int(r_bboxes[i][3] + 20)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (20, 15, 255), 4)
            cv2.putText(img_raw, "bad for face reco", (int(r_bboxes[i][0] - 1), int(r_bboxes[i][3] + 20)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (220, 185, 25), 1)
    # 返回进行人脸识别的人脸图像，人脸识别框， 扩展后的人脸区域，绘制人脸图像的map
    return faces_identify, faces_identify_bboxes, r_bboxes, face_map
