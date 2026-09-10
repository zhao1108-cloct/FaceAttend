# -*- coding: utf-8 -*-
"""
车牌检测入口 —— 直接运行 python main.py 即可
用法：把要检测的图片/视频路径填到下面 SOURCE 里，然后运行。
"""
import detect

if __name__ == '__main__':
    detect.run(
        weights='best.pt',        # 车牌权重文件（已放在 yolo_v5 目录）
        source='1.jpeg',        # 要检测的图片/视频，改成你自己的文件路径
        data='data/face.yaml',    # 类别名会自动从权重里读取，这里只需文件存在即可
        imgsz=(640, 640),         # 输入尺寸
        conf_thres=0.25,          # 置信度阈值：调低→框更多（可能误检），调高→框更少更准
        iou_thres=0.45,           # NMS 的 IoU 阈值，一般不用动
        view_img=True,          # 想实时弹窗显示结果，就取消这一行的注释
    )
