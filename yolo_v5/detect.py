# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
目标检测
Run inference on images, videos, directories, streams, etc.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
# 参数根据配置信息进行设置
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # 创建保存预测结果的路径
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # 模型加载
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # 获取数据
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 进行模型预测
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    # 计时记录预处理时间，推理时间和后处理时间
    dt, seen = [0.0, 0.0, 0.0], 0
    # 遍历图像数据获取：path, img, img0, self.cap, s
    for path, im, im0s, vid_cap, s in dataset:
        # 计时
        t1 = time_sync()
        # 类型转换并写入到GPU中
        im = torch.from_numpy(im).to(device)
        # 将img转换为fp16/32
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        # 归一化处理
        im /= 255  # 0 - 255 to 0.0 - 1.0
        # 增加一个bacth维
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # 计时
        t2 = time_sync()
        # 预处理时间
        dt[0] += t2 - t1

        # 是否显示
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        # 模型预测
        pred = model(im, augment=augment, visualize=visualize)
        # 计时
        t3 = time_sync()
        # 获取推理时间
        dt[1] += t3 - t2

        # 进行NMS,获取检测结果
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # 获取后处理时间
        dt[2] += time_sync() - t3

        # 预测结果的处理，遍历每一个预测结果
        for i, det in enumerate(pred):  # per image
            # 检测图像增1
            seen += 1
            # 若是摄像头
            if webcam:
                # 获取路径，图像和帧数
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                # 获取路径，图像和帧数
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            # 转换为Path类型
            p = Path(p)  # to Path
            # 图像保存路径
            save_path = str(save_dir / p.name)  # im.jpg
            # txt保存路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            # 获取图像的宽高
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # 获取图像的副本
            imc = im0.copy() if save_crop else im0  # for save_crop
            # 获取图像的预测结果
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # 对框的尺寸进行调整
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # 打印每个类别目标的检测个数
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # 结果保存
                for *xyxy, conf, cls in reversed(det):
                    # 保存TXT文件
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # 将检测结果绘制在图像上
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # 图像显示
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # 保存绘制的预测结果的图像
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # 打印预测时间 (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # 打印结果
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    # 模型权重的位置
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    # 要处理的图像或视频
    parser.add_argument('--source', type=str, default=ROOT / 'data/xwlb.mp4', help='file/dir/URL/glob, 0 for webcam')
    # 数据配置文件
    parser.add_argument('--data', type=str, default=ROOT / 'data/face.yaml', help='(optional) dataset.yaml path')
    # 图像的大小
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    # 预测结果的置信度的阈值
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    # NMS的IOU阈值
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    # 一幅图像中最大的目标检测数量
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    # 设备信息
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 是否显示图片信息
    parser.add_argument('--view-img', action='store_true', help='show results')
    # 是否将检测结果保存在txt中
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # 是否保存置信度
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # 保存截断的检测框
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    # 是否保存图像会视频
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # 类别信息
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    # 是否按照类别进行NMS
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # 预测时是否增强：多尺度和TTA预测
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # 是否进行可视化
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    # 是否更新模型
    parser.add_argument('--update', action='store_true', help='update all models')
    # 预测结果的保存路径
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    # 当前次预测的保存位置
    parser.add_argument('--name', default='exp', help='save results to project/name')
    # 若存在是否创建新的文件夹
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # 绘制框的线宽
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    # 是否隐层类别信息
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    # 是否隐藏置信度结果
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    # 是否进行半精度预测
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # 是否使用opencv中的dnn端
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    # 检测环境是否满足要求
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    # print(opt)
    main(opt)
