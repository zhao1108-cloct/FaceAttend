import torch
from face_detect_V5.models.common import DetectMultiBackend
from face_detect_V5.utils.torch_utils import select_device
from face_detect_V5.utils.augmentations import letterbox
from face_detect_V5.utils.general import scale_coords
from face_detect_V5.utils.utils import plot_one_box,non_max_suppression


# 人脸检测过程的实现
class yolo_v5_face_model(object):
    # 初始化
    def __init__(self,
                 weights='./components/face_detect/weights/face_yoloV5_640.pt',
                 img_size=640,
                 conf_thres=0.4,
                 nms_thres=0.4, ):
        """
        :param weights: 模型权重存储位置
        :param img_size: 送入网络中图像的大小
        :param conf_thres: 置信度阈值
        :param nms_thres: nms iou的阈值
        """
        # 获取设备信息
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        # 属性赋值
        # 图像大小
        self.img_size = img_size
        # 类别信息
        self.classes = ["Face"]
        # 类别个数
        self.num_classes = len(self.classes)
        # 阈值信息
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        # anchor的大小:使用在训练阶段聚类出的anchor进行处理
        anchors = [(4, 5), (6, 8), (10, 13), (15, 19), (24, 29), (38, 48), (58, 81), (96, 136), (192, 289)]

        # 模型实例化
        self.model = DetectMultiBackend(weights, device=self.device)
        # 模型设置为 eval，并写入到设备中
        self.model.eval()
        self.model = self.model.to(self.device)

    # 模型预测
    def predict(self, img_, vis):
        with torch.no_grad():
            # 图像预处理：图像大小调整，归一化，通道，类型等
            img = letterbox(img_, (self.img_size, self.img_size), auto=True)[0]
            # Convert
            img = img.transpose((2, 0, 1))[::-1].copy()  # HWC to CHW, BGR to RGB
            im = torch.from_numpy(img).to(self.device)
            # 将img转换为fp16/32
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            # 归一化处理
            im /= 255  # 0 - 255 to 0.0 - 1.0
            # 增加一个bacth维
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            # 图片检测
            pred = self.model(im)
            # NMS 去除冗余，获取检测结果
            detections = non_max_suppression(pred, self.conf_thres, self.nms_thres)[0]  # nms
            # 若检测结果为空，返回空类别
            if (detections is None) or len(detections) == 0:
                return []
            # 检测框是相对于640*640的图像的，调整检测框使其适应于原图像
            # 对框的尺寸进行调整
            detections[:, :4] = scale_coords(im.shape[2:], detections[:, :4], img_.shape).round()
            # 获取最终的检测结果
            output_dict_ = []
            # 遍历得到坐标，置信度，类别概率和类别信息
            for *xyxy, conf, cls_conf, cls in detections:
                label = '%s %.2f' % (self.classes[0], conf)
                x1, y1, x2, y2 = xyxy
                # 坐标信息和置信度
                output_dict_.append((float(x1), float(y1), float(x2), float(y2), float(conf.item())))
                if vis:
                    plot_one_box(xyxy, img_, label=label, color=(0, 175, 255), line_thickness=2)
            return output_dict_
