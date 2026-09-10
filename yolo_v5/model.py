from models.yolo import Model
import torch
import numpy as np

# 模型配置信息
# 模型实例化
# 导入相关的工具包
from models.yolo import Model
import torch
import numpy as np

# 模型配置信息
cfg = '/Users/mac/Documents/02.计算机视觉/03.人脸支付/02.code/facetoPay/yolo_v5/models/yolov5s.yaml'
# 模型实例化
model = Model(cfg, ch=3, nc=1)
print(model)

#  定义模型输入：NCHW
dummy_input = torch.Tensor(5, 3, 640, 640)
# 模型训练：打印网络输出结果
print("-----------train")
model.train()
for res in model(dummy_input):
    print("res:", np.shape(res))
# 模型预测：打印网络输出结果
print("-----------eval")
model.eval()
inference_out, train_out = model(dummy_input)
print("inference_out:", np.shape(inference_out))
for o in train_out:
    print("train_out:", np.shape(o))


