# FaceAttend — 基于多模型协同的实时人脸识别考勤系统

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5c3ee8?logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> 一个端到端的实时人脸识别考勤原型系统，覆盖员工**人脸注册 → 实时打卡识别 → 异常过滤 → 考勤记录管理**全流程。  
> 基于 **检测 + 关键点 + 姿态 + 对齐 + 识别** 四模型协同，支持摄像头 / 视频流输入下的身份核验。

---

## 📑 目录

- [✨ 核心特性](#-核心特性)
- [🧠 技术栈](#-技术栈)
- [🏗️ 系统架构](#️-系统架构)
- [📁 目录结构](#-目录结构)
- [🚀 快速开始](#-快速开始)
- [📖 使用说明](#-使用说明)
- [⚙️ 核心配置说明](#️-核心配置说明)
- [🔬 技术细节](#-技术细节)
- [📊 性能基准](#-性能基准)
- [🎯 Roadmap](#-roadmap)
- [❓ FAQ](#-faq)
- [🤝 贡献](#-贡献)
- [🙏 致谢](#-致谢)
- [📄 License](#-license)
- [📮 联系方式](#-联系方式)

---

## ✨ 核心特性

- 🚀 **端到端实时识别**：从视频流采集到身份输出，单链路全自动化，无需人工干预。
- 🎯 **多模型协同**：YOLOv5 检测 + ResNet34 多任务 + ResNet18 姿态 + InsightFace 识别，**模块化解耦**，单模型可独立升级替换。
- 🛡️ **质量过滤**：自动剔除偏转过大、面积过小的人脸，显著降低误识别率。
- 🔁 **TTA 增强**：水平翻转测试时增强（Test-Time Augmentation），提升复杂姿态下的识别鲁棒性。
- 📦 **底库管理**：内置 `make_facebank` 工具，支持员工多角度照片**批量入库**与**热更新**。
- 🎨 **可视化**：OpenCV 实时绘制人脸框、196 关键点、姿态角度、IoU 触发区、识别置信度。
- ⚙️ **配置驱动**：`facepay.cfg` 统一管理模型路径、阈值、输入尺寸，**不改代码即可调参**。
- 🖥️ **实时可视化**：基于 OpenCV HighGUI 渲染画面，叠加人脸框、196 关键点、三轴姿态角与识别置信度，支持键盘交互。

---

## 🧠 技术栈

| 类别 | 技术 | 关键参数 |
|---|---|---|
| **编程语言** | Python | 3.8+ |
| **深度学习框架** | PyTorch | 1.10+ |
| **目标检测** | YOLOv5（自训练） | 输入 640×640，conf=0.4，NMS IoU=0.45 |
| **多任务网络** | ResNet34（自训练） | 196 关键点 + 性别 + 年龄 |
| **姿态估计** | ResNet18（自训练） | yaw / pitch / roll 三轴回归 |
| **人脸识别** | InsightFace IR-SE-50 | 512 维 L2 归一化嵌入 |
| **人脸对齐** | 仿射变换（基于眼部关键点） | 对齐至 112×112 |
| **图像处理** | OpenCV + NumPy | 图像读取 / 预处理 / 可视化 |
| **可视化 / 交互** | OpenCV HighGUI | `cv2.imshow` 实时渲染 + 键盘交互 |
| **配置管理** | 自研 cfg 体系 | 模型路径 / 阈值 / 设备 |
| **特征存储** | PyTorch tensor + NumPy ndarray | `facebank.pth` + `names.npy` |

---

## 🏗️ 系统架构

### 端到端识别流水线

```
┌─────────────────┐
│ 摄像头/视频流   │
│  (VideoCapture) │
└────────┬────────┘
         │ 每帧
         ▼
┌─────────────────────────────┐
│ ① YOLOv5 人脸检测           │
│   • 640×640 输入             │
│   • conf ≥ 0.4               │
│   • NMS IoU = 0.45           │
│   • 输出：bbox + 置信度       │
└────────┬────────────────────┘
         │ 人脸框
         ▼
┌─────────────────────────────┐
│ ② 人脸框扩展 +12%           │
│   • 防止边缘人脸关键点丢失   │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ ③ ResNet34 多任务           │
│   • 196 关键点               │
│   • 性别（2 类）             │
│   • 年龄（1 维）             │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ ④ ResNet18 姿态估计         │
│   • yaw / pitch / roll      │
│   • 输出 ×90° 反归一化       │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ ⑤ 质量过滤                  │
│   • |yaw| < 36°             │
│   • |pitch| < 36°           │
│   • 人脸面积 > 60×60 px     │
│   • 不通过则直接丢弃         │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ ⑥ 仿射变换对齐              │
│   • 基于双眼坐标             │
│   • 对齐至 112×112          │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ ⑦ InsightFace IR-SE-50      │
│   • 512 维 L2 归一化特征    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ ⑧ 底��比对                  │
│   • facebank.pth 特征矩阵   │
│   • names.npy 身份标签       │
│   • 欧氏距离（threshold=1.2）│
│   • 水平翻转 TTA 增强        │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ ⑨ IoU 触发区域判定         │
│   • 仅当人脸进入打卡区       │
│   • IoU 阈值 = 0.8           │
│   • 输出：姓名 + 置信度      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────┐
│ ⑩ 打卡事件记录          │
│   • OpenCV 可视化       │
│   • 日志 / 数据库写入    │
└─────────────────────────┘
```

---

## 📁 目录结构

```
FaceAttend/
├── dpcas/                            # 主程序包
│   ├── main.py                       # 项目启动入口
│   │
│   ├── applications/                 # 应用层
│   │   └── FacePay_local_app.py      # 本地应用（视频流主流程）
│   │
│   ├── components/                   # 模型组件层（解耦可替换）
│   │   ├── face_detect_V5/           # YOLOv5 人脸检测（实际使用）
│   │   │   └── yolo_V5_face.py
│   │   ├── face_detect/              # YOLOv3 人脸检测（备用）
│   │   ├── face_multi_task/          # ResNet34 多任务（关键点+性别+年龄）
│   │   │   └── face_multi_task_component.py
│   │   ├── face_euler_angle/         # ResNet18 头部姿态
│   │   │   └── face_euler_angle_component.py
│   │   └── insight_face/             # InsightFace IR-SE-50 人脸识别
│   │       ├── face_verify.py
│   │       └── model.py
│   │
│   ├── lib/facepay_lib/              # 核心工具库
│   │   ├── cfg/
│   │   │   └── facepay.cfg           # 全局配置文件
│   │   ├── cores/
│   │   │   └── facepay_fuction.py    # 对齐、批处理、IoU 计算
│   │   ├── make_facebank_tools/
│   │   │   └── make_facebank.py      # 人脸底库构建工具
│   │   └── utils/                    # 工具函数
│   │
│   └── wyw2smodels/                  # 模型权重与底库
│       ├── facebank.pth              # 人脸特征矩阵（512 维 × N）
│       ├── names.npy                 # 身份标签数组
│       └── *.pth                     # 各模型预训练权重
│
├── face_euler_angle/                 # 独立训练模块：姿态估计
├── face_multi_task/                  # 独立训练模块：多任务（关键点+属性）
├── insight_face/                     # 独立训练模块：人脸识别
├── yolo_v3/                          # YOLOv3 训练模块（备用）
└── yolo_v5/                          # YOLOv5 训练模块
```

---

## 🚀 快速开始

### 1. 环境要求

- **Python**：3.8 或更高版本（推荐 3.9 / 3.10）
- **操作系统**：Windows 10/11、Linux、macOS
- **硬件**：
  - CPU 模式：可运行（门禁考勤场景可满足实时性）
  - GPU 模式（推荐）：NVIDIA 显卡 + CUDA（速度提升 5~10 倍）
- **摄像头**：USB 摄像头或内置摄像头（用于实时识别）

### 2. 克隆仓库

```bash
git clone https://github.com/zhao1108-cloct/FaceAttend.git
cd FaceAttend
```

### 3. 安装依赖

推荐使用 Anaconda 创建独立虚拟环境：

```bash
# 创建虚拟环境
conda create -n faceattend python=3.9 -y
conda activate faceattend

# 安装 PyTorch（CPU 版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 或安装 GPU 版本（以 CUDA 11.8 为例）
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖（详见 requirements.txt）
pip install -r requirements.txt
```

### 4. 准备模型权重

将以下模型权重放入 `dpcas/wyw2smodels/` 目录：

| 文件名 | 说明 |
|---|---|
| `face_yoloV5_640.pt` | YOLOv5 人脸检测权重（输入 640×640） |
| `face_multitask-resnet_34_imgsize-256-20210423.pth` | ResNet34 多任务（196 关键点 + 性别 + 年龄）权重 |
| `euler_angle-resnet_18_imgsize_256.pth` | ResNet18 头部姿态（yaw/pitch/roll）权重 |
| `face_verify-model_ir_se-50.pth` | InsightFace IR-SE-50 人脸识别权重 |
| `facebank/facebank.pth` | 人脸底库特征矩阵（512 维 × N） |
| `facebank/names.npy` | 底库身份标签 |

> 权重文件体积较大（单个 40MB ~ 470MB），因此 **未纳入 Git 版本管理**。请从 [Releases 页面](https://github.com/zhao1108-cloct/FaceAttend/releases/latest) 下载后放入该目录；底库请使用 `make_facebank` 工具基于自己的照片生成。

<details>
<summary>📥 权重下载（点击展开）</summary>

| 文件 | 下载链接 |
|---|---|
| `face_yoloV5_640.pt` | [Releases](https://github.com/zhao1108-cloct/FaceAttend/releases/latest/download/face_yoloV5_640.pt) |
| `face_multitask-resnet_34_imgsize-256-20210423.pth` | [Releases](https://github.com/zhao1108-cloct/FaceAttend/releases/latest/download/face_multitask-resnet_34_imgsize-256-20210423.pth) |
| `euler_angle-resnet_18_imgsize_256.pth` | [Releases](https://github.com/zhao1108-cloct/FaceAttend/releases/latest/download/euler_angle-resnet_18_imgsize_256.pth) |
| `face_verify-model_ir_se-50.pth` | [Releases](https://github.com/zhao1108-cloct/FaceAttend/releases/latest/download/face_verify-model_ir_se-50.pth) |

</details>

### 5. 构建人脸底库

`make_facebank.py` 会从「每人一个子文件夹」的图片目录批量生成底库：

```
employee_photos/
├── zhangli/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.jpg
├── lisi/
│   └── ...
```

脚本为原型工具，路径在文件内配置（暂未提供命令行参数）：

```python
# dpcas/lib/facepay_lib/make_facebank_tools/make_facebank.py
path_images   = "./images/"      # 员工照片目录
facebank_path = "./facebank/"    # 底库输出目录
```

执行 `python make_facebank.py` 后，对每人的多张照片提取特征 → 求均值 → L2 归一化，输出两个文件：

| 输出 | 说明 |
|---|---|
| `facebank.pth` | 特征矩阵（N × 512） |
| `names.npy` | 身份标签数组，首项固定为 `Unknown` |

把生成的 `facebank/` 目录放到 `dpcas/wyw2smodels/facebank/` 即可（与 `facepay.cfg` 中 `facebank_path` 保持一致）。

### 6. 启动应用

```bash
cd dpcas
python main.py
```

---

## 📖 使用说明

### 运行方式

程序以**视频文件 / 摄像头流**作为输入源（`cv2.VideoCapture(video_path)`）。默认入口 `dpcas/main.py` 中指定了示例视频：

```python
main_facePay(video_path='zhangli.mp4', cfg_file=cfg_file)
```

如需切换为摄像头实时输入，把 `video_path` 改为 `0` 即可：

```python
main_facePay(video_path=0, cfg_file=cfg_file)
```

### 画面元素说明

运行后 OpenCV 窗口会实时叠加显示：

- 🟢 **人脸框**：YOLOv5 检测到的人脸包围框
- 📍 **关键点**：196 个面部关键点（眼眶 / 鼻尖 / 嘴角等）
- 🧭 **姿态角度**：yaw / pitch / roll 三轴实时数值
- 🎯 **识别区域**：距画面边界 (150, 60) 的矩形触发区
- 📝 **识别结果**：姓名 + 欧氏距离距离值
- 🖼️ **辅助面板**：人脸对齐图（112×112）与商品 SKU 面板（原型演示用）

### 交互与输出

- 按 `ESC` 键退出主循环
- 渲染画面通过 `cv2.VideoWriter` 同步录制，输出 `recording_YYYY-MM-DD_HH-MM-SS.mp4`

### 识别触发机制

帧内检测到的人脸需与「识别区域」的 IoU **> 0.8** 才会送入识别网络，避免画面边缘的无效计算：

| 判定 | 条件 |
|---|---|
| 偏转过大 | \|yaw\| ≥ 36° 或 \|pitch\| ≥ 36° → 丢弃 |
| 人脸过小 | 面积 ≤ 60×60 像素 → 丢弃 |
| 未进入识别区 | 与识别区 IoU ≤ 0.8 → 不做识别 |

> 📌 **说明**：本仓库是**算法原型**，核心价值在于「检测 → 关键点/姿态 → 质量过滤 → 对齐 → 识别」这条推理链路；**考勤业务层**（员工信息管理、打卡记录落库、统计报表）未包含在当前代码中，属于 [Roadmap](#-roadmap) 规划范围。

---

## ⚙️ 核心配置说明

主配置文件：`dpcas/lib/facepay_lib/cfg/facepay.cfg`

```ini
# ---- 人脸检测模型 ----
detect_model_path        = ./wyw2smodels/face_yoloV5_640.pt
detect_input_size        = 640
detect_conf_thres        = 0.4
detect_nms_thres         = 0.45

# ---- 人脸识别模型（InsightFace IR-SE-50）----
face_verify_backbone_path = ./wyw2smodels/face_verify-model_ir_se-50.pth
facebank_path             = ./wyw2smodels/facebank
face_verify_threshold     = 1.2

# ---- 人脸多任务模型（196 关键点 + 性别 + 年龄）----
face_multitask_model_path = ./wyw2smodels/face_multitask-resnet_34_imgsize-256-20210423.pth
face_multitask_model_arch = resnet34

# ---- 头部姿态模型（yaw / pitch / roll）----
face_euler_model_path     = ./wyw2smodels/euler_angle-resnet_18_imgsize_256.pth
```

**关键参数说明**

| 参数 | 含义 | 取值 |
|---|---|---|
| `detect_input_size` | 检测网络输入分辨率 | 640×640 |
| `detect_conf_thres` | 检测置信度阈值 | 0.4 |
| `detect_nms_thres` | NMS IoU 阈值 | 0.45 |
| `face_verify_threshold` | 人脸比对欧氏距离阈值 | 1.2 |
| `facebank_path` | 底库目录（内含 `facebank.pth` + `names.npy`） | — |

> 质量过滤阈值（\|yaw\|<36°、\|pitch\|<36°、人脸面积>60×60）与识别区 IoU 阈值（0.8）以常量形式定义在 `dpcas/lib/facepay_lib/cores/facepay_fuction.py` 与 `dpcas/applications/FacePay_local_app.py` 中。

---

## 🔬 技术细节

### 为什么选 YOLOv5 而不是 YOLOv8？

- YOLOv5 在**人脸检测**这一细粒度任务上已足够鲁棒
- 工程部署生态成熟，社区文档丰富
- 模型权重轻量化（YOLOv5s 约 14 MB），CPU 可跑
- 注：项目中也保留了 YOLOv3 的实现（`face_detect/`），作为对照实验

### 为什么选 InsightFace IR-SE-50？

- IR-SE-50 = Improved Residual + SE（Squeeze-and-Excitation）注意力模块
- 在 LFW / CFP-FP / AgeDB-30 等公开基准上准确率领先
- 512 维特征向量在**精度 vs 存储**之间平衡最佳
- 支持 L2 归一化后直接用欧氏距离 / 余弦相似度比对

### 为什么要做质量过滤？

- 偏转过大的人脸会引入**特征漂移**，导致识别准确率下降
- 过滤后进入识别链路的样本质量更高，可减少无效计算与误匹配风险
- 滤掉的低质量样本不会进入识别链路，**降低无效计算**

### 为什么用仿射变换对齐？

- InsightFace 训练时使用的是**对齐后的人脸**
- 推理时不对齐会导致特征空间不一致，识别准确率断崖式下降
- 基于**双眼坐标**做仿射变换是最经典、鲁棒的对齐方法

### 水平翻转 TTA 的原理

```python
# 原始人脸特征
feat_orig = model(face_image)

# 水平翻转后的人脸特征
face_flip = torch.flip(face_image, dims=[-1])
feat_flip = model(face_flip)

# 两者融合（提升鲁棒性）
feat_final = (feat_orig + feat_flip) / 2
feat_final = F.normalize(feat_final, p=2, dim=-1)
```

对**对称性脸**（左右基本对称的人脸）效果提升明显；对极端侧脸提升有限（已被质量过滤剔除）。

---

## 📊 性能说明

> ⚠️ 本仓库**未包含基准测试数据**。整条流水线包含 4 个串行推理模型（检测 → 多任务 → 姿态 → 识别），实际 FPS 强依赖硬件、输入分辨率与画面中的人脸数量，请以**本机实测**为准。建议自测方式：

```bash
cd dpcas
python main.py     # 终端会持续打印每帧处理耗时
```

**影响性能的主要因素**

| 因素 | 影响 |
|---|---|
| `detect_input_size` | 640 → 416 时，检测输入像素量降至约 42%（面积比） |
| 画面人脸数量 | 多任务 / 姿态 / 识别按人脸逐张推理，人脸越多耗时越长 |
| 质量过滤命中率 | 未通过过滤的人脸可跳过对齐与识别，减少无效计算 |
| 计算设备 | 代码内 `use_cuda` 参数可切换 CPU / GPU，GPU 有数倍加速 |
| 触发策略 | 与识别区 IoU ≤ 0.8 的人脸不做识别，规避边缘无效计算 |

---

## 🎯 Roadmap

- [ ] **支持口罩人脸识别**（遮挡场景刚需）
- [ ] **支持活体检测**（防照片 / 视频攻击）
- [ ] **Docker 一键部署**（Web 后端 + REST API）
- [ ] **Web 管理后台**（员工信息管理、底库维护）
- [ ] **多线程 / 多路摄像头并发**
- [ ] **模型量化（INT8 / TensorRT）加速**
- [ ] **考勤业务层**：打卡记录落库、重复打卡去重、统计报表导出
- [ ] **接入企业微信 / 钉钉打卡 API**

---

## ❓ FAQ

**Q1：项目里的人脸数据是如何处理的？是否会上传云端？**  
A：完全本地化处理，不上传任何人脸数据。`facebank.pth` 是脱敏后的特征向量，无法还原人脸图片。

**Q2：能识别戴口罩的人脸吗？**  
A：当前版本未做口罩适配。Roadmap 已列入，未来会基于口罩专用数据集微调 InsightFace。

**Q3：InsightFace IR-SE-50 和 ArcFace 的关系？**  
A：InsightFace 团队提出的 ArcFace 损失函数是该项目的核心贡献之一，IR-SE-50 是常用的 Backbone。两者是 **"模型 + 损失函数"** 的关系。

**Q4：CPU 模式能满足生产环境需求吗？**  
A：单路门禁考勤场景可以。多路（>4 路）并发建议 GPU 部署 + TensorRT 加速。

**Q5：如何添加新员工？**  
A：把该员工的照片按「一人一个子文件夹」放入图片目录，重跑 `make_facebank.py` 重新生成底库，再重启程序加载新底库即可。

---

## 🤝 贡献

欢迎提交 Issue 和 PR！

- 🐛 **Bug 报告**：[Issues](../../issues)
- 💡 **功能建议**：[Discussions](../../discussions)
- 🔧 **代码贡献**：Fork → 修改 → Pull Request

---

## 🙏 致谢

本项目参考并借鉴了以下开源项目 / 论文：

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) — 人脸检测基线
- [InsightFace](https://github.com/deepinsight/insightface) — 人脸识别 Backbone
- [ArcFace: Additive Angular Margin Loss](https://arxiv.org/abs/1801.07698) — 损失函数设计

---

## 📄 License

本项目基于 **MIT License** 开源 — 详见 [LICENSE](LICENSE) 文件。

---

## 📮 联系方式

| 渠道 | 联系方式 |
|---|---|
| 作者 | 赵昱焜 |
| 邮箱 | 1787435040@qq.com |
| GitHub | [@zhao1108-cloct](https://github.com/zhao1108-cloct) |

> 简历 / 项目合作请优先通过邮箱联系。

---

<p align="center">
  ⭐ 如果这个项目对您有帮助，欢迎 Star！<br>
  📌 有任何问题欢迎提 Issue 或 PR<br>
  <strong>—— 让每一次打卡都精准、高效、可靠</strong>
</p>