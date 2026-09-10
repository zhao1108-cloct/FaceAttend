# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
导出工具
Export a YOLOv5 PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit
"""

import os
import sys
from pathlib import Path

import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def export_formats():
    # YOLOv5 export formats
    x = [['PyTorch', '-', '.pt', True],
         ['TorchScript', 'torchscript', '.torchscript', True],
         ['ONNX', 'onnx', '.onnx', True],
         ['OpenVINO', 'openvino', '_openvino_model', False],
         ['TensorRT', 'engine', '.engine', True],
         ['CoreML', 'coreml', '.mlmodel', False],
         ['TensorFlow SavedModel', 'saved_model', '_saved_model', True],
         ['TensorFlow GraphDef', 'pb', '.pb', True],
         ['TensorFlow Lite', 'tflite', '.tflite', False],
         ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False],
         ['TensorFlow.js', 'tfjs', '_web_model', False]]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'GPU'])
