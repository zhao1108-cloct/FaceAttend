#-*-coding:utf-8-*-
# date:2020-10-19.7.23.24
# Author: Eric.Lee
# function: main

import os
import argparse
import warnings
warnings.filterwarnings("ignore")
import sys
# 添加模型组件路径
sys.path.append("./components/")


if __name__ == '__main__':
    cfg_file = "./lib/facepay_lib/cfg/facepay.cfg"
    # 加载 face pay 应用
    from applications.FacePay_local_app import main_facePay
    # 加载 face pay  应用
    main_facePay(video_path='zhangli.mp4', cfg_file = cfg_file)
    print(" well done ~")
