import glob
import os
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from skimage import color, data, filters, graph, measure, morphology
import preprocess
import subprocess
import argparse

def getVideoId(folder_path):
    print("folder_path",folder_path)
    NUMBER_video = 0
    NUMBER_gt = 0
    my_list = []
    for item in os.listdir(folder_path): #遍历所有患者
        if item==".gitkeep":
            continue
        path = folder_path + "/" + item + "/ground_truth"  # /CVAI-1207LAO44_CRA29"
        gts = os.listdir(path)
        for gts0 in gts: #遍历所有视频的标注
            if len(gts0.split("CATH")) == 1: #只看二分类标注的结果
                NUMBER_video = NUMBER_video + 1 #视频数量
                source_path = os.path.join(folder_path + "/" + item + "/images", gts0)
                NUMBER_gt = NUMBER_gt + len(os.listdir(source_path)) #有标注图像数量
                my_list.append(gts0)
    print("视频总个数:",NUMBER_video)
    print("有标注视频的总帧数:", NUMBER_gt)
    return my_list

from main import Main
import yaml
if __name__ == "__main__":
    # 指定 YAML 文件路径
    script_path = os.path.abspath(__file__)
    ROOT1 = os.path.dirname(script_path)
    file_path = os.path.join(ROOT1, './confs/newConfig.yaml')
    # 打开并读取 YAML 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        lines=getVideoId(config["my"]["datasetPath"])#'job_specs/vessel.txt'
        ##################   预.删除文件   ##################
        file_path= os.path.join(ROOT1, config['my']['filePathRoot'], "experiment_results.csv")
        if os.path.exists(file_path):
            os.remove(file_path)
        ##################   一.分割处理   ##################
        for i in range(len(lines)):
            Main(config,lines[i])
            print("处理进度:"+str(i+1)+"/"+str(len(lines)),lines[i],"/n")
        ##################   二.分析结果   ##################
        from eval.eval import Evaluate
        Evaluate().get() # subprocess.call(f"python ./eval/eval.py", shell=True)

'''
pip install pillow==10.2.0 scikit-image==0.22.0 scipy==1.12.0 matplotlib==3.8.3 opencv-python==4.9.0.80 tensorboard==2.16.2 torch==2.2.1 torchvision==0.17.1 tqdm==4.66.2 hydra-core==1.3.2
export PATH="~/anaconda3/bin:$PATH"
source activate DNVR
python main_batch.py
'''
