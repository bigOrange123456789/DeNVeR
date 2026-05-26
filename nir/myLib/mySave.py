import os
import numpy as np
from itertools import chain
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import re
import torch.nn as nn
import itertools
import math

from nir.model import Siren
from nir.util import get_mgrid, jacobian
from nir.util import Dataset,ToTensor
from nir.myLib.Layer import Layer
from nir.myLib.Decouple import Decouple

from free_cos.main import mainFreeCOS
from eval.eval import Evaluate

evaluate = Evaluate()
def check(predictPath,video_id,tag):
    def natural_sort_key(s):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r'(\d+)', s)]

    # 统一转成单通道，再转 Tensor
    transform = T.Compose([
        T.Grayscale(num_output_channels=1),  # 确保是单通道
        T.ToTensor()  # 转 Tensor，范围 [0,1]
    ])
    y_list=[]
    for name in sorted(os.listdir(os.path.join(predictPath,"filter")),key=natural_sort_key):
        img0 = Image.open(os.path.join(predictPath,"filter",name))  # PIL Image
        t0 = transform(img0)  # (1, H, W)
        # t0[t0>=0.5]=1
        # t0[t0<0.5]=0
        y_list.append(t0)
    evaluate.analysis(tag, video_id, 255*torch.cat(y_list, dim=0), -1)

# def save2img(imgs,path):
#     if not os.path.exists(path):os.makedirs(path)
#     for i in range(imgs.shape[0]):
#         image_array = imgs[i]
#         image = Image.fromarray(image_array, mode='L')
#         image.save(os.path.join(path, str(i).zfill(5) + '.png'))

def save2img(imgs, path):
    if not os.path.exists(path):os.makedirs(path)

    # 如果是 torch tensor，先转 numpy
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.detach().cpu().numpy()

    # 确保是 uint8
    imgs = imgs.astype(np.uint8)
    if imgs.ndim == 4: #[图片数，宽，高，通道]
        for i in range(imgs.shape[0]):
            image_array = imgs[i]      # [H, W, 3]
            image = Image.fromarray(image_array, mode='RGB')
            image.save(os.path.join(path, f'{i:05d}.png'))
    else:
        for i in range(imgs.shape[0]):
            image_array = imgs[i]      # [H, W]
            image = Image.fromarray(image_array, mode='L')
            image.save(os.path.join(path, f'{i:05d}.png'))

def save0(o_scene,tag):
    N, _, H, W = orig.size()
    o_scene = o_scene.view(H, W, N, 1).permute(2, 0, 1, 3).cpu().detach().numpy()
    o_scene = (o_scene * 255).astype(np.uint8)
    save2img(o_scene[:, :, :, 0], os.path.join(outpath, tag))

#################################  后续是将中间日志保存到csv文件  #################################
import csv
def create_csv(path, csv_head):
    if os.path.exists(path):
        os.remove(path) # 如果文件已存在，则删除
    with open(path, 'w', newline='') as f:
        csv_write = csv.writer(f)
        # csv_head = ["good","bad"]
        csv_write.writerow(csv_head)

def write_csv(path, data_row):
    # path  = "aa.csv"
    with open(path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        # data_row = ["1","2"]
        csv_write.writerow(data_row)

import os
# from config import config
def getPath_csv():
    folder_path = os.path.join("log_27",'logs') # folder_path = os.path.join('logs', config.logname + '.log')
    try:# 使用 os.makedirs 创建文件夹，如果父文件夹不存在也会一并创建
        os.makedirs(folder_path, exist_ok=True)  # exist_ok=True 表示如果文件夹已存在，不会抛出异常
    except OSError as error:
        print(f"创建文件夹 '{folder_path}' 时出错: {error}")
    return folder_path


