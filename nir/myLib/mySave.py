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
        print("name",name)
        img0 = Image.open(os.path.join(predictPath,"filter",name))  # PIL Image
        t0 = transform(img0)  # (1, H, W)
        # t0[t0>=0.5]=1
        # t0[t0<0.5]=0
        y_list.append(t0)
    evaluate.analysis(tag, video_id, 255*torch.cat(y_list, dim=0), -1)

def save2img(imgs,path):
    if not os.path.exists(path):os.makedirs(path)
    for i in range(imgs.shape[0]):
        image_array = imgs[i]
        image = Image.fromarray(image_array, mode='L')
        image.save(os.path.join(path, str(i).zfill(5) + '.png'))

def save0(o_scene,tag):
    N, _, H, W = orig.size()
    o_scene = o_scene.view(H, W, N, 1).permute(2, 0, 1, 3).cpu().detach().numpy()
    o_scene = (o_scene * 255).astype(np.uint8)
    save2img(o_scene[:, :, :, 0], os.path.join(outpath, tag))
