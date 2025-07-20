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

from free_cos.main import mainFreeCOS
from eval.eval import Evaluate

from nir.model import Siren
from nir.util import get_mgrid, jacobian
from nir.util import Dataset,ToTensor

from nir.myLib.Decouple import Decouple
from nir.myLib.mySave import check,save2img,save0
#####################################################################
from nir.myLib.Layer import Layer
class Layer_rigid(nn.Module):
    def __init__(self,useGlobal=True,useLocal=True,useMatrix=True,useDeformation=False,
                 deformationSize=3*0.004,hidden_features=128):
        super().__init__()
        self.useGlobal=useGlobal
        self.useLocal=useLocal
        self.useMatrix=useMatrix
        self.useDeformation=useDeformation
        self.deformationSize=deformationSize
        self.f_2D = Siren(in_features=2, out_features=1,
                          hidden_features=hidden_features, hidden_layers=4,
                          outermost_linear=True)
        self.f_2D.cuda()
        self.parameters = [self.f_2D.parameters()]
        if self.useGlobal:
            self.g_global = Siren(in_features=1, out_features=4 if self.useMatrix else 2,
                                  # hidden_features=16, hidden_layers=2,
                                  hidden_features=hidden_features, hidden_layers=4,
                                  outermost_linear=True)
            self.parameters.append(self.g_global.parameters())
            self.g_global.cuda()
        if self.useLocal:
            self.g_local = Siren(in_features=3, out_features=2,
                                 hidden_features=hidden_features, hidden_layers=4,
                                 outermost_linear=True)
            self.g_local.cuda()
            self.parameters.append(self.g_local.parameters())
    def forward(self,xyt,stage):
        t = xyt[:, [-1]]
        h_global = self.g_global(t)
        ############################################################
        # xy_ = xyt[:, :-1] + h_global #+ h_local
        if self.useMatrix:
            c =h_global
            u = xyt[:, 0]
            v = xyt[:, 1]
            if c.shape[1]==6:
                new_u = c[:,0] * u + c[:,1] * u + c[:,2]
                new_v = c[:,3] * v + c[:,4] * v + c[:,5]
            else:#4
                # 提取参数 (忽略可能的第五个参数)
                tx = c[:, 0]  # X轴位移
                ty = c[:, 1]  # Y轴位移
                rotation = torch.tensor(0)#c[:, 2]  # 旋转角度(弧度)
                scale = torch.tensor(1)#c[:, 3]  # 放缩因子
                # 计算旋转和放缩后的坐标
                cos_theta = torch.cos(rotation)
                sin_theta = torch.sin(rotation)
                # 向量化计算所有点
                u=scale*u
                v=scale*v
                new_u = (u * cos_theta - v * sin_theta) + tx
                new_v = (u * sin_theta + v * cos_theta) + ty
            # 组合成新坐标张量
            new_uv = torch.stack([new_u, new_v], dim=1)
            xy_ = new_uv
        else: #不使用变换矩阵
            xy_ = xyt[:, :-1] + h_global
        ############################################################
        if stage==0:#纹理学习，不分析局部位移
            h_local = torch.tensor(0.0)# h_local = self.g_local(xyt) if self.useLocal else torch.tensor(0.0)
        else:#只分析局部位移，不学习整体运动和纹理
            h_local = self.g_local(xyt)
            h_local = 2 * torch.sigmoid(h_local) - 1
            h_local = h_local * self.deformationSize
            xy_ = xy_ + h_local
        color = torch.sigmoid(self.f_2D(xy_))
        return color,{
            "xy_":xy_,
            "h_local":h_local,
            "h_global":h_global.clone().detach()
        }

#####################################################################
