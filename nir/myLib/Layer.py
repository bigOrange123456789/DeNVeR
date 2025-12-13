import os
import numpy as np
from itertools import chain
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import re

from nir.model import Siren
from nir.util import get_mgrid, jacobian
from nir.util import Dataset,ToTensor

import torch.nn as nn
import torch.nn.functional as F
class Tex2D(nn.Module):
    def __init__(self,size):
        # super(Tex2D, self).__init__()
        super().__init__()
        # 使用nn.Parameter创建叶子张量
        # self.image = nn.Parameter(torch.zeros(1,512, 512))
        self.image = nn.Parameter(torch.zeros(1,size, size)).cuda()
    
    # def forward(self):
    #     # forward可以返回运算结果
    #     return self.image 
    def forward(self, coords, scale=1.0):
        """
        从图像中采样指定坐标的像素值
        
        Args:
            coords: 形状为 (N, 2) 的张量，坐标范围 [-2, 2]
                    coords[:, 0] 是 x 坐标（横坐标），范围 [-2, 2]
                    coords[:, 1] 是 y 坐标（纵坐标），范围 [-2, 2]
        
        Returns:
            采样到的像素值，形状为 (N, channels)
        """
        # if not self.useTex2D:
        #     return 0
        # 1. 将坐标从 [-2, 2] 范围归一化到 [-1, 1] 范围
        #    PyTorch的grid_sample期望坐标在[-1, 1]范围内
        normalized_coords = coords/scale #/ 2.0
        
        # 2. 添加batch维度和最后一个维度（grid_sample需要的格式）
        #    原始形状: (N, 2) -> 变换后: (1, N, 1, 2)
        grid = normalized_coords.unsqueeze(0).unsqueeze(2)
        
        # 3. 添加batch维度到图像张量
        #    原始形状: (C, H, W) -> 变换后: (1, C, H, W)
        image_batch = self.image.unsqueeze(0)
        
        # 4. 使用grid_sample进行双线性插值采样
        #    align_corners=True 确保坐标映射正确
        sampled = F.grid_sample(
            image_batch,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        # 5. 调整输出形状: (1, C, N, 1) -> (N, C)
        return sampled.squeeze(0).squeeze(-1).permute(1, 0)
class Layer(nn.Module):
    def __init__(self,useGlobal=True,useLocal=True,useMatrix=True,useDeformation=False,
                 deformationSize=8,hidden_features=128,
                 config={},
                 ):
        super().__init__()
        hidden_layers_map=4
        hidden_layers_global=2#3
        hidden_layers_local=4
        hidden_features_map=128
        hidden_features_global=128
        hidden_features_local=128
        if not config is None:
            hidden_layers_map = config["hidden_layers_map"]   
            hidden_layers_global = config["hidden_layers_global"]
            hidden_layers_local = config["hidden_layers_local"] 
            hidden_features_map = config["hidden_features_map"] 
            hidden_features_global = config["hidden_features_global"] 
            hidden_features_local = config["hidden_features_local"] 
        ####################################
        self.useGlobal=useGlobal
        self.useLocal=useLocal
        self.useMatrix=useMatrix
        self.useDeformation=useDeformation
        self.deformationSize=deformationSize
        self.f_2D = Siren(in_features=2, out_features=1,
                          hidden_features=hidden_features_map,#hidden_features, 
                          hidden_layers=hidden_layers_map,#4,
                          outermost_linear=True)
        self.f_2D.cuda()
        self.parameters = [self.f_2D.parameters()]
        if False:
            self.tex2D = Tex2D(512)
            self.parameters.append(self.tex2D.parameters())
        if self.useGlobal:
            self.g_global = Siren(in_features=1, out_features=4 if self.useMatrix else 2,
                                  hidden_features=hidden_features_global,#16, 
                                  hidden_layers=hidden_layers_global,#2,
                                  outermost_linear=True)
            self.parameters.append(self.g_global.parameters())
            self.g_global.cuda()
        if self.useLocal:
            self.g_local = Siren(in_features=3, out_features=2,
                                 hidden_features=hidden_features_local,#hidden_features, 
                                 hidden_layers=hidden_layers_local,#4,
                                 outermost_linear=True)
            self.g_local.cuda()
            self.parameters.append(self.g_local.parameters())
    def forward(self,xyt):
        h_local = self.g_local(xyt) if self.useLocal else torch.tensor(0.0)
        if self.useDeformation:
            h_local=2*torch.sigmoid(h_local)-1
            h_local=h_local*self.deformationSize
        if self.useMatrix and self.useGlobal:
            c =self.g_global(xyt[:, [-1]])
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
            xy_ = new_uv + h_local
        else:
            h_global = self.g_global(xyt[:, [-1]]) if self.useGlobal else 0
            xy_ = xyt[:, :-1] + h_global + h_local
        color = torch.sigmoid(self.f_2D(xy_))
        # color = torch.sigmoid(self.tex2D(xy_))
        return color,{
            "xy_":xy_,
            "h_local":h_local
        }

class PositionalEncoder(nn.Module):
    """位置编码器，将低维输入映射到高维空间。"""
    def __init__(self, input_dim, num_freqs, include_input=True):
        super().__init__()
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.output_dim = input_dim * (2 * num_freqs + include_input)
        self.freq_bands = 2.0 ** torch.linspace(0, num_freqs-1, num_freqs)

    def forward(self, x):
        encoded = [x] if self.include_input else []
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * torch.pi * x))
            encoded.append(torch.cos(freq * torch.pi * x))
        return torch.cat(encoded, dim=-1)
    
class Layer_video(nn.Module): #用来拟合视频的模块
    def __init__(
                self,
                config={
                     "hidden_features":512,
                     "hidden_layers":4,
                     "use_residual":True,
                     "posEnc":{
                        "num_freqs_pos":10,
                        "num_freqs_time":4,
                     },
                     
                },
            ):
        super().__init__()
        self.use_posEnc="posEnc" in config and config["posEnc"]
        # print("self.use_posEnc",self.use_posEnc)
        in_features_num = 3
        if self.use_posEnc:
            # 对空间位置进行编码
            self.pos_encoder = PositionalEncoder(2, num_freqs=config["posEnc"]["num_freqs_pos"])
            # 对时间进行编码
            self.time_encoder = PositionalEncoder(1, num_freqs=config["posEnc"]["num_freqs_time"])
            in_features_num = (
                self.pos_encoder.output_dim +
                self.time_encoder.output_dim
                )
        self.f2 = Siren(    
            in_features=in_features_num,#3, 
            out_features=1, 
            hidden_features=config["hidden_features"],#512,#hidden_features,
            hidden_layers=config["hidden_layers"],#4, 
            outermost_linear=True,
            use_residual=config["use_residual"]
            ) 
        self.f2.cuda()
        self.parameters = [self.f2.parameters()]

    def forward(self,xyt):
        if self.use_posEnc:
            xy_ = xyt[:, :-1]
            t = xyt[:, [-1]]
            x_encoded = self.pos_encoder(xy_)
            t_encoded = self.time_encoder(t) # 确保时间有正确的维度
            combined = torch.cat([x_encoded, t_encoded], dim=-1)
        else:
            combined = xyt
        # print("xy_",xy_.shape)
        # print("t",t.shape)
        # print("x_encoded",x_encoded.shape)
        # print("t_encoded",t_encoded.shape)
        # print("combined",combined.shape,10+4)
        # exit(0)
        color = torch.sigmoid(self.f2(combined))
        return color

