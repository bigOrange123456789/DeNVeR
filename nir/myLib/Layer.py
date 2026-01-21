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
class LearnableVariable(nn.Module):
    def __init__(self,value):
        super().__init__()
        self.v = nn.Parameter(torch.tensor(value, dtype=torch.float32).cuda())
    def forward(self):
        return self.v
    
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
class Layer(nn.Module): #用于表示软体层
    def getFeatureMask(self, k):
        dim = self.hidden_features_local

        # 创建索引
        indices = torch.arange(0, dim, dtype=torch.float32).cuda()
        
        # 创建基础向量：前k_int个元素为1，其余为0
        vec = k*dim-indices
        
        # return torch.clamp(vec, 0, 1)
        return  torch.sigmoid( vec**3 )
    def __init__(self,useGlobal=True,useLocal=True,useMatrix=True,useDeformation=False,
                 deformationSize=8,#hidden_features=128,
                 config={},
                 use_dynamicFeatureMask=False,
                 ):
        super().__init__()
        hidden_layers_map=4
        hidden_layers_global=2#3
        hidden_layers_local=4
        hidden_features_map=128
        hidden_features_global=128
        hidden_features_local=128
        self.dynamicTex=False # 纹理神经网络是否输入时间
        if not config is None:
            hidden_layers_map    = config["hidden_layers_map"]   
            hidden_layers_global = config["hidden_layers_global"]
            hidden_layers_local  = config["hidden_layers_local"] 
            hidden_features_map    = config["hidden_features_map"] 
            hidden_features_global = config["hidden_features_global"] 
            hidden_features_local  = config["hidden_features_local"]
            if "dynamicTex" in config:
                self.dynamicTex = config["dynamicTex"]  
        
        self.hidden_features_local = hidden_features_local #局部运动MLP的宽度
        ####################################
        self.useGlobal=useGlobal
        self.useLocal=useLocal
        self.useMatrix=useMatrix
        self.useDeformation=useDeformation
        self.deformationSize=deformationSize
        self.f_2D = Siren(
                        in_features=3 if self.dynamicTex else 2,# 动态纹理输入uvt，静态纹理输入uv 
                        out_features=1, #灰度值
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
        self.use_dynamicFeatureMask = use_dynamicFeatureMask
        if use_dynamicFeatureMask:
            self.kFeatureMask = LearnableVariable(1) #nn.Parameter(torch.tensor(1, dtype=torch.float32).cuda())
            self.parameters.append(self.kFeatureMask.parameters())
    def forward(self,xyt):
        featureMask=None
        if self.use_dynamicFeatureMask:
            featureMask=self.getFeatureMask(
                self.kFeatureMask()#.detach().clone() #这里必须进行梯度回传, 因此不能进行detach
            )
        h_local = self.g_local(xyt, featureMask=featureMask) if self.useLocal else torch.tensor(0.0)
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
            # print("h_global",h_global)
            # print("h_local",h_local)
            # exit(0) #符合预期
            xy_ = xyt[:, :-1] + h_global + h_local
        if self.dynamicTex:
            t = xyt[:, [-1]]#1000,1
            xy_t = torch.cat([xy_, t], dim=1) #1000,1 1000,2 => 1000,3
        color = torch.sigmoid(
            self.f_2D(xy_t if self.dynamicTex else xy_)
            )
        # color = torch.sigmoid(self.tex2D(xy_))
        return color,{
            "xy_":xy_,
            "h_local":h_local
        }
class Layer2(nn.Module): #用于表示软体层和流体层、能够实现PE和纹理featureMask
    def _getFeatureMask(self, k, tag):#tag=motion运动 或 map
        dim = self.config["hidden_features_map"] if tag=="map" else self.config["hidden_features_local"]

        # 创建索引
        indices = torch.arange(0, dim, dtype=torch.float32).cuda()
        
        # 创建基础向量：前k_int个元素为1，其余为0
        vec = k*dim-indices
        
        # return torch.clamp(vec, 0, 1)
        return  torch.sigmoid( vec**3 )
    
    def __init__(self,
                 useGlobal=True,useMatrix=True,
                 useLocal=True,useDeformation=False,deformationSize=8, #useDeformation用于让局部形变不太大
                 use_dynamicFeatureMask=False,
                 config={
                    #  "hidden_features":512,
                    #  "hidden_layers":4,
                     "use_residual":False,
                     "posEnc":{
                        "num_freqs_pos":10,
                        "num_freqs_time":4,
                        "APE":{
                            "total_steps":2000,
                            "warmup_steps":1000,
                        },
                     },
                     "use_featureMask":False, #没有在新版本中实现
                     "fm_total_steps":1000,
                     "use_maskP":True,
                 },
                 ):
        super().__init__()
        hidden_layers_map=4
        hidden_layers_global=2#3
        hidden_layers_local=4
        hidden_features_map=128
        hidden_features_global=128
        hidden_features_local=128
        self.dynamicTex=False # 纹理神经网络是否输入时间
        if not config is None:
            # print(config)
            hidden_layers_map      = config["hidden_layers_map"] if "hidden_layers_map" in config else config["hidden_layers"]  
            hidden_features_map    = config["hidden_features_map"] if "hidden_features_map" in config else config["hidden_features"]  
            hidden_layers_global   = config["hidden_layers_global"] if "hidden_layers_global" in config else 1
            hidden_features_global = config["hidden_features_global"] if "hidden_features_global" in config else 0
            hidden_layers_local    = config["hidden_layers_local"] if "hidden_layers_local" in config else 1
            hidden_features_local  = config["hidden_features_local"] if "hidden_features_local" in config else 0
            # config["hidden_layers_map"]=hidden_layers_map 
            config["hidden_features_map"]=hidden_features_map 

            if "dynamicTex" in config: #用于视频数据
                self.dynamicTex = config["dynamicTex"]  
        
        self.config = config
        self.use_featureMask=config["use_featureMask"]
        #################################### 开始实现PE部分 ####################################
        '''
            PE只用于纹理MLP
            只有当启用dynamicTex的时候, PE模块才生效
        '''
        self.use_posEnc = "posEnc" in config and config["posEnc"]
        if self.use_posEnc:
            self.use_APE= "APE" in config["posEnc"] and config["posEnc"]["APE"]
        in_features_num = 3
        if self.use_posEnc:
            if self.use_APE:
               # 对空间位置进行编码
                self.pos_encoder = AdaptivePositionalEncoder(2, num_freqs=config["posEnc"]["num_freqs_pos"],
                                                             total_steps =config["posEnc"]["APE"]["total_steps"],
                                                             warmup_steps=config["posEnc"]["APE"]["warmup_steps"])
                # 对时间进行编码
                self.time_encoder = AdaptivePositionalEncoder(1, num_freqs=config["posEnc"]["num_freqs_pos"],
                                                             total_steps =config["posEnc"]["APE"]["total_steps"],
                                                             warmup_steps=config["posEnc"]["APE"]["warmup_steps"])
            else:
                # 对空间位置进行编码
                self.pos_encoder = PositionalEncoder(2, num_freqs=config["posEnc"]["num_freqs_pos"])
                # 对时间进行编码
                self.time_encoder = PositionalEncoder(1, num_freqs=config["posEnc"]["num_freqs_time"])
            in_features_num = (
                self.pos_encoder.output_dim +
                self.time_encoder.output_dim
                )
        #################################### 结束PE部分代码 ####################################
        self.useGlobal=useGlobal
        self.useLocal=useLocal
        self.useMatrix=useMatrix
        self.useDeformation=useDeformation
        self.deformationSize=deformationSize
        self.f_2D = Siren(
                        in_features=in_features_num if self.dynamicTex else 2,# 动态纹理输入uvt，静态纹理输入uv 
                        out_features=1, #灰度值
                        hidden_features=hidden_features_map,#hidden_features, 
                        hidden_layers=hidden_layers_map,#4,
                        use_residual=config["use_residual"],#这个功能没啥用
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
        self.use_dynamicFeatureMask = use_dynamicFeatureMask
        if use_dynamicFeatureMask:
            self.kFeatureMask = LearnableVariable(1) #nn.Parameter(torch.tensor(1, dtype=torch.float32).cuda())
            self.parameters.append(self.kFeatureMask.parameters())
    def forward(self,xyt,current_step):  #这部分是整个程序的核心       
        # 1.整体运动 
        if self.useGlobal:
            if self.useMatrix: #使用矩阵运动
                c =self.g_global(xyt[:, [-1]])
                u = xyt[:, 0]
                v = xyt[:, 1]
                if c.shape[1]==6: #矩阵变换
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
                    v=scale*v #一个问题：放缩与旋转应该不能同时连续变化 #但是整体放缩应该是可以的
                    new_u = (u * cos_theta - v * sin_theta) + tx
                    new_v = (u * sin_theta + v * cos_theta) + ty
                # 组合成新坐标张量
                new_uv = torch.stack([new_u, new_v], dim=1)
                # xy_ = new_uv + h_local
            else: #不使用矩阵计算、只模拟整体位移
                new_uv = xyt[:, :-1] + self.g_global(xyt[:, [-1]]) 
        else: #不开启对象整体运动
            new_uv = xyt[:, :-1]
        
        # 2.局部位移
        if self.useLocal:
            # 2.1 自适应遮挡向量
            featureMask=self._getFeatureMask(#动态参数
                self.kFeatureMask(),#.detach().clone() #这里必须进行梯度回传, 因此不能进行detach
                "motion"
            ) if self.use_dynamicFeatureMask else None
            # 2.2 计算局部位移
            h_local = self.g_local(xyt, featureMask=featureMask) if self.useLocal else torch.tensor(0.0)
            # 2.3 限制形变程度
            if self.useDeformation:
                h_local=2*torch.sigmoid(h_local)-1
                h_local=h_local*self.deformationSize
        else:
            h_local = torch.tensor(0.0)
        xy_ = new_uv + h_local
        
        # 3.全景图
        # 3.1 位置编码
        x_encoded = self.pos_encoder(xy_,current_step) if self.use_posEnc else xy_
        if self.dynamicTex: #动态全景图
            t = xyt[:, [-1]]
            t_encoded = self.time_encoder(t,current_step)  if self.use_posEnc else t
            combined_in = torch.cat([x_encoded, t_encoded], dim=-1)
        else:
            combined_in = x_encoded
        # 3.2.渐进式遮挡向量
        featureMask = None
        if self.use_featureMask:
            if current_step is None: # 推理的时候这个似乎是None
                current_step=self.config["fm_total_steps"]
            k=current_step/self.config["fm_total_steps"]
            featureMask = self._getFeatureMask(k,"map")
        color = torch.sigmoid( self.f_2D(
            combined_in,
            featureMask=featureMask
        ))

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

    def forward(self, x,current_step=None):
        encoded = [x] if self.include_input else []
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * torch.pi * x))
            encoded.append(torch.cos(freq * torch.pi * x))
        return torch.cat(encoded, dim=-1)

class AdaptivePositionalEncoder(nn.Module):
    """
    自适应位置编码器，在训练过程中动态调整编码频率
    基于《Few-shot NeRF by Adaptive Rendering Loss Regularization》的思想
    """
    
    def __init__(self, input_dim=3, num_freqs=10, include_input=True, 
                 total_steps=20000, warmup_steps=5000, mode='linear'):
        """
        Args:
            input_dim: 输入维度（如3D坐标的维度为3）
            num_freqs: 位置编码的频率数量
            include_input: 是否包含原始输入
            total_steps: 总训练步数
            warmup_steps: 热身步数，在此期间频率逐渐增加
            mode: 频率调整模式 ('linear', 'cosine', 'exponential')
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.mode = mode
        
        # 预计算频率带宽（几何级数）
        self.freq_bands = 2.0 ** torch.linspace(0, num_freqs-1, num_freqs)
        
        # 注册为buffer，这些张量会被保存到模型状态中但不参与梯度更新
        self.register_buffer('current_step', torch.tensor(0))
        
        # 计算输出维度
        self.output_dim = input_dim * (2 * num_freqs + include_input)
        
    def get_frequency_mask(self, current_step=None):
        """
        根据当前训练步数计算频率掩码
        Returns:
            freq_mask: 形状为 [num_freqs] 的掩码张量，1表示启用，0表示遮挡
        """
        if current_step is None:
            current_step = self.current_step
            
        # 确保步数在合理范围内
        current_step = min(current_step, self.total_steps)
        
        if current_step >= self.warmup_steps:
            # 热身阶段结束，启用所有频率
            return torch.ones(self.num_freqs, device=self.freq_bands.device)
        
        # 计算当前启用的频率比例
        progress = current_step / self.warmup_steps
        
        if self.mode == 'linear':
            enabled_ratio = progress
        elif self.mode == 'cosine':
            enabled_ratio = 0.5 * (1 - torch.cos(torch.tensor(progress * np.pi)))
        elif self.mode == 'exponential':
            enabled_ratio = 1 - torch.exp(-5 * progress)
        else:
            enabled_ratio = progress
        
        # 计算应该启用的频率数量
        num_enabled_freqs = int(enabled_ratio * self.num_freqs)
        
        # 创建掩码（启用低频，遮挡高频）
        freq_mask = torch.zeros(self.num_freqs, device=self.freq_bands.device)
        freq_mask[:num_enabled_freqs] = 1.0
        
        return freq_mask
    
    def forward(self, x, current_step=None):
        """
        Args:
            x: 输入张量，形状为 [..., input_dim]
            current_step: 当前训练步数（可选）
        Returns:
            encoded: 编码后的张量，形状为 [..., output_dim]
        """
        if current_step is not None:
            self.current_step = torch.tensor(current_step, device=x.device)
        
        # 获取当前步数的频率掩码
        freq_mask = self.get_frequency_mask()
        
        # 如果设置为包含原始输入，则将其加入编码列表
        encoded = [x] if self.include_input else []
        
        # 对每个频率应用位置编码，并根据掩码进行遮挡
        for i, freq in enumerate(self.freq_bands):
            # 应用频率掩码
            mask_val = freq_mask[i]
            if mask_val > 0:
                # 启用该频率
                encoded.append(torch.sin(freq * x) * mask_val)
                encoded.append(torch.cos(freq * x) * mask_val)
            else:
                # 遮挡该频率：添加零张量以保持输出维度一致
                zero_sin = torch.zeros_like(x)
                zero_cos = torch.zeros_like(x)
                encoded.append(zero_sin)
                encoded.append(zero_cos)
        
        # 沿最后一个维度拼接所有编码后的特征
        return torch.cat(encoded, dim=-1)
    
    def get_current_freq_info(self):
        """返回当前启用的频率信息，用于监控训练过程"""
        freq_mask = self.get_frequency_mask()
        enabled_freqs = torch.sum(freq_mask).item()
        return {
            'current_step': self.current_step.item(),
            'enabled_freqs': enabled_freqs,
            'total_freqs': self.num_freqs,
            'enabled_ratio': enabled_freqs / self.num_freqs,
            'freq_mask': freq_mask.detach().cpu().numpy()
        }

class Layer_video(nn.Module): #用来拟合视频的模块
    def getFeatureMask(self, k):
        dim = self.config["hidden_features"]

        # 创建索引
        indices = torch.arange(0, dim, dtype=torch.float32).cuda()
        
        # 创建基础向量：前k_int个元素为1，其余为0
        vec = (k*dim-indices).float()
        
        return torch.clamp(vec, 0, 1)

    def __init__(
                self,
                config={
                     "hidden_features":512,
                     "hidden_layers":4,
                     "use_residual":True,
                     "posEnc":{
                        "num_freqs_pos":10,
                        "num_freqs_time":4,
                        "APE":{
                            "total_steps":2000,
                            "warmup_steps":1000,
                        },
                     },
                     "use_featureMask":False,
                     "fm_total_steps":1000,
                    #  "use_maskP":True,
                },
            ):
        super().__init__()
        self.config=config
        self.use_posEnc = "posEnc" in config and config["posEnc"]
        if self.use_posEnc:
            self.use_APE= "APE" in config["posEnc"] and config["posEnc"]["APE"]
        # self.use_maskP=config["use_maskP"]#True#能否自动遮挡该图层的输出
        self.use_featureMask=config["use_featureMask"]
        # if self.use_featureMask:
        #     self.featureMaskK=nn.Parameter(torch.tensor(0, dtype=torch.float32)).cuda() #最开始完全遮挡
        #     self.featureMask=torch.zeros(512)
        # print("self.use_posEnc",self.use_posEnc)
        in_features_num = 3
        if self.use_posEnc:
            if self.use_APE:
               # 对空间位置进行编码
                self.pos_encoder = AdaptivePositionalEncoder(2, num_freqs=config["posEnc"]["num_freqs_pos"],
                                                             total_steps =config["posEnc"]["APE"]["total_steps"],
                                                             warmup_steps=config["posEnc"]["APE"]["warmup_steps"])
                # 对时间进行编码
                self.time_encoder = AdaptivePositionalEncoder(1, num_freqs=config["posEnc"]["num_freqs_pos"],
                                                             total_steps =config["posEnc"]["APE"]["total_steps"],
                                                             warmup_steps=config["posEnc"]["APE"]["warmup_steps"])
            else:
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
        # self.parameters = [self.f2.parameters()]
        # if self.use_maskP:
        #     self.parameters.append(self.maskP)
        self.parameters = [self.parameters()]

    def forward(self,xyt,current_step):
        if self.use_posEnc:
            xy_ = xyt[:, :-1]
            t = xyt[:, [-1]]
            x_encoded = self.pos_encoder(xy_,current_step)
            t_encoded = self.time_encoder(t,current_step) # 确保时间有正确的维度
            combined = torch.cat([x_encoded, t_encoded], dim=-1)
        else:
            combined = xyt
        # print("xy_",xy_.shape)
        # print("t",t.shape)
        # print("x_encoded",x_encoded.shape)
        # print("t_encoded",t_encoded.shape)
        # print("combined",combined.shape,10+4)
        # exit(0)
        featureMask = None
        if self.use_featureMask:
            if current_step is None:#推理的时候这个似乎是None
                current_step=self.config["fm_total_steps"]
            k=current_step/self.config["fm_total_steps"]
            featureMask = self.getFeatureMask(k)
        # if current_step%200==0:
        #     print("featureMask.sum():",featureMask.sum(),"myLib/Layer.py")
        color = torch.sigmoid(self.f2(combined, featureMask=featureMask))
        # if self.use_maskP:
        #     color = self.maskP * color + (1-self.maskP)*1
        return color,None #第二个输出原始None是为了和其他层结构保持统一

class Layer_rigid(nn.Module):
    def getFeatureMask(self, k):
        dim = self.hidden_features_global

        # 创建索引
        indices = torch.arange(0, dim, dtype=torch.float32).cuda()
        
        # 创建基础向量：前k_int个元素为1，其余为0
        vec = k*dim-indices
        
        # return torch.clamp(vec, 0, 1)
        return  torch.sigmoid( vec**3 )
    def __init__(self,useGlobal=True,useLocal=True,useMatrix=True,useDeformation=False,
                 deformationSize=3*0.004,
                #  hidden_features=128,
                #  hidden_layers_map=4,
                #  hidden_layers_global=4,
                #  hidden_layers_local=4,
                #  hidden_features_map=128,
                #  hidden_features_global=128,
                #  hidden_features_local=128,
                 config=None,
                 useSmooth=False, # 是否计算平滑损失函数
                 stillness=False, # 是否静止不动
                 v=None,#原始视频数据的解析对象, 这里读取该对象是为了获取视频的尺寸长度参数
                 interval=1.0,
                 use_dynamicFeatureMask=False,
                 ):
        super().__init__()
        self.useGlobal=useGlobal
        self.useLocal=useLocal
        self.useMatrix=useMatrix
        self.useDeformation=useDeformation
        self.deformationSize=deformationSize
        self.useSmooth=useSmooth
        self.interval=interval
        self.stillness=stillness
        hidden_layers_map=4
        hidden_layers_global=4
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
        self.hidden_features_global=hidden_features_global
            
        self.v=v #原始视频数据的解析对象, 这里读取该对象是为了获取视频的尺寸长度参数
        # print(2*4*512, 2)
        # print("hidden_features=",hidden_features_map, "hidden_layers=",hidden_layers_map)
        self.f_2D = Siren(in_features=2, out_features=1,
                          hidden_features=hidden_features_map, hidden_layers=hidden_layers_map,
                          outermost_linear=True)
        self.f_2D.cuda()
        self.parameters = [self.f_2D.parameters()]
        self.useTex2D=False#True
        if self.useTex2D:
            # self.tex2D = nn.Parameter(torch.zeros(512, 512))
            # self.parameters.append(self.tex2D.parameters())
            # self.tex2D = torch.zeros(1, 512, 512, requires_grad=True)
            # self.parameters.append(self.tex2D)
            self.tex2D = Tex2D(512)
            self.parameters.append(self.tex2D.parameters())
        if self.useGlobal:
            self.g_global = Siren(in_features=1, out_features=4 if self.useMatrix else 2,
                                  # hidden_features=16, hidden_layers=2,
                                  hidden_features=hidden_features_global, hidden_layers=hidden_layers_global,
                                  outermost_linear=True)
            self.parameters.append(self.g_global.parameters())
            self.g_global.cuda()
        if self.useLocal:
            self.g_local = Siren(in_features=3, out_features=2,
                                 hidden_features=hidden_features_local, hidden_layers=hidden_layers_local,
                                 outermost_linear=True)
            self.g_local.cuda()
            self.parameters.append(self.g_local.parameters())    
        if use_dynamicFeatureMask:
            self.kFeatureMask = LearnableVariable(1) #nn.Parameter(torch.tensor(1, dtype=torch.float32).cuda())
            self.parameters.append(self.kFeatureMask.parameters())
        self.use_dynamicFeatureMask = use_dynamicFeatureMask
    
    def forward(self,xyt,openLocalDeform):#openLocalDeform原来是stage,当stage=0的时候对应open=False
        if self.stillness: #静止不动
            xy_ = xyt[:, :-1]
            h_local = torch.zeros_like(xy_)   #torch.tensor(0.0)
            h_global =torch.zeros_like(xy_)   #torch.tensor(0.0)
            loss_smooth =torch.tensor(0.0)
        else:
            t = xyt[:, [-1]]
            featureMask=None
            if self.use_dynamicFeatureMask:
                featureMask=self.getFeatureMask(
                    self.kFeatureMask()#.detach().clone()
                )
            # print("flag")
            # print("self.use_dynamicFeatureMask",self.use_dynamicFeatureMask)
            # print("self.getFeatureMask",self.getFeatureMask)
            # print("self",self)
            # exit(0)
            h_global = self.g_global(t,featureMask=featureMask)
            loss_smooth = torch.tensor(0.0)
            
            '''
                print("h_global:",type(h_global),h_global.shape)
                print("t:",type(t),t.shape)
                print("self.useSmooth:",self.useSmooth)
                j0=jacobian(h_global, t)
                print("j0:",type(j0),j0.shape)
                h0=hessian_vectorized(h_global, t)
                print("h0",type(h0),h0.shape)
                exit(0)
            
                h_global: <class 'torch.Tensor'> torch.Size([32768, 4])
                t: <class 'torch.Tensor'> torch.Size([32768, 1])
                self.useSmooth: 2
                j0: <class 'torch.Tensor'> torch.Size([32768, 4, 1])
                h0 <class 'torch.Tensor'> torch.Size([32768, 4, 1, 1])
            '''
            if self.useSmooth:#>0
                if self.useSmooth==1: #运动速度趋于0 #失败:无法迫使刚体正确运动
                    loss_smooth = loss_smooth + jacobian(h_global, t).abs().mean()
                elif self.useSmooth==2: #加速度趋于0
                    loss_smooth = loss_smooth + hessian_vectorized(h_global, t).abs().mean()
                elif self.useSmooth==3:#全局运动趋向于固定
                    N, C, H, W = self.v.video.size()  # 帧数、通道数、高度、宽度
                    interval0 = torch.tensor(self.interval, device=xyt.device)*2/(N-1)
                    data0 = h_global#[:,:2]#只有前两列被用到了，后两列没有用到
                    data_behind = self.g_global(t+interval0)#[:,:2] #后面
                    speed = (data_behind-data0)/interval0
                    loss_smooth = ( speed**2 ).mean() 
                elif self.useSmooth==4: # 二阶导数
                    N, C, H, W = self.v.video.size()  # 帧数、通道数、高度、宽度
                    interval0 = torch.tensor(self.interval, device=xyt.device)*2/(N-1)
                    data0 = h_global#[:,:2]#只有前两列被用到了，后两列没有用到
                    data_behind = self.g_global(t+interval0)#[:,:2] #后面
                    # data_front = self.g_global(t+interval0)#[:,:2] #前面
                    data_front = self.g_global(t-interval0)#[:,:2] #前面
                    acceleration = (data_behind-2*data0+data_front)/(interval0**2)
                    loss_smooth = (acceleration**2).mean() 
                elif self.useSmooth==5: # 加速度÷速度 #没有正确运动、反而出现了晃动问题
                    N, C, H, W = self.v.video.size()  # 帧数、通道数、高度、宽度
                    interval0 = torch.tensor(self.interval, device=xyt.device)*2/(N-1)
                    data0 = h_global#[:,:2]#只有前两列被用到了，后两列没有用到
                    data_behind = self.g_global(t+interval0)#[:,:2] #后面
                    # data_front = self.g_global(t+interval0)#[:,:2] #前面
                    data_front = self.g_global(t-interval0)#[:,:2] #前面
                    acceleration = (data_behind-2*data0+data_front)/(interval0**2)
                    speed = (data_behind-data0)/interval0 #速度的上限应该是2，速度超过2是不可想象的
                    eps = 10**-10
                    loss_smooth = ((#静止的时候就不能受力？哪有这样的道理
                        acceleration/(torch.sigmoid(speed).clone().detach()+eps)
                        )**2).mean()
                    # loss_smooth = ((
                    #     acceleration/(torch.sigmoid(speed+2).clone().detach()+eps)
                    #     )**2).mean()
                # elif self.useSmooth==6: # 加速度÷速度 #没有正确运动、反而出现了晃动问题
                #     N, C, H, W = self.v.video.size()  # 帧数、通道数、高度、宽度
                #     interval0 = torch.tensor(self.interval, device=xyt.device)*2/(N-1)
                #     # torch.rand((), device=xyt.device)
                #     data0 = self.g_global(t)# h_global#[:,:2] #只有前两列被用到了，后两列没有用到
                #     data_behind = self.g_global(t+interval0)#[:,:2] #后面
                #     data_front = self.g_global(t-interval0)#[:,:2] #前面
                #     acceleration = (data_behind-2*data0+data_front)/(interval0**2)
                #     speed = (data_behind-data0)/interval0 #速度的上限应该是2，速度超过2是不可想象的
                #     eps = 10**-10
                #     loss_smooth = ((#静止的时候就不能受力？哪有这样的道理
                #         acceleration/(torch.sigmoid(speed+2).clone().detach()+eps)
                #         )**2).mean()
                elif self.useSmooth==6: # 加速度÷速度 #没有正确运动、反而出现了晃动问题
                    N, C, H, W = self.v.video.size()  # 帧数、通道数、高度、宽度
                    interval0 = torch.tensor(self.interval, device=xyt.device)*2/(N-1)
                    t_new = t + torch.rand((), device=xyt.device)*2/(N-1) #时间序列添加随机扰动
                    data0 = self.g_global(t_new)# h_global#[:,:2] #只有前两列被用到了，后两列没有用到
                    data_behind = self.g_global(t_new+interval0)#[:,:2] #后面
                    data_front = self.g_global(t_new-interval0)#[:,:2] #前面
                    acceleration = (data_behind-2*data0+data_front)/(interval0**2)
                    speed = (data_behind-data0)/interval0 #速度的上限应该是2，速度超过2是不可想象的
                    eps = 10**-10
                    loss_smooth = ((#静止的时候就不能受力？哪有这样的道理
                        acceleration/(torch.sigmoid(speed+2).clone().detach()+eps)
                        )**2).mean()
                # elif self.useSmooth==7: # 多尺度斜率恒定的约束
                #     N, C, H, W = self.v.video.size()  # 帧数、通道数、高度、宽度
                #     interval0 = torch.tensor(self.interval, device=xyt.device)*2/(N-1)
                #     data0 = h_global
                #     data_behind = self.g_global(t-interval0) #后面
                #     speed0 = (data0-data_behind)/interval0 #速度的上限应该是2，速度超过2是不可想象的
                #     num=3
                #     # speedList=[]
                #     loss_smooth = 0
                #     for i0 in range(num):
                #         i = i0+1
                #         data_front = self.g_global(t+(i+1)*interval0)#[:,:2] #后面
                #         speed1 = (data_front-data0)/interval0
                #         loss_smooth = loss_smooth + ( speed1 - speed0 )**2

                #     eps = 10**-10
                #     loss_smooth = ((
                #         acceleration/(torch.sigmoid(speed+2).clone().detach()+eps)
                #         )**2).mean()

            ############################################################
            # xy_ = xyt[:, :-1] + h_global #+ h_local
            if self.useMatrix: #True
                c =h_global
                u = xyt[:, 0]
                v = xyt[:, 1]
                if c.shape[1]==6: # 6个自由度的全局变换
                    new_u = c[:,0] * u + c[:,1] * u + c[:,2]
                    new_v = c[:,3] * v + c[:,4] * v + c[:,5]
                else: # 4个自由度的全局变换 (后面可能添加5个自由度的模式)
                    # 提取参数 (忽略可能的第五个参数)
                    tx = c[:, 0]  # X轴位移 #范围大致是[-1,+1],不严格
                    ty = c[:, 1]  # Y轴位移
                    #实际上没有使用旋转自由度和缩放自由度
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

                    h_global = torch.cat([
                        c[:, [0]],
                        c[:, [1]]
                    ], dim=-1)
                # 组合成新坐标张量
                new_uv = torch.stack([new_u, new_v], dim=1)
                xy_ = new_uv
            else: #不使用变换矩阵 #两个自由度的模式
                xy_ = xyt[:, :-1] + h_global
            ############################################################
            if not openLocalDeform: # stage==0: # 纹理学习，不分析局部位移  # 不使用局部位移
                h_local = torch.tensor(0.0) # h_local = self.g_local(xyt) if self.useLocal else torch.tensor(0.0)
            elif self.useLocal: # 只分析局部位移，不学习整体运动和纹理
                h_local = self.g_local(xyt)
                if False:
                 if self.useSmooth: # 局部形变也要平滑
                    loss_smooth = loss_smooth + jacobian(h_local, xyt).abs().mean() # if self.useSmooth else 0
                h_local = 2 * torch.sigmoid(h_local) - 1
                h_local = h_local * self.deformationSize
                xy_ = xy_ + h_local
            else:
                h_local = torch.tensor(0.0)
        # color = torch.sigmoid(self.f_2D(xy_))
        # if False:
        color = torch.sigmoid(self.f_2D(xy_))
        if self.useTex2D:
            color = color + torch.sigmoid(self.f_2D(xy_))
        # color = torch.sigmoid(self.f_2D(xy_))
        return color,{
            "xy_":xy_,
            "h_local":h_local, #训练的时候stage为0，所以输出的是一个常数0
            "h_global":h_global, # 不明白为啥之前的全局位移要让它不可学习 #之前的版本是下面一行
            #"h_global":h_global.clone().detach()
            "loss_smooth":loss_smooth
        }
