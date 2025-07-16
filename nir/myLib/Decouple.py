import os
import numpy as np
from itertools import chain
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import re

from nir.model import Siren
from nir.util import get_mgrid, jacobian#, VideoFitting
from nir.util import Dataset,ToTensor

from nir.myLib.VideoFitting import VideoFitting

weight_target = [0.25,100,400]#RSF #信息量比例的目标
weight_init = [10**2, 1, 1] #RSF

import torch.nn as nn
from nir.myLib.Layer import Layer
import itertools
import math
class Decouple(nn.Module):
    def log(self,x): #x中的全部元素均不能为负数
        import math
        e = math.e # 获取自然数 e 的值
        eps = e ** -101
        # return x
        # eps = 1e-10 #torch.finfo(torch.float64).eps
        return -1.*torch.log(x+eps) # return -1.*torch.log(x.abs()+eps)
    def __init__(self, path,maskPath="./nir/data/mask/filter",hidden_features=128):
        super().__init__()
        v = VideoFitting(path,maskPath=maskPath)
        videoloader = DataLoader(v, batch_size=1, pin_memory=True, num_workers=0)
        model_input, ground_truth, mask = next(iter(videoloader))
        model_input, ground_truth, mask = model_input[0].cuda(), ground_truth[0].cuda(), mask[0].cuda()
        ground_truth = ground_truth[:, 0:1]  # 将RGB图像转换为灰度图
        if False:
            print("gt_illu_max",torch.max(ground_truth))
            print("gt_illu_min",torch.min(ground_truth))
            ground_truth = self.log(ground_truth)
            print("gt_dis_max", torch.max(ground_truth))
            print("gt_dis_min", torch.min(ground_truth))
        # exit(0)
        self.v=v
        self.model_input=model_input
        self.ground_truth=ground_truth
        self.mask=mask

        #######################################################################

        # 一、软体
        self.NUM_soft = 1 # 软体层的数据
        self.f_soft_list = []
        for i in range(self.NUM_soft):
            self.f_soft_list.append(Layer(useGlobal=False,hidden_features=hidden_features))
        # 二、刚体
        self.NUM_figid = 3 # 刚体层的数据
        self.f_rigid_list=[]
        for i in range(self.NUM_figid):
            self.f_rigid_list.append(Layer(useDeformation=True,hidden_features=hidden_features))
        # 三、流体
        self.f2 = Siren(in_features=3, out_features=1, hidden_features=hidden_features,
                   hidden_layers=4, outermost_linear=True)
        self.f2.cuda()

        self.parameters=[
            self.f2.parameters()
        ] #+ self.f_soft.parameters #+ self.f_rigid.parameters
        for i in range(self.NUM_figid):
            self.parameters = self.parameters + self.f_rigid_list[i].parameters
        for i in range(self.NUM_soft):
            self.parameters = self.parameters + self.f_soft_list[i].parameters

    def forward(self,xyt): # soft, rigid, fluid
        # 1.刚体
        o_rigid_all = 1
        o_rigid_list = []
        h_local_list = []
        for i in range(self.NUM_figid):
            o_rigid0, p_rigid0 = self.f_rigid_list[i](xyt)
            o_rigid_all = o_rigid_all*o_rigid0
            o_rigid_list.append(o_rigid0)
            h_local_list.append(p_rigid0["h_local"])

        # 2.软体
        o_soft_all = 1
        o_soft_list = []
        for i in range(self.NUM_soft):
            o_soft0, _ = self.f_soft_list[i](xyt)
            o_soft_all = o_soft_all * o_soft0
            o_soft_list.append(o_soft0)

        # 3.流体
        o_fluid = torch.sigmoid(self.f2(xyt))
        o = o_rigid_all * o_soft_all * (1.0 - o_fluid)
        return o , {
            "r": o_rigid_list,
            "s": o_soft_list,
            "f": o_fluid
        } ,{
            "h_local":h_local_list,
            "o_rigid_all":o_rigid_all,
            "o_soft_all":o_soft_all
        }

    def loss(self, xyt, step, start, end):
        ground_truth = self.ground_truth
        mask = self.mask
        o, layers, p = self.forward(xyt)
        o_soft_list  = layers["s"]
        o_fluid = layers["f"]

        def fluidInf1(x): #获取MASK中白色区域的像素数量占比
            EXPONENT = 7
            mask_binary = torch.pow(x - 0.5, EXPONENT) / (0.5 ** EXPONENT) + 0.5
            return mask_binary.abs().mean()
        def fluidInf2(x): #获取MASK中非0区域的像素数量占比
            EXPONENT = 7
            mask_binary = torch.pow(x - 0.5, EXPONENT) / (0.5 ** EXPONENT) + 0.5#将大于0.5的数据变为1
            return mask_binary.abs().mean()
        def fluidInf(x):
            k = 10 ** 10  # 一个无限大的超参数
            # k = 2 ** 64 - 2  # 一个超参数
            x = torch.log(1 + k * x) / math.log(1 + k) #将非零数据都变为1
            mask_binary = torch.log(1 + k * x) / math.log(1 + k) #用同一个函数处理两次、结果更像二值图
            return mask_binary.abs().mean()
        with torch.no_grad():
            def fun0(x):
                eps = math.e ** -101
                x = x.detach().clone()
                x = x.clamp(min=eps)
                # x = x.clamp(max=1 - eps)
                return (-torch.log(x))**4

            def fun1(x):
                eps = 1e-10#torch.tensor([1e-10])
                x = x.detach().clone()
                x=x.clamp(min=eps)
                x=x.clamp(max=1-eps)
                inner_expr = (math.pi/2) * (x+1)
                return -1.0 * torch.tan(inner_expr)

                # return -np.tan( (math.pi/2) * (x +1) )
            # print("0",fun0(torch.tensor([0.0])))
            # print("1", fun0(torch.tensor([1.0])))
            # exit(0)

            # 下述衡量信息量的三个指标都是0-1
            # i_r = p["o_rigid_all"].detach().clone().abs().mean()
            i_r0 = torch.var(p["o_rigid_all"].detach().clone().abs()) # 刚体层方差信息量
            i_s0 = 1-p["o_soft_all"].detach().clone().abs().mean()    # 软体层暗度信息量
            # i_f = o_fluid.detach().clone().abs().mean()              # 流体层亮度信息量
            i_f0 = fluidInf(o_fluid.detach().clone())  # 流体层亮度信息量
            # wr = (1-i_r)*fun0((1-i_s)*(1-i_f)) #本层的信息量越少则约束越小
            # ws = (1-i_s)*fun0((1-i_r)*(1-i_f))
            # wf = (1-i_f)*fun0((1-i_r)*(1-i_s))
            temp=[
                i_r0 / weight_target[0],
                i_s0 / weight_target[1],
                i_f0 / weight_target[2]
            ]
            i_r = temp[0] / sum(temp)
            i_s = temp[1] / sum(temp)
            i_f = temp[2] / sum(temp)

            wr = i_r * fun0(i_s * i_f)  # 1.本层的信息量越少则本层约束越小
            ws = i_s * fun0(i_r * i_f)  # 2.它层的信息量越少则本层约束越大
            wf = i_f * fun0(i_r * i_s)

        loss_recon = ((o - self.ground_truth[start:end]) ** 2) * (10 ** 5)
        if self.v.useMask:
            # print("self.mask[start:end]",self.mask[start:end].shape,self.mask[start:end].max(),self.mask[start:end].min())
            loss_recon = loss_recon*self.mask[start:end]
            loss_recon = loss_recon.sum()/(self.mask[start:end].sum()+1e-8)
        else:
            loss_recon = loss_recon.mean()
        # loss_recon = ((o - ground_truth[start:end]) ** 2).mean() * (10**5)
        if False:# 不对血管区域进行重建监督
            loss_recon = (((1.0-o_fluid)*(o - ground_truth[start:end])) ** 2).mean()*10.
        # if False:
        #     loss_recon = ((self.log(o) - ground_truth[start:end]) ** 2).mean() * 10.
        # 一、刚体
        loss_rigid = 0 #刚体的局部位移尽量小
        for i in range(self.NUM_figid):
            # loss_rigid = loss_rigid+1.20 * p["h_local"][i].abs().mean() # 减少刚体的信息量
            loss_rigid = loss_rigid + p["h_local"][i].abs().mean()  # 减少刚体的信息量
        loss_rigid = loss_rigid * weight_init[0] #(10**2)
        # 二、软体
        # loss_soft = (10**-5) * (1. - p["o_soft_all"]).abs().mean() # 减少软体的信息量
        loss_soft = (1. - p["o_soft_all"]).abs().mean() * weight_init[1] #(0.1) # 减少软体的信息量
        # 三、流体 # 减少流体的信息量
        # loss_fluid = 0.02 * o_fluid.abs().mean() # 减少流体的信息量
        # loss_fluid = loss_fluid + 0.01 * (o_fluid*(1-o_fluid)).abs().mean() # 二值化约束
        # loss_fluid = (10**5) * fluidInf(o_fluid.abs()) # 减少流体的信息量
        loss_fluid = fluidInf(o_fluid.abs()) * weight_init[2] #1 # 减少流体的信息量
        # 刚体:1.2->1;软体:0.1**5->0.1;流体:10**5->10

        if False:#放弃了平衡信息量的做法
            loss_rigid = loss_rigid * wr
            loss_soft  = loss_soft  * ws
            loss_fluid = loss_fluid * wf

        loss = loss_recon + loss_soft + loss_fluid + loss_rigid

        self.layers=layers
        if not step % 200:
            print("Step [%04d]: loss=%0.8f, recon=%0.8f, loss_soft=%0.8f, loss_fluid=%0.8f, loss_rigid=%0.4f" % (
                step, loss, loss_recon, loss_soft , loss_fluid , loss_rigid))
            print("i_r0:", i_r0.item(),"; i_s0:", i_s0.item(),"; i_f0:", i_f0.item(),"\ntemp",
                  [temp[0].item(),temp[1].item(),temp[2].item()])
            print("i_r",i_r.item(), "\twr", wr.item()) # i_f为0
            print("i_s",i_s.item(), "\tws", ws.item())
            print("i_f",i_f.item(), "\twf", wf.item()) #wr、ws为无穷大
            # print(self.log(torch.tensor([0.0])))
            # exit(0)
        return loss

    def train(self,total_steps):
        model_input = self.model_input

        optim = torch.optim.Adam(lr=1e-4, params = itertools.chain.from_iterable(self.parameters))
        # optim = torch.optim.Adam(lr=1e-4, params=chain(self.parameters))

        batch_size = (self.v.H * self.v.W) // 8
        for step in range(total_steps):
            start = (step * batch_size) % len(model_input)
            end = min(start + batch_size, len(model_input))

            xyt = model_input[start:end].requires_grad_()
            loss = self.loss(xyt, step,start,end)

            optim.zero_grad()
            loss.backward()
            optim.step()

    def getVideo(self):
        N, C, H, W = self.v.video.size()  # 帧数、通道数、高度、宽度
        # 创建空列表存储每帧预测结果
        pred_frames = []
        layers_frames = {
            "r": [],
            "s": [],
            "f": []
        }
        p_frames = {
            "o_rigid_all": [],
            "o_soft_all": []
        }
        # 生成时间轴的归一化值（-1到1）
        t_vals = torch.linspace(-1, 1, steps=N).cuda() if N > 1 else torch.zeros(1).cuda()

        # 逐帧推理
        with torch.no_grad():
            for i in range(N):
                # 生成当前帧的空间网格 (H*W, 2)
                spatial_grid = get_mgrid([H, W]).cuda()

                # 为所有空间点添加当前时间值
                t_val = t_vals[i]
                t_column = torch.full((spatial_grid.shape[0], 1), t_val).cuda()
                coords = torch.cat([spatial_grid, t_column], dim=1)

                # 模型推理并激活
                frame_output, layers, p = self.forward(coords)

                # 调整形状为图像格式 (C, H, W)
                frame_image = frame_output.view(H, W, 1)  # .permute(2, 0, 1)
                # print("frame_image",frame_image.shape)

                pred_frames.append(frame_image)
                for id in layers_frames:
                    if id == "f":
                        layers_frames[id].append(layers[id].view(H, W, 1))
                    else:
                        layers_frames[id].append(layers[id])
                for id in p_frames:
                    p_frames[id].append(p[id].view(H, W, 1))
        # return pred_frames, layers_frames, p_frames
            video_pre = torch.stack(pred_frames, dim=0)

            def p01(original_list):
                l = list(map(list, zip(*original_list)))  # 交换列表的前两层
                for i in range(len(l)):
                    for j in range(len(l[i])):
                        l[i][j] = l[i][j].view(H, W, 1)
                        # print(type(l[i][j]),l[i][j].shape)
                        # exit(0)
                    l[i] = torch.stack(l[i], dim=0)
                return l

            layers = {
                "r": p01(layers_frames["r"]),
                "s": p01(layers_frames["s"]),
                "f": torch.stack(layers_frames["f"], dim=0)
            }
            p = {
                "o_rigid_all": torch.stack(p_frames["o_rigid_all"], dim=0),
                "o_soft_all": torch.stack(p_frames["o_soft_all"], dim=0)
            }
            return video_pre, layers, p




