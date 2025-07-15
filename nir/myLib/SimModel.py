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
# from nir.util import get_mgrid, jacobian
from nir.util import get_mgrid, jacobian, VideoFitting
from nir.util import Dataset,ToTensor

from nir.myLib.Layer import Layer
from nir.myLib.Decouple import Decouple
from nir.myLib.mySave import check,save2img,save0

class SimModel(nn.Module):
    def __init__(self, path):
        super().__init__()
        v = VideoFitting(path)
        videoloader = DataLoader(v, batch_size=1, pin_memory=True, num_workers=0)
        model_input, ground_truth = next(iter(videoloader))
        model_input, ground_truth = model_input[0].cuda(), ground_truth[0].cuda()
        ground_truth = ground_truth[:, 0:1]  # 将RGB图像转换为灰度图

        self.v=v
        self.model_input=model_input
        self.ground_truth = ground_truth

        #######################################################################

        # 三、流体
        self.f2 = Siren(in_features=3, out_features=1, hidden_features=128,
                   hidden_layers=4, outermost_linear=True)
        self.f2.cuda()
        self.parameters=[
            self.f2.parameters()
        ] #+ self.f_soft.parameters #+ self.f_rigid.parameters

    def forward(self,xyt): # soft, rigid, fluid
        return torch.sigmoid(self.f2(xyt))

    def loss(self, xyt, step, start, end):
        o= self.forward(xyt)
        loss_recon = ((o - self.ground_truth[start:end]) ** 2) * (10 ** 5)
        loss_recon = loss_recon.mean()

        loss = loss_recon
        if not step % 200:
            print("Step [%04d]: loss=%0.8f" % (step, loss))
        return loss

    def train(self,total_steps):
        model_input = self.model_input

        optim = torch.optim.Adam(lr=1e-4, params = itertools.chain.from_iterable(self.parameters))

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
        N, C, H, W = self.v.video.size()
        # 逐帧推理
        with torch.no_grad():
            # 生成时间轴的归一化值（-1到1）
            t_vals = torch.linspace(-1, 1, steps=N).cuda() if N > 1 else torch.zeros(1).cuda()
            pred_frames = []
            for i in range(N):
                # 生成当前帧的空间网格 (H*W, 2)
                spatial_grid = get_mgrid([H, W]).cuda()

                # 为所有空间点添加当前时间值
                t_val = t_vals[i]
                t_column = torch.full((spatial_grid.shape[0], 1), t_val).cuda()
                coords = torch.cat([spatial_grid, t_column], dim=1)

                # 模型推理并激活
                frame_output = self.forward(coords)

                # 调整形状为图像格式 (C, H, W)
                frame_image = frame_output.view(H, W, 1)  # .permute(2, 0, 1)
                # print("frame_image",frame_image.shape)

                pred_frames.append(frame_image)
        return torch.stack(pred_frames, dim=0)




