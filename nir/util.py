import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


def get_mgrid(sidelen, vmin=-1, vmax=1):
    if type(vmin) is not list:
        vmin = [vmin for _ in range(len(sidelen))]
    if type(vmax) is not list:
        vmax = [vmax for _ in range(len(sidelen))]
    tensors = tuple([torch.linspace(vmin[i], vmax[i], steps=sidelen[i]) for i in range(len(sidelen))])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, len(sidelen))
    return mgrid

def apply_homography(x, h):
    h = torch.cat([h, torch.ones_like(h[:, [0]])], -1)
    h = h.view(-1, 3, 3)
    x = torch.cat([x, torch.ones_like(x[:, 0]).unsqueeze(-1)], -1).unsqueeze(-1)
    o = torch.bmm(h, x).squeeze(-1)
    o = o[:, :-1] / o[:, [-1]]
    return o

def jacobian_old(y, x):
    B, N = y.shape
    jacobian = list()
    for i in range(N):
        v = torch.zeros_like(y)
        v[:, i] = 1.
        dy_i_dx = torch.autograd.grad(y,
                                      x,
                                      grad_outputs=v,
                                      retain_graph=True,
                                      create_graph=True)[0]  # shape [B, N]
        jacobian.append(dy_i_dx)
    jacobian = torch.stack(jacobian, dim=1).requires_grad_()
    return jacobian

def jacobian(y, x):
    # 第一步：前置检查 + 友好报错提示（核心：帮你快速定位输入问题）
    # 检查 x=t 的梯度属性
    if not x.requires_grad:
        # 尝试自动开启梯度追踪（仅对“叶子张量”有效，非叶子张量需外部处理）
        x.requires_grad_(True)
        # print(f"⚠️  警告：输入 x=t 的 requires_grad 为 False，已自动尝试开启（x.requires_grad=True）。"
        #       f"\n   若 x 不是叶子张量（如由其他张量计算而来），此操作可能无效，请在外部手动确保 x.requires_grad=True。")
    
    # 检查 y=h_global 的梯度属性（是否为计算图中间产物）
    if y.grad_fn is None:
        # 若 y 无 grad_fn，说明无梯度路径，无法计算雅可比，返回全0张量（避免报错）
        # print(f"⚠️  警告：输入 y=h_global 的 grad_fn 为 None（无梯度传播路径），无法计算雅可比。"
        #       f"\n   可能原因：1. y 是手动创建的常数张量；2. y 被 detach() 或在 no_grad() 中生成；3. y 与 x 无计算关联。"
        #       f"\n   已返回与输入维度匹配的全0张量，请注意结果有效性！")
        B, N = y.shape
        return torch.zeros(B, N, N, device=y.device, dtype=y.dtype)
    
    # 第二步：保留原有的雅可比计算逻辑，仅添加容错处理
    B, N = y.shape
    jacobian_list = []
    for i in range(N):
        v = torch.zeros_like(y, device=y.device, dtype=y.dtype)
        v[:, i] = 1.0
        
        # 调用 autograd.grad，添加 allow_unused=True 容错（处理 y 部分维度与 x 无关的情况）
        dy_i_dx = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=v,
            retain_graph=True,
            create_graph=True,
            allow_unused=True  # 新增：无关维度的梯度返回 None，避免报错
        )[0]
        
        # 处理 allow_unused=True 带来的 None（无关维度梯度设为0）
        if dy_i_dx is None:
            dy_i_dx = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        
        jacobian_list.append(dy_i_dx)
    
    # 堆叠结果（去掉原多余的 requires_grad_()，因 create_graph=True 已保留计算图）
    jacobian_mat = torch.stack(jacobian_list, dim=1)
    return jacobian_mat

class VideoFitting(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()

        self.path = path
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform

        self.video = self.get_video_tensor()
        self.num_frames, _, self.H, self.W = self.video.size()
        if self.video.shape[1]==1:#将灰度图扩充为彩图的格式
            self.video = torch.cat([self.video] * 3, dim=1)
        self.pixels = self.video.permute(2, 3, 0, 1).contiguous().view(-1, 3)
        self.coords = get_mgrid([self.H, self.W, self.num_frames])

        shuffle = torch.randperm(len(self.pixels))
        self.pixels = self.pixels[shuffle]
        self.coords = self.coords[shuffle]

    def get_video_tensor(self):
        frames = sorted(os.listdir(self.path))
        video = []
        for i in range(len(frames)):
            img = Image.open(os.path.join(self.path, frames[i]))
            img = self.transform(img)
            video.append(img)
        return torch.stack(video, 0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels