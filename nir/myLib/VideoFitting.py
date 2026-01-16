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

class VideoFitting(Dataset):
    def __init__(self, path, transform=None,useMask=True,maskPath="./nir/data/mask/filter"):
        super().__init__()
        self.numChannel=1 #不为3
        self.useMask=useMask
        self.maskPath=maskPath

        self.path = path
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform

        self.video = self.get_video_tensor(path)
        if self.useMask:
            # self.mask_ = 1-self.get_video_tensor(maskPath) #将血管MASK变为背景MASK #这个变换操作太难受了，所以取消这个操作
            self.mask_ = self.get_video_tensor(self.maskPath)
        self.num_frames, _, self.H, self.W = self.video.size()

        self.pixels = self.video.permute(2, 3, 0, 1).contiguous().view(-1, self.numChannel)
        if self.useMask:
            self.mask = self.mask_.permute(2, 3, 0, 1).contiguous().view(-1, self.numChannel)
        self.coords = get_mgrid([self.H, self.W, self.num_frames])

        self.shuffle = torch.randperm(len(self.pixels)) #打乱顺序
        self.pixels = self.pixels[self.shuffle]
        self.coords = self.coords[self.shuffle]
        if self.useMask:
            self.mask = self.mask[self.shuffle]
        else: self.mask =torch.tensor([[1.0]])

    def get_video_tensor(self,path):
        frames = sorted(os.listdir(path))
        video = []
        for i in range(len(frames)):
            img = Image.open(os.path.join(path, frames[i]))
            img = self.transform(img)
            video.append(img)
        return torch.stack(video, 0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError
        return self.coords, self.pixels, self.mask #坐标、图片灰度、背景分割图
    
    def reload_mask(self):
        """
        重新加载mask数据
        Args:
            new_mask_path: 新的mask路径，如果为None则使用原来的路径
        """
        
        # 重新加载mask
        self.mask_ = self.get_video_tensor(self.maskPath)
        self.mask = self.mask_.permute(2, 3, 0, 1).contiguous().view(-1, self.numChannel)
        
        # 重新打乱数据以保持mask与坐标的对齐
        self.mask = self.mask[self.shuffle]






