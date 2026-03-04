import os
import numpy as np
from itertools import chain
from PIL import Image
import torch
# from torch.utils.data import DataLoader
import torchvision.transforms as T
import re

from nir.model import Siren
from nir.util import get_mgrid, jacobian#, VideoFitting
from nir.util import Dataset,ToTensor

class VideoFitting(Dataset):
    def __init__(self, path, path_custom=None, videoId=None, transform=None,useMask=True,maskPath="./nir/data/mask/filter",reconFlow=False):
        super().__init__()
        self.numChannel=1 #不为3
        self.useMask=useMask
        self.maskPath=maskPath
        self.reconFlow=reconFlow

        self.path = path
        self.path_custom =path_custom
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform

        self.video = self.get_video_tensor(path)
        self.num_frames, _, self.H, self.W = self.video.size()

        self.pixels = self.video.permute(2, 3, 0, 1).contiguous().view(-1, self.numChannel)
        self.coords = get_mgrid([self.H, self.W, self.num_frames])

        self.shuffle = torch.randperm(len(self.pixels)) #打乱顺序
        self.pixels = self.pixels[self.shuffle]
        self.coords = self.coords[self.shuffle]

        if self.useMask:
            self.mask_ = self.get_video_tensor(self.maskPath) #血管
            self.mask = self.mask_.permute(2, 3, 0, 1).contiguous().view(-1, self.numChannel) 
            # permute维度重排 "(B, C, H, W) 重排为 (H, W, B, C)""
            # contiguous 创建一个新的连续内存布局的张量
            # view(-1, self.numChannel) 重塑形状
            self.mask = self.mask[self.shuffle]
        else: self.mask =torch.tensor([[1.0]])

        if self.reconFlow and (not path_custom is None):
            pathGap1 = os.path.join(path_custom, "flow_imgs_gap1", videoId)
            pathGap2 = os.path.join(path_custom, "flow_imgs_gap-1", videoId)
            pathSkel = os.path.join(path_custom, "skeltoize", videoId)
            dataGap1 = self.get_video_tensor(pathGap1, +1)
            dataGap2 = self.get_video_tensor(pathGap2, -1)
            datapathSkel = self.get_video_tensor(pathSkel)
            self.dataGap1 = dataGap1.permute(2, 3, 0, 1).contiguous().view(-1, self.numChannel)[self.shuffle] 
            self.dataGap2 = dataGap2.permute(2, 3, 0, 1).contiguous().view(-1, self.numChannel)[self.shuffle] 
            self.datapathSkel = datapathSkel.permute(2, 3, 0, 1).contiguous().view(-1, self.numChannel)[self.shuffle] 
            self.pixels = torch.cat([ #更每个像素点增加更多的维度
                self.pixels, 
                self.dataGap1, 
                self.dataGap2, 
                self.datapathSkel
                ], dim=1)

        self._getVesselSet()#将血管区域单独提取出来

    def _getVesselSet(self): #目前是冗余的代码
        #接下来获取只保留血管像素的对象
        valid_vessel = (self.mask > 0.5).view(-1)
        self.pixels_vessel = self.pixels[valid_vessel]
        if self.useMask:
            self.mask_vessel = self.mask[valid_vessel]
        self.coords_vessel = self.coords[valid_vessel]

        self.shuffle_vessel = torch.randperm(len(self.pixels_vessel)) #打乱顺序
        self.pixels_vessel = self.pixels_vessel[self.shuffle_vessel]
        self.coords_vessel = self.coords_vessel[self.shuffle_vessel]
        if self.useMask:
            self.mask_vessel = self.mask_vessel[self.shuffle_vessel]
        else: self.mask_vessel =torch.tensor([[1.0]])

        self.ratio=len(self.pixels_vessel)/len(self.pixels) #血管数据全部数据中的占比

    def get_video_tensor(self,path,repeat=None):
        frames = sorted(os.listdir(path))
        if not repeat is None:
            if repeat == 1:#正向光流，重复末尾
                frames.append(frames[-1])
            if repeat == -1:#逆向光流，重复头部
                frames.insert(0, frames[0])

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
        return self.coords, self.pixels, self.mask, self.coords_vessel, self.pixels_vessel, self.mask_vessel #坐标、图片灰度、背景分割图
        # return self.coords, self.pixels, self.mask #坐标、图片灰度、背景分割图
    
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
        self._getVesselSet()






