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

        self.path = path
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform

        self.video = self.get_video_tensor(path)
        if self.useMask:
            self.mask_ = 1-self.get_video_tensor(maskPath)
        self.num_frames, _, self.H, self.W = self.video.size()

        self.pixels = self.video.permute(2, 3, 0, 1).contiguous().view(-1, self.numChannel)
        if self.useMask:
            self.mask = self.mask_.permute(2, 3, 0, 1).contiguous().view(-1, self.numChannel)
        self.coords = get_mgrid([self.H, self.W, self.num_frames])

        shuffle = torch.randperm(len(self.pixels))
        self.pixels = self.pixels[shuffle]
        self.coords = self.coords[shuffle]
        if self.useMask:
            self.mask = self.mask[shuffle]
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
        return self.coords, self.pixels, self.mask






