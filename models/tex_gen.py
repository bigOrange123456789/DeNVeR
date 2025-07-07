import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from .blocks import *
from .unet import UNet
import sys
import numpy as np
import os 
sys.path.append("..")
DEVICE = torch.device("cuda")

ROOT = os.path.abspath("__file__/..")
# ROOT = os.path.dirname(BASE_DIR)
print("test_ROOT",ROOT)
# ROOT = "/project/wujh1123/denver"

def resample_textures(texs, coords, random_comp=False):#从全局纹理图中根据给定的坐标重新采样获取每个时刻的纹理
    """
        :param texs (M, C, H, W)        #[背景和血管2层,3通道彩图,全局纹理分辨率256**2]          #[2, 3, 256, 256]
        :param coords (b, m, h, w, 2)   #[9帧数据,背景和血管2层,所有点128**2,每个点的2维纹理坐标]  #[9, 2, 128, 128, 2]
        ***全局纹理图的分辨率比原视频的图片分辨率要高***
    """
    out = {}
    M, C, H, W = texs.shape
    b, m, h, w, _ = coords.shape
    tex_rep = texs[None].repeat(b, 1, 1, 1, 1)  # (b, M, 3, H, W)
    apprs = utils.resample_batch(tex_rep, coords, align_corners=False)
    rgb_apprs = get_rgb_layers(apprs, random_comp=random_comp) #去除透明通道再添加一些扰动(本工作中没有透明通道,只进行了扰动)

    out["raw_apprs"] = apprs
    out["apprs"] = rgb_apprs
    # apprs的形状应为`(b, M, C, h, w)`，即每个批次、每个纹理层、每个通道在目标尺寸`(h, w)`上的采样结果。
    return out

def get_rgb_layers(apprs, random_comp=False):
    '''
     这个函数的目的是将每个层的多通道外观转换为RGB图像，同时可能根据随机合成标志应用一些随机化处理。
     1. 如果 `random_comp` 为 False（默认情况）：
          - 如果 `C == 3`，说明已经是RGB图像，直接返回 `apprs`。
          - 否则（即 `C == 4`），我们假设前3个通道是RGB，第4个通道是透明度（alpha）。那么，我们计算 RGB 乘以 alpha（即 `apprs[:, :, :3] * apprs[:, :, 3:4]`）。注意，这里使用切片保持维度，所以透明度通道被扩展为 `(B, M, 1, H, W)` 以便广播。
     2. 如果 `random_comp` 为 True：
              计算整个图像的平均亮度。
              创建一个二值掩码 `maxed`，标记那些平均亮度低于 `lo` 或高于 `hi` 的像素（即非常暗或非常亮的像素）。
              然后，对于被标记的像素，用随机值替换（使用 `torch.rand_like` 生成与 `apprs` 同样形状的随机数），而未被标记的像素保留原值。
    '''
    """
    :param apprs (B, M, C, H, W)
    """
    B, M, C, H, W = apprs.shape
    if not random_comp:
        return apprs if C == 3 else apprs[:, :, :3] * apprs[:, :, 3:4]

    if C == 3:
        lo = 1e-3 * torch.rand(M, device=apprs.device).view(1, -1, 1, 1, 1)
        hi = 1 - 1e-3 * torch.rand(M, device=apprs.device).view(1, -1, 1, 1, 1)
        avg = apprs.detach().mean(dim=2, keepdim=True)
        maxed = (avg < lo) | (avg > hi)  # (B, M, 1, H, W)
        return ~maxed * apprs + maxed * torch.rand_like(apprs)

    rgb, fac = apprs[:, :, :3], apprs[:, :, 3:4]
    return fac * rgb + (1 - fac) * torch.rand_like(rgb)



class TexUNet(nn.Module):
    def __init__(
        self,
        n_layers,       #2
        target_shape,   #(256, 256) #目标纹理的尺寸
        n_channels=3,   #3          #
        n_levels=3,     #4          #
        d_hidden=32,    #16         #隐藏维度16
        fac=2,          #2          #
        norm_fn="batch",#batch      #
        random_comp=True,#True      #随机补偿
        data_path=None, #CVAI-2828RAO11_CRA11 #数据路径
        **kwargs
    ):
        super().__init__()

        self.d_code = d_hidden // 2     #编码维度 #8
        self.n_layers = n_layers        #层数    #2
        self.random_comp = random_comp  #随机补偿 #True

        tex_init = torch.rand(1, n_layers, self.d_code, *target_shape)#纹理初始化(1,2,8,256,256)
        # print("texture code shape", tex_init.shape)
        self.register_parameter("codes", nn.Parameter(
            tex_init, requires_grad=False)) #设定纹理编码对象self.codes
        # print("codes",self.codes.shape)
        self.data_path = data_path #CVAI-2828RAO11_CRA11
        self.blocks = UNet(#几何分析器和纹理分析器基于同一个UNet网络?
            self.d_code,        #8  #输出MASK的通道数量？
            n_channels,         #3  #输入图片的通道数量
            n_levels=n_levels,  #4  #上/下采样的分级数量
            d_hidden=d_hidden,  #16 #隐含层维度
            fac=fac,            #2  #池化窗口的大小
            norm_fn=norm_fn,    #batch
        )#使用UNet框架获取
        # print(self.d_code,"self.d_code")
        # print({
        #     "self.d_code":self.d_code,  # 8  #输出MASK的通道数量？
        #     "n_channels":n_channels,  # 3  #输入图片的通道数量
        #     "n_levels" : n_levels,  # 4  #上/下采样的分级数量
        #     d_hidden : d_hidden,  # 16 #隐含层维度
        #     fac : fac,  # 2  #池化窗口的大小
        #     norm_fn : norm_fn,
        # })
        # exit(0)
        print("我认为上一句代码原作者写错了,正确的写法见注释《./models/tex_gen.py TexUNet()》")
        # self.blocks = UNet(  # 几何分析器和纹理分析器基于同一个UNet网络?
        #     n_classes=self.d_code,  # 8  #输出MASK的通道数量
        #     n_channels=n_channels,  # 3  #输入图片的通道数量
        #     n_levels=n_levels,  # 4  #上/下采用的分级数量
        #     d_hidden=d_hidden,  # 16 #隐含层维度
        #     fac=fac,  # 2  #池化窗口的大小
        #     norm_fn=norm_fn,  # batch
        # )  # 使用UNet框架获取
        '''
        n_channels, #3  #输入图片的通道数量
        n_classes,  #1  #输出MASK的通道数量
        n_levels=3, #2  #上/下采用的分级数量
        d_hidden=16,#24 #隐含层维度
        fac=2,      #2  #池化窗口的大小
        norm_fn="batch",#batch
        init_std=0, #0.1#
        '''

    def forward(self, coords=None, vis=False):
        """
        returns the per-layer textures, optionally resamples according to provided coords
        返回每层纹理，可选地根据提供的坐标重新采样
        :param coords (B, M, H, W, 2) optional
        :returns apprs (B, M, 3, H, W), texs (M, 3, H, W)
        """
        x = self.codes[0]  # (M, D, H, W) #纹理初始化(1,2,8,256,256) #血管和背景两张纹理
        # print("x.shape",x.shape)
        # print("self.blocks(x)",self.blocks(x).shape)
        texs = torch.sigmoid(self.blocks(x))  # (M, 3, H, W)
        #我感觉过一遍UNet就是多此一举，直接将这张图片视为可学习的向量就可以
        #直接将这张图片视为可学习的向量则坐标之间不连续，可以用类似NerF的方法
        '''
            x: [2, 8, 256, 256]
            blocks(x): [2, 3, 256, 256]
            texs: [2, 3, 256, 256]
            ------------------------------
            输入两个8通道的编码图片
            输出两个3通道的全景纹理图
        '''
        texs_clone = texs.clone() #这里难道不会影响前面的那些参数的学习么？
        with torch.no_grad():
            back_ground = cv2.imread(
                f"{ROOT}/nirs/{self.data_path}/scene.png").astype(np.float32) / 255
            width, height, _ = back_ground.shape
            back_ground = cv2.resize(back_ground, (width*2, height*2))
            back_ground = cv2.cvtColor(back_ground, cv2.COLOR_BGR2RGB)
            back_ground = torch.from_numpy(back_ground).permute(2, 0, 1)
        texs_clone[1] = back_ground
        # texs = texs.detach()
        out = {"texs": texs_clone[None]}  # (1, M, 3, H, W) #[1, 2, 3, 256, 256]#两张全局纹理图

        if coords is not None:
            random_comp = self.random_comp and not vis
            out.update(resample_textures(texs, coords, random_comp))
        '''
            out=[  
                texs,      #两张全局纹理图:不可学习的背景纹理和可学习的血管纹理 
                raw_apprs, #每帧的纹理(没有加噪声)
                apprs      #每帧的纹理(添加了噪声)
            ] #在本项目中全局纹理的边长是单帧纹理的2倍
        '''
        return out
