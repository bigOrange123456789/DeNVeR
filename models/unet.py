import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import *


class UNet(nn.Module):
    def __init__(
        self,
        n_channels, #3  #输入图片的通道数量
        n_classes,  #1  #输出MASK的通道数量
        n_levels=3, #2  #上/下采用的分级数量
        d_hidden=16,#24 #隐含层维度
        fac=2,      #2  #池化窗口的大小
        norm_fn="batch",#batch
        init_std=0, #0.1#
    ):
        super().__init__()
        # exit(0)
        enc_dims = get_unet_dims(n_levels, d_hidden)
        self.enc = Encoder(n_channels, enc_dims, norm_fn=norm_fn, fac=fac)#编码器
        self.dec = Decoder(n_classes, enc_dims, norm_fn=norm_fn, fac=fac) #解码器
        self.in_dims = self.enc.in_dims + self.dec.in_dims
        self.out_dims = self.enc.out_dims + self.dec.out_dims
        # print("fac",fac)
        # print("in_dims",self.in_dims,"")

        ## init the last layer #初始化最后一层
        if init_std > 0:
            print("init last zero")
            init_normal(self.dec.outc, std=init_std)
        else:
            print("init last kaiming")
            init_kaiming(self.dec.outc)

        # print("n_classes", n_classes)
        # print({
        #     "n_channels":n_channels,  # 3  #输入图片的通道数量
        #     "n_classes":n_classes,  # 1  #输出MASK的通道数量
        #     "n_levels":n_levels ,  # 2  #上/下采用的分级数量
        #     d_hidden :d_hidden,  # 24 #隐含层维度
        #     fac :fac,  # 2  #池化窗口的大小
        #     norm_fn :norm_fn,  # batch
        #     init_std :init_std
        # })

    def forward(self, x, idx=None, ret_latents=False, **kwargs):
        # print("x_shape",x.shape)
        z, dn_latents = self.enc(x)
        out, up_latents = self.dec(z, dn_latents, ret_latents=ret_latents)
        if ret_latents:
            dn_latents.append(z)
            return out, dn_latents, up_latents
        return out


class Encoder(nn.Module):#这个编码器能否直接分析前后帧之间的关系?不能，这只是并行处理了一批图片。
    def __init__(self, n_channels, dims, norm_fn="batch", fac=2):
        # n_channels=3 dims=[24, 48, 48] batch=2
        super().__init__()
        self.n_channels = n_channels #输入是3通道的彩色图像
        self.in_dims, self.out_dims = dims[:-1], dims[1:]
        # 输入维度是除最后一个层外其他层的维度，输出维度是除第一层外其他层的维度
        # in_dims=[24, 48] ; out_dims=[48, 48]

        self.inc = ConvBlock(n_channels, dims[0], norm_fn=norm_fn)#[3,24]

        self.down_layers = nn.ModuleList([])
        for d_in, d_out in zip(self.in_dims, self.out_dims):
            print("Down layer", d_in, d_out)#[24,48] [48,48]
            self.down_layers.append(DownSkip(d_in, d_out, norm_fn=norm_fn, fac=fac))
            # 这个UNet网络很神奇，下采样过程中只增加通道数、不降低分辨率

        print("INITIALIZING WEIGHTS")
        self.apply(init_kaiming)

    def forward(self, x):
        x = self.inc(x)
        dn_latents = []
        for layer in self.down_layers:
            dn_latents.append(x)
            x = layer(x)
        return x, dn_latents #输出最终的下采样结果和每一层的输出


class Decoder(nn.Module):
    def __init__(self, n_classes, dims, norm_fn="batch", fac=2):
        # n_classes:1, dims:[24, 48, 48], norm_fn:batch, fac:2
        super().__init__()
        self.n_classes = n_classes #1

        dims = dims[::-1]   # [48,48,24] #逆向取值
        d_in = dims[0]      # 24
        skip_dims = dims[1:]# [48,48]
        out_dims = []       #

        self.up_layers = nn.ModuleList([])
        for d_skip in skip_dims:
            d_out = d_skip
            print("Up layer", d_in, d_out)
            self.up_layers.append(UpSkip(d_in, d_skip, norm_fn=norm_fn, fac=fac)) #上采样层
            # print("d_in, d_skip, norm_fn, fac",d_in, d_skip, norm_fn, fac)
            d_in = d_out + d_skip #输入的维度为上一层的输出和编码器对等层的输出
            out_dims.append(d_in)

        self.outc = nn.Conv2d(d_in, n_classes, kernel_size=1)

        self.in_dims = dims[0:1] + out_dims[:-1]
        self.out_dims = out_dims

        print("INITIALIZING WEIGHTS")
        self.apply(init_kaiming)

    def forward(self, x, dn_latents, ret_latents=False, **kwargs):
        up_latents = []
        for layer, z in zip(self.up_layers, dn_latents[::-1]):
            x = layer(x, z, ret_latents=ret_latents)
            if ret_latents:
                up_latents.append(x)
        out = self.outc(x)
        if ret_latents:
            return out, up_latents
        return out, None


def get_unet_dims(n_levels, d_hidden): #n_levels上/下采样的次数 ; d_hidden初始隐含层的维度
    #n_levels: 2 d_hidden: 24
    dims = [d_hidden * 2 ** i for i in range(n_levels)]#计算每一层的维度
    # dims: [24, 48]
    dims.append(dims[-1])#最后一层重复设置？
    # dims: [24, 48, 48]
    return dims


class DownSkip(nn.Module):
    def __init__(self, d_in, d_out, norm_fn="batch", nl_fn="relu", fac=2):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(
                d_in,
                d_out,
                kernel_size=3,
                padding=1,
                stride=1,
                norm_fn=norm_fn,
                nl_fn=nl_fn,
            ),
            nn.MaxPool2d(fac),#fac是池化窗口的尺寸和池化步长
        )

    def forward(self, x):
        return self.block(x)


class UpSkip(nn.Module):
    def __init__(self, d_in, d_out, norm_fn="batch", nl_fn="relu", fac=2):
        super().__init__()

        # use the normal convolutions to reduce the number of channels
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=fac, mode="bilinear", align_corners=False),
            ConvBlock(
                d_in,
                d_out,
                kernel_size=3,
                padding=1,
                stride=1,
                norm_fn=norm_fn,
                nl_fn=nl_fn,
            ),
        )

    def forward(self, x1, x2, ret_latents=False):
        x1 = self.block(x1)
        x1 = pad_diff(x1, x2)
        x1 = torch.cat([x1, x2], dim=1)
        return x1
