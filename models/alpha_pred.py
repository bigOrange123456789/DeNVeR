import numpy as np
import torch
import torch.nn as nn

from . import blocks
from .unet import UNet

import sys

sys.path.append("..")
import utils


class AlphaModel(nn.Module):#进行视频分割的UNet模型，输入视频，输出每个对象的MASK。
    def __init__(self, n_layers, net_args, d_in=3, fg_prob=0.1,
                 pathParam="/home/lzc/桌面/DeNVeR/../DeNVeR_in/models_config/freecos_Seg.pt",
                 useFreeCOS=True,
                 **kwargs):
        super().__init__()
        '''
            n_layers: 2
            net_args: {'n_levels': 2, 'd_hidden': 24, 'fac': 2, 'norm_fn': 'batch', 'init_std': 0.1}
            d_in: 3 
            fg_prob: 0.1
        '''
        assert n_layers > 1
        self.n_outputs = n_layers - 1 #1
        self.n_layers = n_layers      #2
        # fg_prob = 0.01
        ## recenter the nonlinearity such predicted 0 maps to fg_prob
        ##  1 / (1 + exp(-(x - shift))) = fg_prob
        ## x = 0 --> shift = log(1/fg_prob - 1)
        bg_shift = np.log(1 / fg_prob - 1)#背景打分
        '''
            fg_prob应该是一个标量，表示前景（foreground）的概率。
            bg_shift = log[p / (1 - p)]
                    y=log[p/(1-p)]将(0~1)转换为(-无穷，+无穷)
        '''
        n_remaining = torch.arange(1, self.n_outputs).float()
        '''
            生成一个从 1 到 outputs-1 的浮点型张量
            这里n_remaining=tensor([])
        '''
        print('n_remaining', n_remaining)
        # exit(0)
        fg_shift = np.log(2 * n_remaining - 1)#前景打分？ #打分为[]
        # print("fg_shift",fg_shift,"I guess [].")
        shift = torch.cat([fg_shift, torch.ones(1) * bg_shift])
        self.register_buffer("shift", shift.view(1, -1, 1, 1))
        print("shifting zeros to", self.shift) #将零转换为张量[[[[2.1972]]]] #这个偏移值是啥?

        '''
            d_in: 3
            n_outputs: 1
            net_args: {
                n_levels: 2, 
                d_hidden: 24, 
                fac: 2, 
                norm_fn: 'batch', 
                init_std: 0.1
            }
            ----------------------------------------------------------
            我猜这是一个多分类的UNet网络，其中n_outputs表示除了背景之外的类别数量。
            只有生成精灵图的时候才需要固定编码。
        '''
        ###################### 这里加载FreeCOS的分割模型 ######################
        # print("self.shift",self.shift)
        self.useFreeCOS=useFreeCOS #True
        if self.useFreeCOS:
            from free_cos.ModelSegment import ModelSegment
            n_channels = 1
            num_classes = 1
            self.Segment_model = ModelSegment(n_channels, num_classes).cuda()
            checkpoint = torch.load(pathParam, map_location=torch.device('cpu'))  # 如果模型是在GPU上训练的，这里指定为'cpu'以确保兼容性
            self.Segment_model.load_state_dict(checkpoint['state_dict'])
        else:
            self.backbone = UNet(d_in, self.n_outputs, **net_args) #这是一个UNet网络
            # 一个关键的问题：这个UNet能否直接分析前后帧之间的关系？不能，这只是并行处理了一批图片。应该将这10张3通道图片变为1张10通道图片。

    def forward(self, x, **kwargs):
        """
        :param x (N, C, H, W)
        """
        if self.useFreeCOS:
            x = (x[:,0:1]+x[:,1:2]+x[:,2:3])/2
            x = (x - torch.mean(x)) / torch.std(x)
            pred = self.Segment_model(x)["pred"]
        else:
            pred = self.backbone(x, **kwargs)  # (N, M-1, H, W)
        # # x: [10, 3, 128, 128]    #应该是10张3通道的图像
        # # pred: [10, 1, 128, 128] #应该是MASK图片
        return self._pred_to_output(pred)

    def _pred_to_output(self, x): #这个函数的主要作用是考虑层间遮挡，但对单层数据来说用处不大
        """
        turn model output into layer weights 将模型输出转换为层权重
        :param x (B, M-1, H, W)      #x.shape=[10,1,128**2] #x是单通道
        """
        if self.useFreeCOS:
            x = x.unsqueeze(2)
            return {
                "masks": torch.cat([x, 1-x], dim=1),
                "alpha": torch.ones_like(x),
                "pred": x
            }
        ## predict occupancies 预测占用率
        if not self.useFreeCOS:
            x = x - self.shift #预测结果-偏移值
        '''
            shift.shape = [1, 1, 1, 1]
            x.shape = [10, 1, 128**2]
        '''
        x = torch.sigmoid(x).unsqueeze(2)  # (B, M-1, 1, H, W) #[9, 1, 1, 128, 128]
        '''
            sigmoid函数将打分转为概率
            通过unsqueeze增加了一个维度(这么操作的意义是什么?)
            x.shape = [10, 1, 1, 128, 128]
        '''

        ## we predict the complement of bg to initialize all weights near 0
        ## 我们预测bg的补码，以初始化接近0的所有权重
        fg = x[:, :-1]  # (B, M-2, ...) #[0]
        bg = 1 - x[:, -1:]  # (B, 1, ...) #[1-x]
        '''
            fg.shape = [10, 0, 1, 128, 128] 
            bg.shape = [10, 1, 1, 128, 128] 
            前几层作为前景概率
            最后一层取反作为背景概率
            ---------------------------------
            在这个问题中只有背景层，没有前景层
        '''
        occ = torch.cat( # [0,前景,1] #[0,1]
            [torch.zeros_like(x[:, :1]), fg, torch.ones_like(x[:, :1])], dim=1
        )
        '''
            在前景预测组中，前面加一组0、后面加一组1。
            [10, 1+0+1, 128, 128]
            ----------------------------------
            x[:, :1].shape = [10, 1, 128, 128]
            torch.zeros_like(x[:, :1]).shape = [10, 1, 128, 128]
            fg.shape = [10, 0, 1, 128, 128]
            occ.shape = [10, 2, 128, 128]
            ----------------------------------
            occ是前景对背景的遮挡情况
        '''

        ## compute visibility from back to front 计算从后到前的能见度
        ## v(1) = 1
        ## v(2) = (1 - o(1))
        ## v(M-1) = (1 - o(1)) * ... * (1 - o(M-2))
        ## v(M) = 0
        vis = torch.cumprod(1 - occ, dim=1) # 沿着ID=1的维度计算累积乘积 #[1,0]
        '''
            vis有两层: 0号层全1、1号层全0
            每一层与上一层的结果再进行相乘
                vis.shape = [9, 2, 1, 128, 128]
                vis.mean = 0.5 #总共两层，1层全0、另一层全1,因此均值为0.5
        '''
        # print("vis[:, :-1]",vis[:, :-1].shape,torch.mean(vis[:, :-1]))
        acc = vis[:, :-1] * occ[:, 1:]  # (B, M-1, ...) #该层显现并且前面的几层都不可见(前景层真正的可见概率) #[1]
        # print("occ[:, 1:]",occ[:, 1:].shape,torch.mean(occ[:, 1:]))
        # exit(0)
        # acc、vis[:, :-1]、occ[:, 1:]这三者都是一层全1数据
        # print('acc', acc.shape)
        # print(vis.shape,"vis")
        # exit(0)
        # print("(1 - bg) * acc",acc.shape,torch.mean((1 - bg) * acc))
        # print("bg",torch.mean(bg),bg.shape)
        # print("test", torch.mean(bg-((1 - bg) * acc)))#
        # exit(0)
        weights = torch.cat([(1 - bg) * acc, bg], dim=1) #只有背景没有前景的时候这两个通道的内容也不一致 #[x,1-x]
        # 这里我很不理解：这里的意思是背景会遮挡前景？
        '''
            vis: [10, 2, 1, 128, 128]
            occ: [10, 1, 1, 128, 128]
            vis[:, :-1]: [10, 1, 1, 128, 128] #去除最后面的那组
            occ[:, 1:]:  [10, 1, 1, 128, 128] #去除最前面的那组
            weights: [10, 2, 1, 128, 128]
            vis是可见情况: 只有当没有被任何前景层遮挡的时候背景层才可见。
            acc是考虑可见性后的视频MASK
            -------------------------------------------------
            vis[:, :-1]
                    “:-1”表示去掉最后面的数据。
            occ[:, 1:]
                    “1:”表示去掉最前面的数据。
            -------------------------------------------------
            三个返回值的含义：
                masks：进行背景遮挡和层间遮挡(包含背景和全部前景层)
                alpha：只进行层间遮挡
                pred ：不进行遮挡
            解释一下为啥三个返回值的通道数量不一致
        '''
        # 输入是单通道
        # masks: 双通道(1个通道表示血管、另一个表示1-血管), alpha:单通道(全1数据), pred:单通道(mask[0])
        '''
            weights:[10, 2, 1, 128, 128] 1=weights[:,0]+weights[:,1] [x,1-x]
            acc    :[10, 1, 1, 128, 128] 1=acc                       [1]
            x      :[10, 1, 1, 128, 128]           
        '''
        return {"masks": weights, "alpha": acc, "pred": x}



