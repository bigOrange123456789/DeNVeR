import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import utils
import data
from skimage import io, color, morphology
import cv2
from torch.utils.tensorboard import SummaryWriter
import time
time_value = time.time()

class ReconLoss(nn.Module):
    """
    Mixed L2 and Laplacian loss
    """

    def __init__(
        self,
        weight,
        lap_ratio=1e-3,
        norm=1,
        n_levels=3,
        ksize=3,
        sigma=1,
        detach_mask=True,
    ):
        super().__init__()
        self.weight = weight
        self.lap_ratio = lap_ratio
        self.abs_fnc = torch.abs if norm == 1 else torch.square
        self.n_levels = n_levels
        self.ksize = ksize
        self.sigma = sigma
        self.detach_mask = detach_mask

    def forward(self, batch_in, batch_out, split=False):
        '''
            batch_in:
                [rgb, fwd, bck, epi, occ, disocc, ske, idx]
            batch_out:
                [masks, alpha, pred, coords, view_grid, fgcoords, texs, raw_apprs, apprs, recons, layers]
        '''
        rgb = batch_in["rgb"] #原始视频数据 #10张图片的视频: [10, 3, 128, 128]
        # print("rgn:", rgb.shape)
        if self.weight <= 0:
            return None
        if self.detach_mask: #True
            appr = batch_out["apprs"]         #逐帧纹理     #[10, 2, 3, 128, 128]
            mask = batch_out["masks"].detach()#预测出的MASK #[10, 2, 1, 128, 128]
            # print("appr:",appr.shape)
            # print("mask:",mask.shape)
            recon = (mask * appr).sum(dim=1) #重构后的视频
            # [10, 3, 128, 128] <= [10, 2, 3, 128, 128] <= [10, 2, 1, 128, 128]*[10, 2, 3, 128, 128]
        else:
            recon = batch_out["recons"]

        l2_err = self.abs_fnc(recon - rgb).mean() #计算重构后的视频与原视频的差异

        recon_lp = utils.get_laplacian_pyr( # 生成重建图像的金字塔
            recon, self.n_levels, self.ksize, self.sigma)
        target_lp = utils.get_laplacian_pyr( # 生成真实图像的金字塔
            rgb, self.n_levels, self.ksize, self.sigma)
        H, W = recon.shape[-2:]
        lap_err = sum( # 计算金字塔损失
            [
                torch.abs(recon_lp[i] - target_lp[i]).sum() * (4 ** i)
                for i in range(self.n_levels)
            ]
        ) / (H * W)

        return self.weight * (l2_err + self.lap_ratio * lap_err)

from Test import Test
class EpipolarLoss(nn.Module):
    """
    Penalize background layer's pixels with high sampson error,
    lightly penalizes the foreground layers' pixels with low sampson error
    """

    def __init__(self, weight, neg_ratio=2e-3, clip=10.0):
        super().__init__()
        self.weight = weight # cfg.w_epi=0.5 #一个固定权重
        self.neg_ratio = neg_ratio
        self.clip = clip

    def whiten_distance(self, err):
        e_max = err.max()
        #         e_max = torch.clamp_min(e_max, self.clip)
        err = torch.clamp_max(err, e_max) / e_max #没有考虑最大值为0的情况
        return err # 简单来说就是除以最大值、从而让值域范围变为(0,1]

    def forward(self, batch_in, batch_out):
        """
        we pre-compute sampson error with a fundamental matrix computed with LMeDS,
        and threshold the sampson error with the median
        我们用LMeDS计算的基本矩阵预先计算桑普森误差，并用中值对桑普森误差进行阈值处理
        """
        ok, err, _ = batch_in["epi"]  # (B, H, W) # 黑塞矩阵MASK
        if ok.sum() < 1: # err.shape = [10, 128**2]
            return None

        masks = batch_out["masks"][ok]  # (B, M, 1, H, W) #几何分支得到的血管MASK # masks.shape = [10, 2, 1, 128**2]
        # Test(masks[4:6,-1,0])
        err = self.whiten_distance(err[ok]) # 通过归一化将最大值变为1 #血管区域是白色
        bg_mask = masks[:, -1, 0] # (B, H, W) # [9, 128**2]<-[9, 2, 1, 128**2]
        bg_loss = bg_mask * err # 血管的交集
        # print(bg_loss.shape,"bg_loss")
        # Test(bg_loss[4:6])
        # exit(0)
        # 0.002
        # self.neg_ratio = 0.2
        # print("neg_ratios",self.neg_ratio)
        # exit(0)
        fg_loss = self.neg_ratio * (1 - bg_mask) * (1 - err) #neg_ratios=0.15 #非血管区域的交集
        loss = bg_loss + fg_loss #损失函数减小=>{血管区域变为0,非血管区域变为1}

        return self.weight * loss.mean() #计算所有像素点的均值

    def vis(self, batch_in, batch_out):
        ok, err, _ = batch_in["epi"]  # (B, H, W)
        err = self.whiten_distance(err[ok])
        return {"epi": err[:, None, None]}


class Parallelloss(nn.Module):
    """
    parallel loss
    """

    def __init__(self, weight=0.05, neg_ratio=2e-3, clip=10.0):
        super().__init__()
        self.weight = weight
        self.global_step = 0 

    def compute_skeleton_direction(self, skeleton_mask): # 基于OpenCV的骨架方向计算
        gradient_x = cv2.Sobel(skeleton_mask, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(skeleton_mask, cv2.CV_64F, 0, 1, ksize=3)
        gradient_direction = np.arctan2(gradient_y, gradient_x)

        return gradient_direction

    def compute_skeleton_gradient(self, skeleton_mask): # 基于OpenCV的骨架梯度计算
        gradient_x = cv2.Sobel(skeleton_mask, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(skeleton_mask, cv2.CV_64F, 0, 1, ksize=3)
        # gradient_direction = np.arctan2(gradient_y, gradient_x)

        return gradient_x, gradient_y

    def tensorcompute_skeleton_direction(self, skeleton_mask): # 基于PyTorch的骨架方向计算
        sobel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)

        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        conv2d_x = nn.Conv2d(1, 1, kernel_size=3, bias=False)
        conv2d_y = nn.Conv2d(1, 1, kernel_size=3, bias=False)
        conv2d_x.weight.data = sobel_x
        conv2d_y.weight.data = sobel_y
        gradient_x = conv2d_x(skeleton_mask)
        gradient_y = conv2d_y(skeleton_mask)
        gradient_direction = torch.atan2(gradient_y, gradient_x)

        return gradient_direction
    
    def tensorcompute_skeleton_gradient(self, skeleton_mask): # 骨架梯度计算
        sobel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)

        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        conv2d_x = nn.Conv2d(1, 1, kernel_size=3, bias=False)
        conv2d_y = nn.Conv2d(1, 1, kernel_size=3, bias=False)
        conv2d_x.weight.data = sobel_x
        conv2d_y.weight.data = sobel_y
        gradient_x = conv2d_x(skeleton_mask)
        gradient_y = conv2d_y(skeleton_mask)
        # gradient_direction = torch.atan2(gradient_y, gradient_x)

        return gradient_x,gradient_y
    def forward(self, batch_in, batch_out):
        """
        we pre-compute sampson error with a fundamental matrix computed with LMeDS,
        and threshold the sampson error with the median
        -----------------------------------------------------
        目的：计算骨架方向与光流方向之间的平行性损失
        核心思想：鼓励光流方向与骨架方向保持一致（平行或反平行）
        输入：
            batch_in：包含输入数据（如骨架图）
            batch_out：包含模型输出（如光流场）
        输出：加权后的平行性损失值
        """
        # 1. 提取光流和骨架数据
        flow = batch_out["coords"][:, 0, :, :] #通过样条参数获取的光流图？
        # [9, 128, 128, 2]<=[9, 2, 128, 128, 2] #提取前景的运动 #为啥我感觉coords是不可学习的固定值
        ske = batch_in["ske"][:, 0, :, :]  # (B, 3, H, W) # 获取骨架图 (B, 3, H, W)
        # [9, 128, 128]<=[9, 3, 128, 128]       #血管的骨架图视频
        # flow = flow[:, 0, :, :]  # (B, 2, H, W)

        # 分离光流的u,v分量
        u = flow[:, :, :, 0]  # 水平分量 (B, H, W) #[9, 128, 128]
        v = flow[:, :, :, 1]  # 垂直分量 (B, H, W)

        # 2. 重组光流向量
        # 展平光流分量
        tensor1 = u.contiguous().view(-1).unsqueeze(0)  # 水平分量展平
        tensor2 = v.contiguous().view(-1).unsqueeze(0)  # 垂直分量展平
        # 组合成光流向量 (2, N) 其中 N = B×H×W
        u_v_vector = torch.cat((tensor1, tensor2), dim=0)
        # 将光流重组为二维向量场，每个像素一个向量
        # 形状变化：(B, H, W) → (1, B×H×W) → (2, B×H×W)

        # 3. 计算骨架梯度方向
        B,H,W = ske.shape
        device = ske.device
        with torch.no_grad():
            skeletonized_mask = ske.cpu().numpy()
        self.global_step+=1
        # print(ske.shape)
        # print(skeletonized_mask.shape)
        skeleton_direction_x, skeleton_direction_y = self.compute_skeleton_gradient(
            skeletonized_mask)
        # skeleton_direction_x:<numpy>(10, 128, 128)

        # 4. 重组骨架梯度向量
        x = torch.from_numpy(skeleton_direction_x).to(device)
        y = torch.from_numpy(skeleton_direction_y).to(device)
        # x:<Tensor>[10, 128, 128]

        tensor1 = x.view(-1).unsqueeze(0)
        tensor2 = y.view(-1).unsqueeze(0)
        x_y_vector = torch.cat((tensor1, tensor2), dim=0)

        # 5. 计算余弦相似度损失
        cosine_similarity = F.cosine_similarity(u_v_vector, x_y_vector, dim=0) # 计算余弦相似度 (N,)
        abs_cosine_similarity = torch.abs(cosine_similarity) # 取绝对值（考虑平行和反平行）
        cosine_similarity_loss = abs_cosine_similarity.mean() # 计算平均损失
        # print(cosine_similarity_loss)

        # 6. 返回加权损失
        return self.weight * cosine_similarity_loss        

def get_stats(X, norm=2):
    """
    :param X (N, C, H, W)
    :returns mean (1, C, 1, 1), scale (1)
    """
    mean = X.mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
    if norm == 1:
        mag = torch.abs(X - mean).sum(dim=1)  # (N, H, W)
    else:
        mag = np.sqrt(2) * torch.sqrt(torch.square(X -
                                                   mean).sum(dim=1))  # (N, H, W)
    scale = mag.mean() + 1e-6
    return mean, scale


class FlowGroupingLoss(nn.Module):
    def __init__(
        self,
        weight,
        norm=1,
        sep_fac=0.1,
        bg_fac=2.0,
        detach_mean=True,
    ):
        super().__init__()
        self.weight = weight
        self.norm_fnc = torch.abs if norm == 1 else torch.square
        self.detach_mean = detach_mean
        self.sep_fac = sep_fac
        self.bg_fac = bg_fac

    def forward(self, batch_in, batch_out, split=False):
        """
        :param masks (*, 1, H, W)
        :param src (*, C, H, W)
        """
        ok, flow = batch_in["fwd"]

        if ok.sum() < 1:
            print("NO FLOWS")
            return None

        masks = batch_out["masks"][ok]

        B, M, _, H, W = masks.shape
        device = masks.device

        flow = flow[ok]  # (B, 2, H, W)
        f_mean, f_std = get_stats(flow)
        flow = ((flow - f_mean) / f_std).unsqueeze(1)

        with torch.no_grad():
            mass = masks.sum(dim=(-1, -2), keepdim=True) + 1e-6
            mean = (masks * flow).sum(dim=(-1, -2), keepdim=True) / mass

        dists = self.norm_fnc(flow - mean).sum(dim=2, keepdim=True)

        fac = torch.cat(
            [
                torch.ones(M - 1, device=device),
                self.bg_fac * torch.ones(1, device=device),
            ]
        )
        rand = torch.cat(
            [
                torch.zeros(1, device=device),
                self.sep_fac * (torch.rand(M - 1, device=device) + 1),
            ]
        )
        masks = masks * (fac + rand).view(1, -1, 1, 1, 1)

        wdists = masks * dists
        return self.weight * wdists.mean()


class FlowWarpLoss(nn.Module): #UV的变化情况应与光流一致
    def __init__(
        self,
        weight,
        tforms,
        fg_tforms,
        gap,
        src_name="fwd",
        norm=1,
        unscaled=False,
        detach_mask=True,
    ):
        """
        Loss that supervises the view->cano transform for each frame
        For a point A in view, T1(A) takes A -> A' in cano
        FLOW_12(A) takes A -> B in view, T2(B) takes B -> B' in cano
        A' and B' should be the same point in cano

        Minimizes distance between A' and B' in cano

        :param gap (int) the spacing between flow pairs
        :param norm (int, optional) the norm to use for distance
        ----------------------------------------------------------------
        利用光流信息来监督视图空间到规范空间的变换
        核心思想
        通过光流约束相邻帧在规范空间中的一致性：
            对于视图中的点 A，使用当前帧变换 T1 将其映射到规范空间 A'
            光流 FLOW_12 将点 A 映射到相邻帧的点 B
            使用相邻帧变换 T2 将点 B 映射到规范空间 B'
            理想情况下 A' 和 B' 应重合（同一点在规范空间）
            损失函数最小化 A' 和 B' 之间的距离
        ----------------------------------------------------------------
        weight,          # 损失权重                      #1
        tforms,          # 背景变换模型（PlanarMotion实例）#
        fg_tforms,       # 前景变换模型（PlanarMotion实例）#
        gap,             # 帧间隔（正=后向，负=前向）       #1
        src_name="fwd",  # 光流来源（"fwd"或"bwd"）       #fwd
        norm=1,          # 距离范数（1=L1, 其他=L2）      #1
        unscaled=False,  # 是否按缩放因子归一化            #False
        detach_mask=True # 是否分离掩码梯度               #True
        ----------------------------------------------------------------
        weight,tforms,fg_tforms 1 BSplineTrajectory(
          (final_nl): Tanh()
        ) FG_BSplineTrajectory(
          (final_nl): Tanh()
        )
        gap,src_name,norm 1 fwd 1
        unscaled,detach_mask False True
        """
        super().__init__()
        print("Initializing flow warp loss with {} and {}".format(src_name, gap))
        self.weight = weight
        self.tforms = tforms
        self.fg_tforms = fg_tforms
        self.gap = gap
        self.src_name = src_name
        self.norm_fnc = torch.abs if norm == 1 else torch.square #这里采用1范式
        self.detach_mask = detach_mask
        self.unscaled = unscaled

    def forward(self, batch_in, batch_out, split=False, no_reduce=False):
        # 1. 基础准备
        gap = self.gap
        idx = batch_in["idx"] # 帧索引 (B)

        # 检查是否跳过计算
        if self.weight <= 0 or len(idx) < abs(gap):
            return None

        # 获取光流数据 (B,2,H,W) -> (B,H,W,2)
        # flow (B, 2, H, W) -> (B, H, W, 2)
        ok, flow = batch_in[self.src_name]  # 光流有效性标志和光流 (B,2,H,W)
        flow = flow.permute(0, 2, 3, 1)     # 调整维度 (B,H,W,2)
        # flow的帧数和原视频一致，因此会有一个空白帧，空白帧的位置通过ok标出

        # 获取输出数据
        masks = batch_out["masks"]    # (B, M, 1, H, W) # 层掩码 (B,M,1,H,W)   #预测的分割结果 #前景和背景的损失分开计算
        coords = batch_out["coords"]  # (B, M, H, W, 2) # 规范坐标 (B,M,H,W,2) #UV
        grid = batch_out["view_grid"] # (B, M, H, W, 3) # 视图网格 (B,M,H,W,3) #不可学习的固定值

        # 2. 掩码处理
        B, M, _, H, W = masks.shape

        if self.detach_mask:  # primarily fitting the transform
            masks = masks.detach()  # 分离掩码梯度（专注优化变换）#此时的MASK计算不用于梯度优化

        masks = masks.view(B, M, H, W)  # 调整形状 (B,M,H,W)

        # 3. 构建帧对
        if gap > 0:  # 正间隔（向前看）
            # 0 ... B-1-gap #去掉最后一帧
            W1 = masks[:-gap]   # 新视频掩码
            V1 = ok[:-gap]      # 新视频光流有效性
            I1 = idx[:-gap]     # 新视频索引
            P1 = coords[:-gap]  # 新视频UV
            G1 = grid[:-gap]    # 新视频像素坐标
            F12 = flow[:-gap]   # 新视频光流(当前帧→目标帧光流)
            # gap ... B-1 #去除第一帧
            I2 = idx[gap:]      # 新视频索引
            V2 = ok[gap:]       # 新视频光流有效性
        else:   # 负间隔（向后看）
            # gap ... B-1
            W1 = masks[-gap:]
            V1 = ok[-gap:]
            I1 = idx[-gap:]
            P1 = coords[-gap:]
            G1 = grid[-gap:]
            F12 = flow[-gap:]
            # 0 ... B-1-gap
            I2 = idx[:gap]
            V2 = ok[:gap]

        # 4. 筛选有效数据
        valid = V1 & V2 # 1和2都有有效光流的位置
        I1 = I1[valid]  # 有效索引
        W1 = W1[valid]  # 有效掩码
        P1 = P1[valid]  # 有效UV
        G1 = G1[valid]  # 有效像素坐标
        F12 = F12[valid]# 有效光流
        I2 = I2[valid]  # 有效目标帧索引

        # 5. 计算目标点位置
        F12 = torch.cat([F12, torch.zeros_like(F12[..., -1:])], dim=-1)
        # [7, 128, 128, 3] <= ([7, 128, 128, 2], [7, 128, 128, 1])
        # 将光流位移(dx,dy)变为(dx,dy,0)
        G2 = G1 + F12[:, None] #通过光流图计算新的坐标
        #  (x2,y2,1) = (x1,y1,1) + (dx,dy,0)
        G2_bg = G2[:,1,:,:,:].unsqueeze(1) #背景v_pos
        G2_fg = G2[:,0,:,:,:].unsqueeze(1) #前景v_pos
        P2_bg = self.tforms(I2, grid=G2_bg) #根据运动参赛，计算每一帧的像素位置
        P2_fg = self.fg_tforms(I2, grid=G2_fg) #
        P2 = torch.cat((P2_fg, P2_bg), dim=1)
        # rescale to pixel coordinates (0 - W, 0 - H)

        # 6. 计算坐标差异（核心）
        scale_fac = (W + H) / 4

        # self.unscaled:False
        if self.unscaled: # 缩放归一化模式
            # 1. 获取当前帧(I1)的缩放因子
            s1_bg = self.tforms.get_scale(I1)    # 背景层缩放因子 (B, M_bg) #不处理背景层，这里计算背景层的放缩因子也没有意义
            s1_fg = self.fg_tforms.get_scale(I1) # 前景层缩放因子 (B, M_fg)
            s1 = torch.cat((s1_fg, s1_bg), dim=1).view(-1, M, 1, 1, 1) # 合并前景和背景缩放因子

            # 2. 获取目标帧(I2)的缩放因子
            s2_bg = self.tforms.get_scale(I2)    # [8, 1]
            s2_fg = self.fg_tforms.get_scale(I2) # [8, 1]
            s2 = torch.cat((s2_fg, s2_bg), dim=1).view(-1, M, 1, 1, 1) # [8, 2, 1, 1, 1]

            # 3. 特殊处理背景层 #背景层不进行缩放(放缩因子为1)
            s1 = torch.cat([s1[:, :-1], torch.ones_like(s1[:, -1:])], dim=1) # [8, 2, 1, 1, 1]
            s2 = torch.cat([s2[:, :-1], torch.ones_like(s2[:, -1:])], dim=1) # [8, 2, 1, 1, 1]

            # 4. 计算归一化的坐标差异
            diffs = scale_fac * (P1 - P2) / (s1 + s2 + 1e-5) # [8, 2, 128, 128, 2] # 每一张图片除以不同的值
        else: # 标准模式
            diffs = scale_fac * (P2 - P1)

        # 逐个像素点计算1范式再求均值
        wdists = W1 * self.norm_fnc(diffs).sum(dim=-1)
        return self.weight * wdists.mean()

class MaskWarpLoss(nn.Module):#根据光流图预测的下一帧要与真实的下一帧一致
    def __init__(self, weight, gap, norm=1):
        super().__init__()
        assert norm == 1 or norm == 2 or norm == "xent"
        self.gap = gap
        self.weight = weight
        self.norm = norm

    def forward(self, batch_in, batch_out, split=False):
        """
        We consider point A in frame 1. Flow takes A to B in frame 2: FLOW_12(A) -> B.
        The mask value of point A in 1 should be the same as the mask value of point B in 2,
        unless point A is occluded.
        我们考虑帧1中的点A。流程在帧2中从A到B：Flow_12（A）->B。
        1中点A的掩码值应与2中点B的掩码值相同，
        除非点A被遮挡。
        :param masks (B, M, 1, H, W)
        :param flow (B, 2, H, W)
        :param occ_map (B, 1, H, W)
        return (B-gap, M, H, W) distance between the corresponding masks
        """
        gap = self.gap # 1
        masks = batch_out["masks"]  # (B, M, 1, H, W) # [9, 2, 1, 128, 128] #分割的结果

        if len(masks) < abs(self.gap):  # not enough in batch #如果视频太短不够一帧
            return torch.zeros(1, dtype=torch.float32, device=masks.device)

        if gap > 0: #正向光流
            ok, flow = batch_in["fwd"] # (B, 2, H, W) #没有找到对应的文件才会不OK
            # ok=[ True,  True,  True,  True,  True,  True,  True,  True, False]
            # flow.shape=[9, 2, 128, 128] 光流图是双通道
            occ_map = batch_in["occ"][0] #occ.shape=[9, 1, 128, 128] #前向遮挡图
            # 不知道这个遮挡图是个啥
            ok = ok[:-gap]#去除最后一个false
            F12 = flow[:-gap].permute(0, 2, 3, 1)  # 0 ... B-1-gap #[8,128,128,2] #有效的光流图视频
            O12 = occ_map[:-gap] #[8,1,128,128] #遮挡图视频
            M1 = masks[:-gap, :, 0, ...]  # 0 ... B-1-gap #[8, 2, 128, 128] #去除最后一张MASK图
            M2 = masks[gap:, :, 0, ...]   # gap ... B-1   #[8, 2, 128, 128] #去除第一张MASK图
        else: #逆向光流
            ok, flow = batch_in["bck"]
            occ_map = batch_in["disocc"][0]
            ok = ok[-gap:]
            F12 = flow[-gap:].permute(0, 2, 3, 1)  # gap ... B-1
            O12 = occ_map[-gap:]
            M1 = masks[-gap:, :, 0, ...]  # gap ... B-1
            M2 = masks[:gap, :, 0, ...]  # 0 ... B-1-gap

        M1, M2 = M1[ok], M2[ok] #输出的MASK
        F12, O12 = F12[ok], O12[ok] #输入的光流图和遮挡图

        # mask 1 resampled from mask 2
        W1 = utils.inverse_flow_warp(M2, F12, O12) #根据光流图，从终点MASK2得到起始MASK1

        if self.norm == 1: #1范数
            dist = (~O12) * torch.abs(W1 - M1)
        elif self.norm == 2: #2范数
            dist = (~O12) * torch.square(W1 - M1)
        elif self.norm == "xent":
            W1 = W1.detach()  # (B, M, H, W)
            dist = -W1 * torch.log(M1 + 1e-8) - (1 - W1) * \
                torch.log(1 - M1 + 1e-8)
        else:
            raise NotImplementedError

        return self.weight * dist.mean() #计算均值:总和/像素个数

class ContrastiveTexLoss(nn.Module):
    '''
    这个类实现了一个对比纹理损失，用于在多层场景中鼓励不同层的纹理特征具有区分性。
    核心思想是惩罚不同层在相同空间位置具有相似外观的情况，从而促进层间的分离。
    '''
    def __init__(self, weight, thresh=0.25, use_mask=False, detach_mask=False):
        super().__init__()
        self.weight = weight # 损失权重
        self.thresh = thresh # 掩码阈值(默认0.25)
        self.use_mask = use_mask # 是否使用掩码
        self._detach_mask = detach_mask # 是否分离掩码梯度

    def detach_mask(self):
        self._detach_mask = True # 分离掩码梯度

    def attach_mask(self):
        self.use_mask = True     # 启用掩码
        self._detach_mask = False# 保持掩码梯度

    def forward(self, batch_in, batch_out, split=False): #对输出数据本身进行分析
        # 1. 获取输入数据 #apprs.shape=[9, 2, 3, 128, 128]=[9张图片, 2个图层, 3个通道RGB, 128**2分辨率]
        apprs = batch_out["apprs"]  # 多层的外观特征张量 (B, M, _, H, W) #每一帧的加噪纹理
        B, M = apprs.shape[:2] # B=批次大小, M=层数

        # 2. 计算层间相似度 #余弦相似度 #彩色图像可以用余弦值逐个像素进行判断，灰度图逐个像素计算余弦值没有用
        # compute the similarity between each pair of layers
        sim = (apprs.unsqueeze(2) * apprs.unsqueeze(1) #只是通过逐个像素的分析简单判断了不同图层的颜色值是否一致
               ).sum(dim=3)  # (B,M,M,H,W)

        # 3. 排除自相似度
        # zero out the diagonals (similarity with itself)
        idcs = torch.arange(M, device=apprs.device)  # (M)
        sim[:, idcs, idcs] = 0 #对称轴位置上的元素为0

        # 4. 掩码处理(可选)
        if self.use_mask: #use_mask=False #本系统中不使用MASK
            # for every layer, apply its mask on the other layers
            # we don't want the other layer appearances to be similar
            # in the regions that should be explained only by this layer
            masks = batch_out["masks"]  # (B, M, 1, H, W)
            if self._detach_mask:
                masks = masks.detach()
            masks = masks > self.thresh
            sim = masks * sim

        # 5. 计算最终损失
        return self.weight * sim.mean()


def compute_losses(loss_fncs, batch_in, batch_out, step=None):
    loss_dict = {}
    for name, fnc in loss_fncs.items():
        if fnc.weight <= 0:
            continue
        # if name =='epi':

        loss = fnc(batch_in, batch_out)
        if loss is None:
            continue
        loss_dict[name] = loss
    return loss_dict


def get_loss_grad(batch_in, batch_out, loss_fncs, var_name, loss_name=None):
    """
    get the gradient of selected losses wrt to selected variables
    Which losses and which variable are specified with a list of tuples, grad_pairs
    """
    # NOTE: need to re-render to re-populate computational graph
    # in future maybe can also retain graph
    var = batch_out[var_name]
    *dims, C, H, W = var.shape
    var.retain_grad()
    sel_fncs = {loss_name: loss_fncs[loss_name]
                } if loss_name is not None else loss_fncs
    loss_dict = compute_losses(sel_fncs, batch_in, batch_out)
    if len(loss_dict) < 1:
        return torch.zeros(*dims, 3, H, W, device=var.device), 0

    try:
        sum(loss_dict.values()).backward()
    except:
        pass

    if var.grad is None:
        print("requested grad for {} wrt {} not available".format(
            loss_name, var_name))
        return torch.zeros(*dims, 3, H, W, device=var.device), 0

    return utils.get_sign_image(var.grad.detach())
