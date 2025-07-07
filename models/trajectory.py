from functools import partial

import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import get_nl
from .planar import PlanarMotion

import sys

sys.path.append("..")
import utils

def init_trajectory(dset, n_layers, local=False, **kwargs): #trajectory这个词的意思是轨迹
    N, H, W = len(dset), dset.height, dset.width
    if local: #这里一直是True,
        return BSplineTrajectory(N, n_layers, (H, W), **kwargs) #B样条参数
    # 平面运动是父类、样条运动是子类
    else:
        print("!!!这与我的认知相矛盾，local不应该为false : ./models/trajectory.py init_trajectory()!!!")
        exit(0)
    return PlanarTrajectory(N, n_layers, **kwargs) #这是啥?欧拉场？

def estimate_displacements( #评估每一图层在每一帧的整体变化
    flow_dset, masks, init_scale=0.8, thresh=0.5, pad=0.1, fg_pad=0.2, bg_scale=1
):
    #因为进行了掩码二值化，所以基本没有梯度计算
    """
    roughly estimate how much the scene is displaced per frame
    粗略估计每帧场景的位移量
    :params masks (N,M,1,H,W)
    :returns trans (N,M,2), scale (N,M,2), uv_range (M,2)
    -----------------------------------------------------------------------
    这段代码定义了一个函数estimate_displacements，用于估计视频序列中每一帧、每一层（包括前景物体和背景）的位移和缩放变换。
    以下是关键部分的解释：
    输入参数
        flow_dset：光流数据集，包含N帧的光流信息（每帧光流形状为[2, H, W]）
        masks：分割掩码张量，形状为(N, M, 1, H, W)，表示N帧、M个层（含背景）
        init_scale：初始缩放因子（默认0.8）
        thresh：掩码二值化阈值（默认0.5）
        pad：边界填充比例（默认0.1）
        fg_pad：前景边界框扩展比例（默认0.2）
        bg_scale：背景层缩放范围（默认1）
    输出
        trans：平移向量，形状(N, M, 2)
        scale：缩放因子，形状(N, M, 2)
        uv_range：归一化坐标范围，形状(M, 2)
        ok：布尔张量，指示前景层是否有效
    -----------------------------------------------------------------------
    """
    assert len(flow_dset) == len(masks)
    device = masks.device
    N, M = masks.shape[:2]
    thresh = 0.5 
    
    # 1.掩码二值化
    with torch.no_grad():
        # print("-------------------------------------")
        # print("masks",masks.shape)
        # print("masks[:, :, 0]",masks[:, :, 0].shape)
        sel = binarize_masks(masks[:, :, 0], thresh)  # (N, M, H, W) #转换为二值MASK之前去除了一个空白维度
        # print("sel",sel.shape)
        imageio.mimwrite(f"init_masks.gif", (255 * sel[:, 0].cpu()).byte()) #保存前景层的视频
        # print("sel[:, 0]",sel[:, 0].shape)
        # exit(0)
        # 将原始掩码阈值化为0/1，并保存第一层的掩码动画（调试用）。

    # 2.计算平均光流
    # get the mean flows for each layer for all frames 获取所有帧的每一层的平均流量
    # flow element is tuple (valid, flow)
    flows = [flow_dset[i][1].to(device) for i in range(N)]
    flow_vecs = [
        flows[i][:, sel[i, j]] for i in range(N) for j in range(M)
    ]  # B*M list (2, -1)
    med_flow = reduce_tensors(flow_vecs, torch.mean, dim=-1).reshape(N, M, 2)
    med_flow = torch.where(med_flow.isnan(), torch.zeros_like(med_flow), med_flow) #将空的位置替换为0
    '''
        提取每帧光流数据
        对每帧的每个层，仅选择掩码覆盖区域的光流向量
        计算这些区域的光流均值，得到每帧每层的平均位移向量med_flow
    '''

    # 3.计算前景边界框
    # estimate the bboxes of each layer except background
    bb_min, bb_max, ok = compute_bboxes(sel[:, :-1])
    # print("3.计算前景边界框 ok:",ok)
    # print("中断位置在trajectory.py estimate_displacements()")
    # exit(0)
    '''
        为前景层（M-1层）计算边界框(bb_min, bb_max)
        ok标记哪些层有有效掩码（防止空掩码）
    '''

    # 4.处理背景层边界
    bb_min = torch.cat(
        [bb_min - fg_pad, -bg_scale * torch.ones(N, 1, 2, device=device)], dim=1
    )
    bb_max = torch.cat(
        [bb_max + fg_pad, bg_scale * torch.ones(N, 1, 2, device=device)], dim=1
    )
    '''
        前景层：扩展边界（±fg_pad）
        背景层：固定为[-bg_scale, bg_scale]
    '''

    # 5.计算变换参数
    '''
        调用compute_scale_trans函数，基于平均光流和边界框计算：
            trans：位移向量（根据med_flow调整）
            scale：缩放因子（考虑init_scale和边界框）
            uv_range：归一化坐标范围（如[-1,1]）
    '''
    return (*compute_scale_trans(med_flow, bb_min, bb_max, init_scale, pad), ok)

def compute_bboxes(sel, default=0.5):
    """
    :param sel (N, M, H, W)
    :returns bb_min (N, M, 2), bb_max (N, M, 2), ok (N, M)
    --------------------------------------------------------
    对于每个batch中的每个层，检测是否有被选中的像素
    对于有选中像素的位置，计算这些像素的边界框
    对于空位置，返回覆盖整个图像的默认边界框
    特别处理全空的情况（避免对空张量进行计算）
    """
    N, M, H, W = sel.shape #10, 1, 128, 128
    # print("sel.shape",sel.shape)
    # 1.获取UV坐标网格:
    uv = utils.get_uv_grid(H, W, device=sel.device).permute(2, 0, 1)  # (2, H, W)
    # print("uv.shape", uv.shape)
    '''
        生成表示像素位置的归一化坐标网格
        假设坐标范围是[-0.5, 0.5]（左下角(-0.5,-0.5)到右上角(0.5,0.5))
        结果形状为 (2, H, W)，其中第一维是(u,v)坐标
        --------------------------------------------------------
        这个uv对象有啥用?
    '''

    # 2.初始化边界框:
    bb_min = -0.5 * torch.ones(N, M, 2, device=sel.device)
    bb_max = 0.5 * torch.ones(N, M, 2, device=sel.device)
    '''
        创建默认边界框，覆盖整个图像范围[-0.5, 0.5]
        形状为 (N, M, 2)，最后一维存储(x,y)坐标
    '''

    # 3.检测非空掩码:
    # manually fill in the non-empty slots (min is undefined for empty)
    ok = (
        sel.sum(dim=(-1, -2)) > 0 # 每张图片都判断一下是否为全0图片
    )  # (N, M) which layers of which frames have non-empty bboxes
    # sel.sum(dim=(-1, -2)).shape = [10, 1]
    # ok.shape = [10, 1]
    # print("ok", ok)
    ii, jj = torch.where(ok)  # each (T ~ N*M) #ii是层号，jj是帧编号
    # ii.shape = [9]
    # print("ii",ii)
    # print("jj",jj)
    '''
        计算每个(N,M)位置的像素和，判断是否有有效内容
        获取有内容的(batch, layer)索引对
        --------------------------------------
        ok=[[True],[True],[False],[False],[True],[True],[True],[True],[True],[True]]
        ii=[0, 1, 4, 5, 6, 7, 8, 9] #层编号
        jj=[0, 0, 0, 0, 0, 0, 0, 0] #帧编号
    '''

    # 4.提取有效位置的UV坐标:
    # print("sel",sel.shape)
    # print("uv",uv.shape,type(uv))
    # print("ii",ii)
    # print("jj",jj)
    uv_vecs = [uv[:, sel[i, j]] for i, j in zip(ii, jj)] #每一帧返回一组有效点坐标
    '''
        zip 是 Python 内置的函数。
        zip(ii, jj) 将 ii 和 jj 的元素一一配对，生成一个迭代器。
        zip(ii, jj)=[(0, 0), (1, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0)]
        -----------------------
        对每个有内容的位置，收集被选择像素的UV坐标 
        结果是一个列表，每个元素是形状为 (2, P) 的张量（P是被选像素数）
        !!!每个前景图层帧的被选像素数P都是独立的!!!
        -----------------------
        返回有效前景图层帧的物体包围框
        uv_vecs表示一组包围框
    '''

    # 5.计算实际边界框:
    if len(uv_vecs) > 0:
        # print("1-uv_vecs", len(uv_vecs))
        # print("1-uv_vecs[0]", uv_vecs[0], uv_vecs[0].shape)
        # print("1-uv_vecs[1]", uv_vecs[1], uv_vecs[1].shape)
        bb_min[ii, jj] = reduce_tensors(uv_vecs, torch.amin, dim=-1)  # (T, 2) #有效点坐标的最小值
        bb_max[ii, jj] = reduce_tensors(uv_vecs, torch.amax, dim=-1)  # (T, 2) #有效点坐标的最大值
        # print("uv_vecs:",uv_vecs)
        # print("2-uv_vecs",len(uv_vecs))
        # print("2-uv_vecs[0]",uv_vecs[0],uv_vecs[0].shape)
        # print("2-uv_vecs[1]", uv_vecs[1], uv_vecs[1].shape)
        # exit(0)
        '''
          type(uv_vecs) = <class 'list'>   len(uv_vecs)=8
          uv_vecs[0]: 
            tensor([
                [-0.8359, -0.8203],
                [-0.5391, -0.5391]
            ], device='cuda:0')
        '''
    '''
        计算所选像素的x和y最小值 → 边界框左下角
        计算所选像素的x和y最大值 → 边界框右上角
    '''
    return bb_min, bb_max, len(uv_vecs) > 0

def compute_scale_trans(med_flow, bb_min, bb_max, init_scale=0.9, pad=0.2):
    """
    :param med_flow (N, M, 2)
    :param bb_min (N, M, 2)
    :param bb_max (N, M, 2)
    :returns trans (N, M, 2), scale (N, M, 2), uv_range (M, 2)
    """
    N, M, _ = med_flow.shape
    disp = -torch.cumsum(med_flow, dim=0)  # (N, M, 2)
    disp = torch.cat([torch.zeros_like(disp[:1]), disp[:-1]], dim=0)

    # align bboxes using estimated displacement from first frame
    align_min = bb_min + disp
    align_max = bb_max + disp
    min_coord = torch.quantile(align_min, 0.2, dim=0, keepdim=True) - pad  # (1, M, 2)
    max_coord = torch.quantile(align_max, 0.8, dim=0, keepdim=True) + pad  # (1, M, 2)
    uv_range = torch.abs(max_coord - min_coord)  # (1, M, 2)

    scale = init_scale * 2 / uv_range.repeat(N, 1, 1)  # (N, M, 2)
    trans = scale * (disp - min_coord) - init_scale  # (N, M, 2)

    return trans, scale, uv_range

def binarize_masks(masks, thresh=0.5):
    """
    :param masks (B, M, H, W)
    :return sel (B, M, H, W)
    """
    # thresh =0.4
    sel = (masks > thresh).float()
    print("in",sel.shape)
    sel = F.max_pool2d(1 - sel, 3, 1, 1)
    sel = 1 - F.max_pool2d(sel, 3, 1, 1)
    print("out",sel.shape)
    return sel.bool()

def reduce_tensors(tensor_list, fnc, dim=-1):
    """
    reduce tensors in a list with fnc along dim, then stack
    :param tensor_list (list of N (*, d) tensors)
    :returns (N, *) tensor
    -------------------------------------------------------------
    ok=[[False],[False],[False],[False],[False],[ True],[ True],[False],[ True],[ True]]
    ii=[5, 6, 8, 9]
    jj=[0, 0, 0, 0]
    uv_vecs=[  ]
    reduce_tensors(uv_vecs, torch.amin, dim=-1)
    -------------------------------------------------------------
    uv_vecs[ torch.Size([2, n],... ]

    """
    if len(tensor_list) < 1:
        return torch.tensor([])
    # print("tensor_list[0]",tensor_list[0].shape)
    # print("fnc",fnc)
    # print("torch.stack([fnc(t, dim=dim) for t in tensor_list], dim=0)",
    #       torch.stack([fnc(t, dim=dim) for t in tensor_list], dim=0).shape)
    # exit(0)
    return torch.stack([fnc(t, dim=dim) for t in tensor_list], dim=0)
    # 在末尾维度上取最值: 6*[2] <= 6*[2,n]
    # 在头部维度上添加新的维度：[6,2] <= 6*[2]

class PlanarTrajectory(PlanarMotion):
    """
        Planar motion model that interpolates motion parameters between frames 在帧之间插值运动参数的平面运动模型
        (enforces temporal smoothness)（强制时间平滑）
        该类用于在时间维度上对平面运动参数（如缩放和位移）进行插值，以实现平滑的运动轨迹
    """
    def __init__(
        self, n_total, n_layers, t_step=1, degree=2, scale=None, trans=None, **kwargs
    ):
        '''
        Args:
            n_total：总帧数。                         #10
            n_layers：运动层数（如多个运动对象）         #1
            t_step：关键帧的采样步长（默认每帧都是关键帧） #2 (每两帧取一个关键帧)
            degree：B样条插值的阶数（默认2，二次插值）    #2
            scale：初始缩放参数（形状 (N, M, 2)）       #None
            trans：初始位移参数（形状 (N, M, 2)）       #None
            **kwargs：其他传递给父类的参数。            #{'init_bg': True, 'fg_scale': 2, 'lr': 0.001}
        '''
        nk_t = (n_total - 1) // t_step + 1 # 关键帧数量
        tk_range = int(t_step * nk_t) # 关键帧覆盖的时间范围

        self.n_total = n_total
        self.t_step = t_step
        self.nk_t = nk_t
        self.tk_range = tk_range
        self.degree = degree

        # 没有进行位移和缩放
        if trans is not None:
            trans = trans[::t_step] # 按t_step下采样位移
        if scale is not None:
            scale = scale[::t_step] # 按t_step下采样缩放

        ## keep an explicit motion field for the knots #保持一个明确的运动场
        super().__init__(nk_t, n_layers, scale=scale, trans=trans, **kwargs)
        # 父类 PlanarMotion 负责存储关键帧的运动参数（self.theta）

    def get_theta(self, idx):
        """
        :param idx (B)
        目标帧索引（形状为(B,)的张量）
        """
        if self.nk_t == self.tk_range:
            #当关键帧覆盖全部帧时（t_step=1），直接返回父类结果。
            return super().get_theta(idx)

        # 1.准备关键帧数据：
        N, M, *dims = self.theta.shape # N=关键帧数, M=层数
        knots = self.theta.transpose(0, 1).view(M, N, -1)  # (M, nk_t, -1) ## 形状变为 (M, N, -1)
        # knots[0] <-- 0, knots[n] <-- N - 1

        # 2.计算插值位置：
        idx = idx / (self.tk_range - 1) * (self.nk_t - 1) - 0.5
        positions = idx.view(1, -1).expand(M, -1)  # (M, B)

        # 3.执行B样条插值：
        sel_theta = utils.bspline.interpolate_1d(
            knots, positions, self.degree
        )  # (M, B, -1)

        # 4.调整输出形状：
        sel_theta = sel_theta.transpose(0, 1).view(-1, M, *dims)
        return sel_theta

    def update_scale(self, scale):
        """
        :param scale (N, M, 2)
        功能：更新缩放/位移参数。若输入是完整序列（长度 n_total），则先下采样到关键帧再调用父类方法。
        目的：确保父类只存储关键帧数据，非关键帧通过插值获得。
        """
        if len(scale) == self.n_total:
            scale = scale[:: self.t_step] # 下采样到关键帧
        super().update_scale(scale)

    def update_trans(self, trans):
        """
        :param trans (N, M, 2)
        """
        if len(trans) == self.n_total:
            trans = trans[:: self.t_step] # 下采样到关键帧
        super().update_trans(trans)

def FG_init_trajectory(dset, n_layers, local=False, **kwargs):
    N, H, W = len(dset), dset.height, dset.width
    if local:
        return FG_BSplineTrajectory(N, n_layers, (H, W), **kwargs)
    return PlanarTrajectory(N, n_layers, **kwargs)

class FG_BSplineTrajectory(PlanarTrajectory):
    """
    Planar motion + 2D splines
    """

    def __init__(
        self,
        n_total,
        n_layers,
        out_shape,
        t_step=1,
        xy_step=8,
        final_nl="tanh",
        active_local=True,
        bg_local=True,
        max_step=0.1,
        **kwargs
    ):
        super().__init__(n_total, n_layers, t_step=t_step, **kwargs)

        H, W = out_shape
        nk_x, nk_y = W // xy_step, H // xy_step

        self.active_local = active_local
        self.bg_local = bg_local
        self.nk_t = 1
        nk_layers = n_layers if bg_local else n_layers - 1
        knots = torch.zeros(nk_layers, self.nk_t, nk_y, nk_x, 2)
        self.register_parameter("knots_3d", nn.Parameter(knots, requires_grad=True))
        print(
            "Initialized BSpline motion with {} knots".format((self.nk_t, nk_y, nk_x))
        )
        print("knots_3d.shape:", knots.shape)

        self.final_nl = get_nl(final_nl)
        max_step = torch.cat(
            [torch.ones(n_layers - 1) * max_step, torch.ones(1) * 0.5 * max_step]
        )
        self.register_buffer("max_step", max_step.view(1, n_layers, 1, 1, 1))

    def get_rigid_transform(self, idx, grid):
        return super().forward(idx, grid)

    def init_local_field(self):
        self.active_local = True
        print("MOTION FIELD NOW LOCAL")

    def get_knots(self):
        knots = self.knots_3d.transpose(0, 1)  # (N, M, h, w, 2)
        return knots.permute(0, 1, 4, 2, 3)  # (N, M, 2, h, w)

    def get_knots_xy(self, idx):
        M, nk_t, nk_y, nk_x, D = self.knots_3d.shape
        if nk_t == self.tk_range:
            knots_xy = self.knots_3d[:, idx]  # (M, B, nk_y, nk_x, D)
            return knots_xy.view(-1, nk_y, nk_x, D)

        ## interpolate in time, independently in every dimension
        B = idx.shape[0]
        knots_t = self.knots_3d.view(M, nk_t, -1)  # (M, nk_t, nk_y*nk_x*D)
        # need to rescale the query points in terms of number of knots
        idx = idx / (self.tk_range - 1) * (nk_t - 1) - 0.5
        positions_t = idx.view(1, -1).expand(M, -1)  # (M, B)

        ## query the 2d control points in time
        knots_xy = utils.bspline.interpolate_1d(
            knots_t, positions_t, 0
        )  # (M, B, -1)
        knots_xy = knots_xy.view(-1, nk_y, nk_x, D)  # (M*B, nk_y, nk_x, 2)
        return knots_xy

    def get_local_offsets(self, idx, grid):
        """
        :param idx (B)
        :param grid (B, M, H, W, 3)
        """
        M, nk_t, nk_y, nk_x, D = self.knots_3d.shape
        B = idx.shape[0]
        H, W, _ = grid.shape[-3:]

        knots_xy = self.get_knots_xy(idx)

        ## set up the query grid
        grid = grid[:, :M, ..., :2]  # (B, M, H, W, 2)
        grid = grid.transpose(0, 1).reshape(-1, H, W, 2)

        # rescale x and y from [-1, 1] --> [-0.5, nk - 1.5]
        fac = torch.tensor([(nk_x - 1) / 2, (nk_y - 1) / 2], device=grid.device).view(
            1, 1, 1, 2
        )
        query_grid = (grid + 1) * fac - 0.5
        offsets = utils.bspline.interpolate_2d(
            knots_xy, query_grid, self.degree
        )  # (M*B, H, W, 2)
        offsets = offsets.view(M, B, H, W, 2).transpose(0, 1)  # (B, M, H, W, 2)

        if not self.bg_local:
            # we don't keep 2D spline for background, add zeros
            print("bg_local",self.bg_local)
            offsets = torch.cat([offsets, torch.zeros_like(offsets[:, :1])], dim=1)

        offsets = self.final_nl(offsets) * self.max_step
        return offsets

    def forward(self, idx, grid):
        t_rigid = self.get_rigid_transform(idx, grid)  # (B, M, H, W, 2)
        if self.active_local:
            t_local = self.get_local_offsets(idx, grid)  # (B, M, H, W, 2)
            #             print(t_local.square().sum(dim=-1).mean())
            return t_rigid + t_local
        return t_rigid

class BSplineTrajectory(PlanarTrajectory):
    """
    Planar motion + 2D splines
    平面运行+2D样条
    ----------------------------
    它结合了平面运动(Planar Motion)和2D-B样条变形场(Spline.Deformation.Field)，
    用于建模动态场景中的复杂运动。
    ----------------------------
    这个类实现了分层运动轨迹模型：
    全局刚体运动(父类 PlanarTrajectory 实现):
        包括平移、旋转、缩放等基础变换
    局部非刚性变形(本类实现的B样条变形):
        使用B样条控制点建模细节形变
        支持时间插值（帧间平滑运动）
    """

    def __init__(
        self,
        n_total,
        n_layers,
        out_shape,
        t_step=1,
        xy_step=8,
        final_nl="tanh",
        active_local=True,
        bg_local=True,
        max_step=0.1,
        **kwargs
    ):
        super().__init__(n_total, n_layers, t_step=t_step, **kwargs)

        # 1.B样条控制点网格：
        H, W = out_shape #图片分辨率:128*128
        nk_x, nk_y = W // xy_step, H // xy_step #关键点数量:32*32 #每个样条块的大小是4*4

        self.active_local = active_local #False
        self.bg_local = bg_local #True
        # self.nk_t = 1
        nk_layers = n_layers if bg_local else n_layers - 1 # nk_layers=1
        knots = torch.zeros(nk_layers, self.nk_t, nk_y, nk_x, 2)
        # [1, 5, 32, 32, 2] #[层数、时长、水平点数、竖直点数、坐标] #在时间中间也进行插值
        self.register_parameter("knots_3d", nn.Parameter(knots, requires_grad=True))
        print(
            "Initialized BSpline motion with {} knots".format((self.nk_t, nk_y, nk_x))
        )
        print("knots_3d.shape:", knots.shape)

        self.final_nl = get_nl(final_nl)
        # 2.变形幅度限制：
        max_step = torch.cat(
            [torch.ones(n_layers - 1) * max_step, torch.ones(1) * 0.5 * max_step] #n_layers=1, max_step=0.1
        ) #tensor([0.0500])
        self.register_buffer("max_step", max_step.view(1, n_layers, 1, 1, 1)) #tensor([[[[[0.0500]]]]])

    def get_rigid_transform(self, idx, grid):
        return super().forward(idx, grid)

    def init_local_field(self):
        self.active_local = True
        print("MOTION FIELD NOW LOCAL")

    def get_knots(self):
        knots = self.knots_3d.transpose(0, 1)  # (N, M, h, w, 2)
        return knots.permute(0, 1, 4, 2, 3)  # (N, M, 2, h, w)

    def get_knots_xy(self, idx): #时间维度插值 (get_knots_xy)：
        M, nk_t, nk_y, nk_x, D = self.knots_3d.shape
        if nk_t == self.tk_range:
            knots_xy = self.knots_3d[:, idx]  # (M, B, nk_y, nk_x, D)
            return knots_xy.view(-1, nk_y, nk_x, D)

        ## interpolate in time, independently in every dimension
        B = idx.shape[0]
        knots_t = self.knots_3d.view(M, nk_t, -1)  # (M, nk_t, nk_y*nk_x*D)
        # need to rescale the query points in terms of number of knots
        idx = idx / (self.tk_range - 1) * (nk_t - 1) - 0.5
        positions_t = idx.view(1, -1).expand(M, -1)  # (M, B)

        ## query the 2d control points in time
        knots_xy = utils.bspline.interpolate_1d(
            knots_t, positions_t, self.degree
        )  # (M, B, -1)
        knots_xy = knots_xy.view(-1, nk_y, nk_x, D)  # (M*B, nk_y, nk_x, 2)
        return knots_xy

    def get_local_offsets(self, idx, grid): #空间变形场计算 (get_local_offsets)：
        """
        :param idx (B)
        :param grid (B, M, H, W, 3)
        """
        M, nk_t, nk_y, nk_x, D = self.knots_3d.shape
        B = idx.shape[0]
        H, W, _ = grid.shape[-3:]

        knots_xy = self.get_knots_xy(idx)

        ## set up the query grid
        grid = grid[:, :M, ..., :2]  # (B, M, H, W, 2)
        grid = grid.transpose(0, 1).reshape(-1, H, W, 2)

        # rescale x and y from [-1, 1] --> [-0.5, nk - 1.5]
        fac = torch.tensor([(nk_x - 1) / 2, (nk_y - 1) / 2], device=grid.device).view(
            1, 1, 1, 2
        )
        query_grid = (grid + 1) * fac - 0.5
        offsets = utils.bspline.interpolate_2d(
            knots_xy, query_grid, self.degree
        )  # (M*B, H, W, 2)
        offsets = offsets.view(M, B, H, W, 2).transpose(0, 1)  # (B, M, H, W, 2)
        if not self.bg_local:
            # we don't keep 2D spline for background, add zeros
            print("bg_local",self.bg_local)
            offsets = torch.cat([offsets, torch.zeros_like(offsets[:, :1])], dim=1)

        offsets = self.final_nl(offsets) * self.max_step
        return offsets

    def forward(self, idx, grid): # 完整运动计算
        t_rigid = self.get_rigid_transform(idx, grid)  # (B, M, H, W, 2) #获取刚体变换
        if self.active_local:
            t_local = self.get_local_offsets(idx, grid)  # (B, M, H, W, 2) #获取软体变换
            #             print(t_local.square().sum(dim=-1).mean())
            return t_rigid + t_local # 返回刚体+软体
        return t_rigid
