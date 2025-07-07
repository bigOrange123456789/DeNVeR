import utils
import torch
import torch.nn as nn
import torch.nn.functional as F

from .alpha_pred import AlphaModel
from .tex_gen import TexUNet
from .trajectory import *
import sys

sys.path.append("..")


class SpriteModel(nn.Module):#这应该是那个三分支模型本身
    """
    Full sprite model 全精灵模型
    cfg loaded from src/confs/models
    """

    def __init__(self, dset, data_path, n_layers, cfg):
        super().__init__()

        self.dset = dset

        N, H, W = len(dset), dset.height, dset.width
        self.n_layers = n_layers

        # 一、initialize mask prediction 初始化掩码预测
        args = cfg.alpha_pred
        self.alpha_pred = AlphaModel(n_layers, **dict(args))#几何生成器
        optims = [{"params": self.alpha_pred.parameters(), "lr": args.lr}]

        self.has_tex = cfg.use_tex #使用纹理
        if cfg.use_tex:
            # 二、initialize texture generator 初始化纹理生成器
            args = cfg.tex_gen
            TH, TW = args.scale_fac * H, args.scale_fac * W
            self.tex_shape = (TH, TW)
            self.tex_gen = TexUNet(#纹理生成器
                n_layers, self.tex_shape, data_path=data_path, **dict(args))
            optims.append({"params": self.tex_gen.parameters(), "lr": args.lr})

            # 三、initialize motion model 初始化运动模型
            args = cfg.transform
            self.local = args.local
            self.active_local = False
            self.tforms = init_trajectory( #背景的运动参数
                dset,
                1,
                active_local=self.active_local,
                **dict(args),
            )
            optims.append({"params": self.tforms.parameters(), "lr": args.lr})
            self.fg_tforms = FG_init_trajectory( #这是血管的运动参数么？
                dset,
                1,
                active_local=self.active_local,
                **dict(args),
            ) #目前fg_tforms和tforms内部使用的方法是一样的
            optims.append({"params": self.fg_tforms.parameters(), "lr": args.lr})
            view_grid = utils.get_uv_grid(H, W, homo=True)  # (H, W, 3)
            self.register_buffer("view_grid", view_grid.view(1, 1, H, W, 3))

            cano_grid = utils.get_uv_grid(TH, TW, homo=True)  # (TH, TW, 3)
            self.register_buffer("cano_grid", cano_grid.view(1, 1, TH, TW, 3))

        self.optim = torch.optim.Adam(optims)
        self.skip_keys = ["alpha", "pred"]

    def forward(
        self,
        batch, # batch.keys()=[rgb, fwd, bck, epi, occ, disocc, ske, idx]
        quantile=0.8,
        ret_tex=True, #我猜是纹理
        ret_tform=True,#我猜是B样条变换
        vis=False,  #??不确定是不是画面分割
        ret_inputs=False,
    ):
        '''
            第一阶段：
                {ret_tex=ret_tform: False, vis=ret_inputs: False}
                {ret_tex=ret_tform: False, vis=ret_inputs: True} #验证
            其余阶段：
                {ret_tex=ret_tform: True, vis=ret_inputs: False}
                {ret_tex=ret_tform: True, vis=ret_inputs: True} #验证
            -----------------------------------------------------------
            in(batch)=[rgb原视频, fwd光流, bck逆向, epi黑塞分割, occ遮挡, disocc逆向遮挡, ske骨架, idx:帧索引?]
            out=[
                ###***  默认输出  ***###
                    masks: 分割结果(1个通道表示血管、另一个表示1-血管)
                    alpha: 全1数据(单通道)
                    pred: mask[0](单通道)
                ###*** ret_tform ***###
                    view_grid: UV
                    coords: 不可学习的固定值
                    fgcoords: 背景固定UV+前景相对UV
                ###*** ret_tex ***###
                    texs: 全局纹理
                    raw_apprs: 单帧纹理
                    apprs: 加噪单帧纹理
            ]
        '''
        out = {}

        # 在第一阶段MASK训练过程中，只有这里起了效果
        # print("batch[rgb].shape",batch["rgb"].shape)
        alpha_dict = self.alpha_pred(batch["rgb"]) #keys()=[masks, alpha, pred]
        '''
            图片数量有时候是10有时候是9,非常玄学
            -------------------------------------------
            输入为10张3通道的彩色图片  [10, 3, 128, 128]
            输出为10张2通道的图片     [10, 2, 1, 128, 128] 
                因为通道数量为2所以感觉应该不是单纯的MASK
        '''
        # 不知道alpha_dict输出的是什么
        # print("out1:",out.keys())
        out.update(alpha_dict) # 将alpha_dict中的属性全部添加到out中
        # print("out2:", out.keys())
        # exit(0)

        masks = out["masks"] # [9, 2, 1, 128, 128] #应该是10张图片的MASK
        # print(batch["rgb"].shape,masks.shape)
        # print("masks.shape", masks.shape)
        # print("masks.shape",masks.shape,type(masks))
        # print(masks[0,:,:,:,:].shape)
        # show_images(masks[:, 0, :, :, :])
        # print("masks[:, 0, :, :, :]",masks[:, 0, :, :, :].shape)
        # save_tensor_as_images(masks[0, :, :, :, :], "test_out")
        # save_tensor_as_images(masks[0, :, :, :, :], "test_out")
        # exit(0)
        B, M, _, H, W = masks.shape #这个M表达了什么含义?

        ret_tex = ret_tex and self.has_tex
        ret_tform = (ret_tex or ret_tform) and self.has_tex #

        if ret_tform:#True #UV计算
            # get the coordinates from view to canonical #获取从视图到规范的坐标
            tform_dict = self.get_view2cano_coords(batch["idx"])
            # print("batch[idx]",batch["idx"])
            # print("tform_dict",tform_dict.keys())
            # exit(0)
            out.update(tform_dict) #[view_grid, coords, fgcoords] #out:[UV,不可学习的固定值,背景固定UV+前景相对UV]

        if ret_tex:#True #纹理计算
            # get the canonical textures and warped appearances #获得规范纹理和扭曲外观
            # texs (M, 3, H, W) and apprs (B, M, 3, H, W)
            # print("out[coords]",out["coords"])
            # print("vis",vis)
            # print("self.tex_gen",self.tex_gen)
            # exit(0)
            tex_dict = self.tex_gen(out["coords"], vis=vis)
            out.update(tex_dict) # [texs, raw_apprs, apprs] #in:UV out:[全局纹理,加噪单帧纹理,单帧纹理]

            # composite layers
            out["recons"] = (masks * out["apprs"]).sum(dim=1)  # (B, 3, H, W)
            out["layers"] = utils.composite_rgba_checkers(masks, out["apprs"])

        if vis:#False #验证测试时启用
            out["masks_vis"] = utils.composite_rgba_checkers(masks, 1) # masks_vis和masks是完全相同的
            if ret_tform: # 如果请求变换可视化
                # 1. 规范空间可视化 cano_vis
                # 生成规范空间可视化
                out["cano_vis"] = self.get_cano2view_vis(
                    batch["idx"], masks, out.get("texs", None)
                )
                '''
                功能：生成规范空间(canonical space)的可视化
                计算过程：
                    使用当前帧索引 batch["idx"] 获取变换参数
                    结合掩码 masks 和纹理 texs（如果可用）
                    将规范空间内容转换到视图空间进行可视化
                可视化内容：
                    展示规范空间中的纹理如何映射到当前视图
                    显示规范空间的结构布局
                    可视化变换参数的效果
                '''
                # 生成视图空间可视化
                out["view_vis"] = self.get_view2cano_vis(out["coords"], masks)
                '''
                功能：生成视图空间(view space)的可视化
                计算过程：
                    使用预先计算好的规范坐标 out["coords"]
                    结合掩码 masks
                    直接在视图空间渲染结果
                可视化内容：
                    显示视图空间中的规范坐标分布
                    可视化不同层的坐标映射关系
                    展示视图空间的最终渲染效果
                '''

        if ret_inputs:#False #验证测试时启用
            # exit(0)
            out["rgb"] = batch["rgb"]
            out["idx"] = batch["idx"]
            out["flow"] = utils.flow_to_image(batch["fwd"][1])
            out["flow_groups"] = utils.composite_rgba_checkers(
                masks, out["flow"][:, None]
            )
        # exit(0)
        # print("out.keys():",out.keys())

        return out

    def get_view2cano_coords(self, idx):
        '''
            目的：计算从视图空间到规范空间的坐标映射
            分层处理：分别处理前景(第0层)和背景(第1层)
            关键关系：前景运动是相对于背景的附加运动
        '''
        # 1.准备基础网格
        B, M = len(idx), self.n_layers #获取批次大小(B,10帧图片)和层数(M,2层对象)
        view_grid = self.view_grid.expand(B, M, -1, -1, -1)  # (B, M, H, W, 3) #扩展基础视图网格到当前批次大小和层数
        # [1, 1, 128, 128, 3] -> [10, 2, 128, 128, 3]
        '''
            self.view_grid:
                存储的是图像上每个像素的坐标，通常以齐次坐标形式表示（x, y, 1）。
                坐标范围通常是归一化的，比如在[-1, 1]之间（PyTorch中常见的网格坐标范围）。
            获取批次大小(B)和层数(M)
            扩展基础视图网格到当前批次大小和层数
            形状：(B, M, H, W, 3)，其中最后一个维度是齐次坐标(x, y, 1)
        '''

        # 2.处理背景层
        BG_view_grid = view_grid[:,1,:,:,:].unsqueeze(1) # 提取背景层(索引1)的视图网格 (B,1,H,W,3) #[10, 128**2, 3]
        bg_coords = self.tforms(idx, BG_view_grid)       # 应用背景变换模型(tforms)得到背景规范坐标 (B,1,H,W,2) #[10,1,128**2,2]
        # [ in:本身的坐标; out:对应到全景纹理上的坐标 ]

        # 3.准备前景层输入
        FG_view_grid = view_grid[:,0,:,:,:].unsqueeze(1)#[10, 128, 128, 3] #FG_view_grid不可学习,是固定值
        F_coords = torch.cat([bg_coords, torch.zeros_like(bg_coords[..., -1:])], dim=-1)#背景运动
        # [10, 1, 128, 128, 2+1] #在背景层的每个UV坐标后补了一个0
        FG_view_grid = FG_view_grid + F_coords          # 叠加背景变换 #模拟前景附着在背景上的运动
        FG_view_grid = torch.clamp(FG_view_grid,-1,1)   # 限制在[-1,1]范围内 #确保坐标在有效范围内

        # 4.处理前景层
        fg_coords = self.fg_tforms(idx, FG_view_grid) #前景坐标
        tmp = torch.cat((fg_coords, bg_coords), dim=1) #前景相对纹理坐标,背景绝对纹理坐标

        # 5.合成最终坐标
        fg_coords = fg_coords+ bg_coords        # 前景+背景
        # 前景最终位置 = 背景位置 + 前景相对位置
        fg_coords = torch.clamp(fg_coords,-1,1) # 再次限制范围
        result_coords = torch.cat((fg_coords, bg_coords), dim=1) # (B,2,H,W,2)

        return {
            "coords": result_coords,# 最终的UV坐标
            "view_grid": view_grid, # 原始视图网格(不可学习的固定值)
            "fgcoords": tmp         # 前景相对纹理坐标、背景绝对纹理坐标
        }  # (B, M, H, W, 2)

    def get_view2cano_vis(self, coords, masks=None, nrows=16):
        """
        :param coords (B, M, H, W, 2)
        :param masks (optional) (B, M, 1, H, W)
        :param nrows (optional) (int)
        """
        B, M = coords.shape[:2]
        TH, TW = self.tex_shape
        device = coords.device
        cano_grid = utils.get_rainbow_checkerboard(
            TH, TW, nrows, device=device
        )  # (3, H, W)
        cano_grid = cano_grid[None, None].repeat(B, M, 1, 1, 1)
        view_grid = utils.resample_batch(
            cano_grid, coords, align_corners=False)
        if masks is None:
            masks = (view_grid != 0).float()

        view_grid = utils.composite_rgba_checkers(masks, view_grid)
        return view_grid
    def get_view2cano_vis_bg(self, coords, masks=None, nrows=16):
        """
        :param coords (B, M, H, W, 2)
        :param masks (optional) (B, M, 1, H, W)
        :param nrows (optional) (int)
        """
        B, M = coords.shape[:2]
        TH, TW = self.tex_shape
        device = coords.device
        cano_grid = utils.get_rainbow_checkerboard(
            TH, TW, nrows, device=device
        )  # (3, H, W)
        cano_grid = cano_grid[None, None].repeat(B, M, 1, 1, 1)
        view_grid = utils.resample_batch(
            cano_grid, coords, align_corners=False)
        if masks is None:
            masks = (view_grid != 0).float()
        view_grid = utils.composite_rgba_checkers(masks, view_grid)
        return view_grid
    def get_cano2view_vis(self, idx, masks, texs=None, fac=0.3, nrows=16):
        """
        :param idx (B)               #帧索引： 有效帧编号
        :param masks (B, M, 1, H, W) #层掩码： 分割图
        :param texs (1, M, 3, H, W)  #纹理图： texs.shape=[1, 2, 3, 256, 256]
        """
        # 1.准备规范空间网格
        B, M, _, H, W = masks.shape # 帧数:10 通道数:2 分辨率:128*128
        cano_grid = self.cano_grid.expand( #self.cano_grid是一组固定数值，坐标范围是:(-1,+1] #实际对应全局纹理图,尺寸比单帧图像更大
            B, M, -1, -1, -1)  # (B, M, TH, TW, 3) # cano_grid.shape=[10, 2, 256, 256, 3]
        '''
            获取批次大小(B)、层数(M)、图像尺寸(H,W)
            扩展规范空间网格到当前批次和层数
            cano_grid：规范空间的齐次坐标网格，形状 (B, M, TH, TW, 3)
        '''
        # 2.获取变换矩阵 #这些是每张图片整体的“变换矩阵”
        bg_cano2view = self.tforms.get_cano2view(idx)  # (B, M, 3, 3)
        fg_cano2view = self.fg_tforms.get_cano2view(idx)
        fg_cano2view = fg_cano2view + bg_cano2view #前景变换=原前景变换+背景变换
        cano2view = torch.cat((fg_cano2view, bg_cano2view), dim=1) #[前景,背景]
        '''
        bg_cano2view [10, 1, 3, 3]
        fg_cano2view [10, 1, 3, 3]
        cano2view    [10, 2, 3, 3]
        -------------------------------------------------------
            背景变换：bg_cano2view 获取背景的逆变换矩阵
            前景变换：fg_cano2view 获取前景的逆变换矩阵
            关键操作：fg_cano2view = fg_cano2view + bg_cano2view
                模拟前景附着在背景上的变换关系
                前景变换 = 前景相对变换 + 背景全局变换
            合并变换：将前景和背景变换拼接在一起
        '''
        # 3. 应用变换到视图空间
        view_coords = utils.apply_homography_xy1(
            cano2view, cano_grid # 应用变换后的网格坐标 <- 变换矩阵、网格坐标
        )  # (B,M,H,W,2)
        # [10, 2, 256, 256, 2] <= f([10, 2, 3, 3],[10, 2, 256, 256, 3])
        '''
            使用齐次变换将规范空间网格映射到视图空间
            apply_homography_xy1：应用3x3变换矩阵到齐次坐标
            输出 view_coords：视图空间坐标，形状 (B, M, TH, TW, 2)
        '''
        # 4. 创建棋盘格背景    # get_rainbow_checkerboard 获取_彩虹_棋盘
        view = utils.get_rainbow_checkerboard( #不知道生成这个网格的逻辑是什么
            H, W, nrows, device=idx.device
        )  # (3, H, W) # [3, 128, 128] # 为啥是3通道的图片
        view = view[None, None] * masks  # (1, 1, 3, H, W) * (B, M, 1, H, W)
        # [10, 2, 3, 128**2] <= [1, 1, 3, 128**2] * [10, 2, 1, 128**2]
        # exit(0)
        '''
            生成彩虹棋盘格背景，形状 (3, H, W)
            扩展维度并乘以掩码：(B, M, 3, H, W)
            结果：每层有独立的棋盘格可视化
        '''
        # 5. 网格采样到规范空间
        cano_frames = torch.stack(
            [
                F.grid_sample(view[:, i], # 当前层的棋盘格图像
                              view_coords[:, i], # 当前层的视图坐标
                              align_corners=False) #逐帧纹理[10,3,256,256]=g(全局纹理[10,3,128,128],UV[10,256,256,2])
                for i in range(M) # 遍历每一个对象层
            ],
            dim=1,
        ) # (B,M,3,H,W) # [10,2,3,256,256] <= 2*[10,3,256,256]
        # print(cano_frames.shape)
        '''
            ------------------------------------------------------------------------
            假设 view 的形状是(N,C,H,W)，view_coords的形状是(N,H,W,2)，那么这句代码的目的是：
                从 view 的第0个通道中，根据 view_coords 的坐标信息进行采样。
                使用双线性插值（或其他插值方式，取决于默认设置）来获取采样点的值。
                align_corners=False 确保采样点均匀分布在输入张量的整个空间范围内。
        '''
        # [10, 3, 256, 256]
        '''
            ----------------------------------------------------  
            view_coords[:, i]: 
                [10,256,256,2]
            cano_frames: 
                [10,2,3,256,256]
            ----------------------------------------------------- 
            对每层进行网格采样：
                view[:, i]：当前层的棋盘格图像
                view_coords[:, i]：当前层的视图坐标
            F.grid_sample：使用双线性插值采样
            输出 cano_frames：规范空间中的可视化，形状 (B, M, 3, TH, TW)
        '''
        # 6. 与纹理混合
        if texs is None:
            return cano_frames

        #cano_frames: [10, 2, 3, 256, 256]
        #texs:        [ 1, 2, 3, 256, 256]
        return fac * cano_frames + (1 - fac) * texs #动态视频与静态背景混合？

    def init_planar_motion(self, masks):
        '''
            目的：基于给定的掩码(masks)初始化前景和背景的运动参数
            输入：masks - 层掩码张量，形状通常为 (N, M, H, W) #应该不进行梯度回传
            输出：布尔值，表示初始化是否成功
        '''
        # 1. 检查纹理标志
        if self.has_tex: #使用纹理
            # 2. 获取光流数据
            fwd_set = self.dset.get_set("fwd")#正向光流
            '''
              输入数据：
                rgb: #原视频    #custom_videos/PNGImages 
                fwd: #正向光流  #custom_videos/raw_flows_gap1  
                bck: #逆向光流  #custom_videos/raw_flows_gap-1 
                epi: #黑塞掩码  #preprocess/--/binary  
                occ: #正向遮挡
                disocc: #逆向遮挡
                ske: #骨架     #custom_videos/skeltoize 
            '''
            # 3. 估计位移参数
            trans, scale, uv_range, ok = estimate_displacements(fwd_set, masks)
            '''
                trans：平移向量，形状(N, M, 2) #描述每一帧每一层的总位移
                scale：缩放因子，形状(N, M, 2) #描述每一帧每一层的总放缩
                uv_range：归一化坐标范围，形状(M, 2)
                ok：布尔张量，指示前景层是否有效
                ---------------------------------
                trans:    [10, 2, 2]
                scale:    [10, 2, 2]
                uv_range: [ 1, 2, 2]
            '''
            # print("ok",ok)
            # print("中断位置--sprite.py--SpriteModel(nn.Module)--init_planar_motion()--line406")
            # exit(0)
            # 4. 分离前景/背景参数
            fg_scale = scale[:,0,:].unsqueeze(1)
            fg_trans = trans[:,0,:].unsqueeze(1)
            bg_scale = scale[:,1,:].unsqueeze(1)
            bg_trans = trans[:,1,:].unsqueeze(1)
            # 5. 更新运动模型
            self.tforms.update_trans(bg_trans)
            self.tforms.update_scale(bg_scale)
            self.fg_tforms.update_trans(fg_trans)
            self.fg_tforms.update_scale(fg_scale)
            return ok #这个ok在什么情况下为false？
        return False

    def init_local_motion(self): #初始化前景和背景的局部场
        # print("self.has_tex and self.local",self.has_tex , self.local)
        # print("self.has_tex and self.local",self.has_tex and self.local)
        # exit(0) # true
        if self.has_tex and self.local: #这里为true会执行下面的代码 #但是没有使用B样条?
            self.active_local = True
            self.tforms.init_local_field()
            self.fg_tforms.init_local_field()
            # print("中断位置 -- sprite.py init_local_motion")
            # exit(0)
