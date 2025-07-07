from functools import partial
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import cv2
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from skimage import color, data, filters, graph, measure, morphology
import glob
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import data
import models
import utils
from loss import *
from helper import *

DEVICE = torch.device("cuda")

device_id = torch.cuda.current_device()

print("Current GPU Device ID:", device_id)

@hydra.main(config_path="confs", config_name="config")
def main(cfg: DictConfig): #现在最重要的是搞清楚这个三分支架构的三个分支都在哪里
    #  会加载 confs 文件夹中的 config.yaml 文件作为默认配置。
    # 我都没有看到这段代码的训练过程是在哪里执行的
    # 一个项目里面最重要的有三部分：数据集的加载、模型的推理、损失函数
    '''
    cfg={
        data:
        root: /liuzhicheng2/DNVR/custom_videos
        type: custom
        res: 512p
        seq: CVAI-2828RAO2_CRA3test5
        scale: 1
        flow_gap: 1
        model:
        transform:
            init_bg: true
            fg_scale: 2
            t_step: 2
            local: true
            bg_local: true
            lr: 0.001
            xy_step: 4
            max_step: 0.1
            final_nl: tanh
        alpha_pred:
            net_args:
            n_levels: 2
            d_hidden: 24
            fac: 2
            norm_fn: batch
            init_std: 0.1
            lr: 0.0005
            gamma: 0.99
        use_tex: true
        tex_gen:
            n_channels: 3
            n_levels: 4
            d_hidden: 16
            scale_fac: 2
            random_comp: true
            lr: 0.0005
        batch_size: 16
        vis_epochs: 2
        val_epochs: 4
        vis_every: 3000
        val_every: 6000
        vis_grad: true
        save_grid: false
        iters_per_phase:
        epi: 6000
        parallel: 6000
        kmeans: 12000
        planar: 10000
        deform: 18000
        refine: 1000
        epochs_per_phase:
        epi: 6
        parallel: 6
        kmeans: 12
        planar: 10
        deform: 9
        refine: 1
        w_epi: 0.5
        neg_ratio: 0.15
        w_kmeans: 0.05
        w_sparse: 0.001
        w_warp: 0.1
        w_tform: 1
        w_recon: 0.5
        l_recon: 1
        lap_ratio: 0.001
        lap_levels: 3
        w_contr: 0.0005
        n_layers: 2
        resume: true
        preload: true
        log_root: outputs
        exp_name: init_model
    }
    '''
    print(OmegaConf.to_yaml(cfg))
    dset = get_dataset(cfg.data)
    N, H, W = len(dset), dset.height, dset.width #N, H, W 5 512 512
    can_preload = N < 200 and cfg.data.scale < 0.5
    preloaded = cfg.preload and can_preload # cfg.preload:True  can_preload:False

    loader = data.get_random_ordered_batch_loader( #我猜这里是无序加载
        dset,
        cfg.batch_size, #cfg.batch_size 16
        preloaded, #False
    ) #这里非常神奇并且离谱，原数据集中包含77张图片，但是加载器的长度len(loader)为69
    # print("..len(loader)",len(loader))
    # exit(0)
    # print("test120")
    val_loader = data.get_ordered_loader( #我猜这里是有序加载
        dset,
        cfg.batch_size,
        preloaded,
    )
    '''
        数据库中---总共有77张图片
        len(loader): 69*8
        len(val_loader): 5*8 
            每个batch大小为16,数据集大小为77需要5个batch
        ['rgb', 'fwd', 'bck', 'epi', 'occ', 'disocc', 'ske', 'idx']
    '''
    # print("len(loader):",len(loader))
    # print("len(val_loader):", len(val_loader))
    # for batch in val_loader:
    #     #['rgb', 'fwd', 'bck', 'epi', 'occ', 'disocc', 'ske', 'idx']
    #     print("batch",type(batch),len(batch))
    #     print("batch:",dir(batch),batch.keys())
    #     # print("batch[0]", type(batch[0]))
    # print("run_opt.py")
    # exit(0)
    '''
        get_random_ordered_batch_loader 
            函数的作用是创建一个数据加载器，它会以随机顺序加载数据，但同时保持数据的顺序性（可能是按照某种特定的顺序）。
            这种加载方式通常用于训练阶段，因为随机顺序可以避免模型对数据的顺序产生依赖，同时保持数据的顺序性可以确保数据的完整性。
        get_ordered_loader 
            函数的作用是创建一个数据加载器，它会按照固定的顺序加载数据。
            这种加载方式通常用于验证（validation）或测试阶段，因为验证和测试阶段通常需要按照固定的顺序处理数据，
            以确保结果的一致性和可重复性。
    '''
    model = models.SpriteModel(dset,cfg.data.seq ,cfg.n_layers, cfg.model)
    # exit(0)
    model.to(DEVICE) # cuda
    '''
        dset: 数据加载器 <data.CompositeDataset object at 0x7f7c6428ddf0>
        cfg.data.seq: CVAI-2828RAO2_CRA32
        cfg.n_layers: 2
        cfg.model: 模型结构的设置 
        ---------------
        model.has_tex True
    '''

    # determines logging dir in hydra config    # 确定hydra配置中的日志目录
    log_dir = os.getcwd()
    writer = SummaryWriter(log_dir=log_dir)
    print("SAVING OUTPUT TO:", log_dir) # 确定输出路径
    # log_dir: /liuzhicheng2/DNVR/outputs/dev/custom-CVAI-2828RAO2_CRA3test5-gap1-2l/{exp_name}

    if preloaded:
        dset.set_device(DEVICE)
    # optimize the model in phases
    flow_gap = cfg.data.flow_gap
    # flow_gap: 1
    cfg = update_config(cfg, loader)
    save_args = dict(
        writer=writer,          #<torch.utils.tensorboard.writer.SummaryWriter>
        vis_every=cfg.vis_every,#3000
        val_every=cfg.val_every,#6000
        vis_grad=cfg.vis_grad,  #True
        batch_size=cfg.batch_size,#16
        save_grid=cfg.save_grid,#False
    )
    loss_fncs = { #【向前变形损失，向后变形损失】
        "f_warp": MaskWarpLoss(cfg.w_warp, flow_gap),   # MaskWarpLoss() #Mask扭曲损失
        "b_warp": MaskWarpLoss(cfg.w_warp, -flow_gap),  # MaskWarpLoss()
    }

    # loader=val_loader#这句代码正式测试的时候必须删除
    # print("run_opt.py --- main() --- loader=val_loader --- 这句代码正式测试的时候必须删除")
    opt_infer_helper = partial( #opt推断助手
        opt_infer_step,
        loader=loader,
        val_loader=val_loader,
        model=model,
        loss_fncs=loss_fncs,
        **save_args,
    )
    '''
    functools.partial 是 Python 标准库 functools 模块中的一个非常有用的工具，它可以帮助我们创建一个“部分应用函数”（partial function）。
    通过 partial，我们可以将一个函数的某些参数预先绑定为固定值，从而生成一个新的函数。这个新函数在调用时，只需要提供剩余的参数即可。
    2.1 opt_infer_step 是什么？
        opt_infer_step 是一个函数，它可能是用于执行某种优化或推理步骤的函数。
        这个函数的参数包括 loader、val_loader、model、loss_fncs 等。
    2.2 partial 的作用
        partial 将 opt_infer_step 函数的部分参数预先绑定为固定值：
            loader 参数被绑定为 loader。
            val_loader 参数被绑定为 val_loader。
            model 参数被绑定为 model。
            loss_fncs 参数被绑定为 loss_fncs。
            **save_args 是一个字典，它的键值对被展开并作为关键字参数传递给 opt_infer_step。
    2.3 opt_infer_helper 是什么？
        opt_infer_helper 是通过 partial 创建的新函数。它本质上是 opt_infer_step 的一个“简化版”，
        它的部分参数已经被固定，调用时只需要提供 opt_infer_step 中未被绑定的参数即可。
    '''

    # 这几次不同的训练应该只是损失函数的不同
    # 一、warmstart the masks 通过masks进行热开始 #优化MASK几何分割器
    label = "masks"
    model_kwargs = dict(ret_tex=False, ret_tform=False)
    if cfg.epochs_per_phase["epi"] > 0: #epi对应传统的黑塞矩阵MASK
        dset.get_set("epi").save_to(log_dir) #输出了一个不知道是啥的文件
        '''
            dset.get_set(epi): <data.EpipolarDataset object at 0x7b36419c09a0>
            log_dir: /home/lzc/桌面/DeNVeR/outputs/dev/custom-CVAI-2828RAO11_CRA11-gap1-2l/{exp_name}
        '''
        loss_fncs["epi"] = EpipolarLoss(cfg.w_epi,cfg.neg_ratio) # { w_epi:0.5, neg_ratio:0.15 }
        # 第一阶段的三个损失函数，都可以简单理解为不正确像素的个数。
    cfg.epochs_per_phase["kmeans"] = 0
    n_epochs = cfg.epochs_per_phase["epi"] + cfg.epochs_per_phase["kmeans"]
    n_epochs = 1 #在最终训练的过程中这里应该去除 #为啥不能跳过第1阶段
    print("!!!!!这里注释掉了第一阶段的训练过程!!!!!")
    if n_epochs > 0:
        print("epi>0")
        print("model_kwargs:",model_kwargs)
        step_ct, val_dict = opt_infer_helper( #这句代码执行了masks训练过程
            n_epochs, model_kwargs=model_kwargs, label=label
        ) #为啥第二阶段必须要有val_dict
        print("step_ct",step_ct)
    # else:step_ct=0
    # exit(0)

    # print("model.has_tex",model.has_tex)
    if not model.has_tex:#如果模型没有纹理就返回 #这里是True,说明模型有纹理
        return

    # 二、warmstart planar transforms # 热开始平面变换（planar平面）[不知道为啥要训练两遍]
    label = "planar"
    n_epochs = cfg.epochs_per_phase[label]
    n_epochs = 1  # 在最终训练的过程中这里应该去除 #为啥不能跳过第1阶段
    print("!!!!!这里注释掉了第二阶段的训练过程!!!!!")
    print("planar n_epochs",n_epochs)

    loss_fncs["tform"] = FlowWarpLoss(cfg.w_tform, model.tforms,model.fg_tforms ,flow_gap) #光流运动和插值运动结果一致
    loss_fncs["recon"] = ReconLoss(cfg.w_recon, cfg.lap_ratio, cfg.l_recon, cfg.lap_levels) #重构损失
    loss_fncs["contr"] = ContrastiveTexLoss(cfg.w_contr) #纹理对比:逐个像素比较颜色相似度

    # (2.1)
    ok = model.init_planar_motion(val_dict["masks"].to(DEVICE)) # ok=True #初始化面运动是啥
    # 只有在整个前景层没有任何有效点的情况下才会不OK
    # 是否OK取决于前景边界框的计算是否OK：trajectory.py estimate_displacements()
    # print("has_tex",model.has_tex)
    print("ok1",ok)
    # print("这里不是一定会OK么？什么时候会不OK呢？这里竟然不OK，什么原因呢？")
    # exit(0)
    # while not ok:
    if not ok:#这段代码被执行
        # warmstart before estimating scale of textures 在估计纹理比例之前进行热启动
        n_warm = n_epochs // 2
        loss_fncs["tform"].detach_mask = False #进行分割器的优化
        step_ct, val_dict = opt_infer_helper(n_warm, start=step_ct, label=label) #进行训练
        # re-init scale of textures with rough planar motion    # 重新初始化粗糙平面运动纹理的尺度
        ok = model.init_planar_motion(val_dict["masks"].to(DEVICE))
        # 前面不OK，这里也不会OK，不知道对后续操作是否有影响
        # 感觉没啥太大影响，因为不OK就是没有初始化前景关键点的平移，但是后面应该能够自动学习优化
    #     print("flag1")
    # else:
    #     print("flag2")
    # print("test")
    # print("test")
    #     print("ok2", ok)
    # exit(0)

    # (2.2)
    step_ct, val_dict = opt_infer_helper(n_epochs, start=step_ct, label=label) # 这里执行了planar平面训练过程

    # 三、parallel
    label = "parallel"
    n_epochs = cfg.epochs_per_phase["parallel"]
    n_epochs = 1  # 在最终训练的过程中这里应该去除 #为啥不能跳过第1阶段
    print("!!!!!这里注释掉了第三阶段的训练过程!!!!!")
    if cfg.epochs_per_phase["parallel"] > 0:
        loss_fncs["parallel"] = Parallelloss()#平行损失
    print(f"{label} n_epochs",n_epochs)
    step_ct, val_dict = opt_infer_helper(n_epochs, start=step_ct, label=label)
       
    # 四、deform
    # add deformations
    label = "deform"
    model.init_local_motion()
    loss_fncs["tform"].unscaled = True # 每帧每图层使用不同的权重

    n_epochs = cfg.epochs_per_phase[label]
    n_epochs = 1  # 在最终训练的过程中这里应该去除 #为啥不能跳过第1阶段
    print("!!!!!这里注释掉了第四阶段的训练过程!!!!!")
    print(f"{label} n_epochs",n_epochs)
    step_ct, val_dict = opt_infer_helper(n_epochs, start=step_ct, label=label)
    # print("程序中断位置 ---- run_opt.py ---- main() --- 294")
    # exit(0)

    # 五、refine
    # refine masks with gradients through recon loss # 通过重构损失的梯度细化MASK
    # very easy to cheat with these gradients, not recommended # 用这些梯度容易造成欺骗、因此不推荐
    label = "refine"
    n_epochs = cfg.epochs_per_phase[label]
    n_epochs = 1  # 在最终训练的过程中这里应该去除 #为啥不能跳过第1阶段
    print("!!!!!这里注释掉了第五阶段的训练过程!!!!!")
    print(f"{label} n_epochs",n_epochs)
    loss_fncs["recon"].detach_mask = False
    if n_epochs < 1:
        return

    step_ct, val_dict = opt_infer_helper(n_epochs, start=step_ct, label=label)

if __name__ == "__main__":
    main()

'''

配置参数: confs中的全部.yaml文件
数据集处理: helper.py

训练流程管理：run_opt.py(本文件)
三分支主干网络：[ ./models/sprite.py ; SpriteModel ]
    (1)视频分割网络: [ ./models/alpha_pred.py ; AlphaModel ]
    (2)纹理生成网络: 
            [ ./models/tex_gen.py    ; TexUNet    ]
    (2)运动关系建模:
            参数 [ ./models/planar.py     ; PlanarMotion     ] 
            插值 [ ./models/trajectory.py ; PlanarTrajectory ]

-------------------------------------------------------------

训练过程中的损失函数:
    一、masks阶段(训练分割器)
        1.1 MASK黑塞   {epi:EpipolarLoss}
        1.2 MASK光流   {f_warp/b_warp:MaskWarpLoss}
    二、planar阶段 {不明白planar阶段为啥要训练两次}
        2.1 样条光流(训练UV参数、分割器)   {tform:FlowWarpLoss}
        2.2 重构损失(训练逐帧纹理)(包含全局纹理、UV训练)   {recon:ReconLoss}
        2.3 纹理对比(训练逐帧纹理)(包含全局纹理、UV训练)   {contr:ContrastiveTexLoss}
        init_planar_motion 始化整体运动
            if(!ok):-*-*-*- 初始化整体运动失败(init_planar_motion) -*-*-*-
                tform.detach_mask = False # 2.1样条光流优化分割器
                训练
                init_planar_motion 二次始化整体运动
        训练       
    三、parallel阶段
        3.1 流体损失(训练UV参数)   {parallel:Parallelloss}
    四、deform阶段
        init_local_motion    # 开启软体运动模拟
        loss[tform].unscaled # 2.1样条光流加权，每帧每图层使用不同权重。
    五、refine阶段
        recon.detach_mask = False # 2.2重构损失优化分割器

-------------------------------------------------------------

输出产物：
一、masks阶段
二、planar阶段
    init_masks.gif: [
        前景层视频,
        models/trajectory.py/estimate_displacements()
    ]
三、parallel阶段
四、deform阶段
五、refine阶段

-------------------------------------------------------------

export PATH="~/anaconda3/bin:$PATH"
source activate DNVR
python run_opt.py data=custom data.seq=CVAI-2828RAO11_CRA11

'''



