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
def main(cfg: DictConfig):
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

    loader = data.get_random_ordered_batch_loader(#我猜这里是无序加载
        dset,
        cfg.batch_size,#cfg.batch_size 16
        preloaded, #False
    )
    val_loader = data.get_ordered_loader(#我猜这里是有序加载
        dset,
        cfg.batch_size,
        preloaded,
    )
    '''
        get_random_ordered_batch_loader 
            函数的作用是创建一个数据加载器，它会以随机顺序加载数据，但同时保持数据的顺序性（可能是按照某种特定的顺序）。
            这种加载方式通常用于训练阶段，因为随机顺序可以避免模型对数据的顺序产生依赖，同时保持数据的顺序性可以确保数据的完整性。
        get_ordered_loader 
            函数的作用是创建一个数据加载器，它会按照固定的顺序加载数据。
            这种加载方式通常用于验证（validation）或测试阶段，因为验证和测试阶段通常需要按照固定的顺序处理数据，
            以确保结果的一致性和可重复性。
    '''
    # print(f'data_name: {cfg.data.seq}')
    model = models.SpriteModel(dset,cfg.data.seq ,cfg.n_layers, cfg.model)
    model.to(DEVICE)

    # determines logging dir in hydra config #确定hydra配置中的日志目录
    log_dir = os.getcwd()
    writer = SummaryWriter(log_dir=log_dir)
    print("SAVING OUTPUT TO:", log_dir)
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
    loss_fncs = {
        "f_warp": MaskWarpLoss(cfg.w_warp, flow_gap),   # MaskWarpLoss()
        "b_warp": MaskWarpLoss(cfg.w_warp, -flow_gap),  # MaskWarpLoss()
    }

    opt_infer_helper = partial(
        opt_infer_step,
        loader=loader,
        val_loader=val_loader,
        model=model,
        loss_fncs=loss_fncs,
        **save_args,
    )

    # warmstart the masks
    label = "masks"
    model_kwargs = dict(ret_tex=False, ret_tform=False)
    if cfg.epochs_per_phase["epi"] > 0:
        dset.get_set("epi").save_to(log_dir)
        loss_fncs["epi"] = EpipolarLoss(cfg.w_epi,cfg.neg_ratio)
    cfg.epochs_per_phase["kmeans"]=0
    n_epochs = cfg.epochs_per_phase["epi"] + cfg.epochs_per_phase["kmeans"] 
    if n_epochs > 0:
        print("epi>0")
        step_ct, val_dict = opt_infer_helper(
            n_epochs, model_kwargs=model_kwargs, label=label
        )

    if not model.has_tex:
        return

    # warmstart planar transforms
    label = "planar"
    n_epochs = cfg.epochs_per_phase[label] 
    print("planar n_epochs",n_epochs)

    loss_fncs["tform"] = FlowWarpLoss(cfg.w_tform, model.tforms,model.fg_tforms ,flow_gap)
    loss_fncs["recon"] = ReconLoss(
        cfg.w_recon, cfg.lap_ratio, cfg.l_recon, cfg.lap_levels
    )
    loss_fncs["contr"] = ContrastiveTexLoss(cfg.w_contr)

    ok = model.init_planar_motion(val_dict["masks"].to(DEVICE))
    if not ok:
        # warmstart before estimating scale of textures
        n_warm = n_epochs // 2
        loss_fncs["tform"].detach_mask = False
        step_ct, val_dict = opt_infer_helper(
            n_warm, start=step_ct, label=label)
        # re-init scale of textures with rough planar motion
        model.init_planar_motion(val_dict["masks"].to(DEVICE))

    step_ct, val_dict = opt_infer_helper(n_epochs, start=step_ct, label=label)

    label = "parallel"
    n_epochs = cfg.epochs_per_phase["parallel"]
    if cfg.epochs_per_phase["parallel"] > 0:
        loss_fncs["parallel"] = Parallelloss()
    print(f"{label} n_epochs",n_epochs)
    step_ct, val_dict = opt_infer_helper(n_epochs, start=step_ct, label=label)
    
       
    # add deformations
    label = "deform"
    model.init_local_motion()
    loss_fncs["tform"].unscaled = True

    n_epochs = cfg.epochs_per_phase[label] 
    print(f"{label} n_epochs",n_epochs)
    step_ct, val_dict = opt_infer_helper(n_epochs, start=step_ct, label=label)

    # refine masks with gradients through recon loss
    # very easy to cheat with these gradients, not recommended
    label = "refine"
    n_epochs = cfg.epochs_per_phase[label] 
    print(f"{label} n_epochs",n_epochs)
    loss_fncs["recon"].detach_mask = False
    if n_epochs < 1:
        return

    step_ct, val_dict = opt_infer_helper(n_epochs, start=step_ct, label=label)


if __name__ == "__main__":
    main()
