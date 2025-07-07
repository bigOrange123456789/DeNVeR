import os
import glob
from functools import partial

import torch
from torchvision.transforms import functional as TF
from tqdm import tqdm

import data
import models
import utils
from loss import *


DEVICE = torch.device("cuda")

ROOT = os.path.abspath("__file__/..")
def get_dataset(args):#必须先弄清楚输入的数据是什么
    # args.root= ROOT+"/custom_videos"#lzc
    rgb_dir, fwd_dir, bck_dir, gt_dir,ske_dir = data.get_data_dirs(
        args.type, args.root, args.seq, args.flow_gap, args.res
    )
    # print({
    #     "rgb_dir":rgb_dir,
    #     "fwd_dir":fwd_dir,
    #     "bck_dir":bck_dir,
    #     "gt_dir":gt_dir,
    #     "ske_dir":ske_dir
    # })
    # {
    #  'rgb_dir': './custom_videos/PNGImages/CVAI-2828RAO2_CRA32',
    #  'fwd_dir': './custom_videos/raw_flows_gap1/CVAI-2828RAO2_CRA32',
    #  'bck_dir': './custom_videos/raw_flows_gap-1/CVAI-2828RAO2_CRA32',
    #  'gt_dir': None,
    #  'ske_dir': './custom_videos/skeltoize/CVAI-2828RAO2_CRA32'
    #  }

    required_dirs = [rgb_dir, fwd_dir, bck_dir]
    assert all(d is not None for d in required_dirs), required_dirs

    rgb_dset = data.RGBDataset(rgb_dir, scale=args.scale) # 原视频
    fwd_dset = data.FlowDataset(fwd_dir, args.flow_gap, rgb_dset=rgb_dset)  # 前向光流
    bck_dset = data.FlowDataset(bck_dir, -args.flow_gap, rgb_dset=rgb_dset) # 后向光流
    epi_dset = data.EpipolarDataset(fwd_dset,scale=args.scale,seq_name=args.seq) # 黑塞矩阵MASK
    occ_dset = data.OcclusionDataset(fwd_dset, bck_dset)    # 遮挡(行进顺序-前后)
    disocc_dset = data.OcclusionDataset(bck_dset, fwd_dset) # 遮挡(行进顺序-后前)
    ske_dset = data.SKEDataset(ske_dir, scale=args.scale)   # 骨架数据集

    dsets = {
        "rgb": rgb_dset, #custom_videos/PNGImages #原视频
        "fwd": fwd_dset, #custom_videos/raw_flows_gap1  #正向光流
        "bck": bck_dset, #custom_videos/raw_flows_gap-1 #逆向光流
        "epi": epi_dset, #preprocess/--/binary  #黑塞MASK
        "occ": occ_dset,        #遮挡(行进顺序-前后)
        "disocc": disocc_dset,  #遮挡(行进顺序-后前)
        "ske": ske_dset, #custom_videos/skeltoize #骨架
    } # [rgb, fwd, bck, epi, occ, disocc, ske]
    '''
        rgb: #原视频    #custom_videos/PNGImages 
        fwd: #正向光流  #custom_videos/raw_flows_gap1  
        bck: #逆向光流  #custom_videos/raw_flows_gap-1 
        epi: #黑塞掩码  #preprocess/--/binary  
        occ: #正向遮挡
        disocc: #逆向遮挡
        ske: #骨架     #custom_videos/skeltoize 
    '''

    if gt_dir is not None:
        dsets["gt"] = data.MaskDataset(gt_dir, rgb_dset=rgb_dset)
        print("dsets[gt]",len(dsets["gt"]))
    # print({#所有长度都是77
    #     "rgb": len(rgb_dset),
    #     "fwd": len(fwd_dset),
    #     "bck": len(bck_dset),
    #     "epi": len(epi_dset),
    #     "occ": len(occ_dset),
    #     "disocc": len(disocc_dset),
    #     "ske": len(ske_dset),
    # })
    # len(data.CompositeDataset(dsets)) = 10
    # dsets.keys()=['rgb', 'fwd', 'bck', 'epi', 'occ', 'disocc', 'ske']

    return data.CompositeDataset(dsets)


def optimize_model(#优化模型就是在进行训练函数
    n_epochs,
    loader,
    loss_fncs,
    model,
    model_kwargs={},
    start=0,
    label=None,
    writer=None,
    vis_every=0,
    vis_grad=False,
    **kwargs,
):
    step_ct = start #start=0
    out_name = None if label is None else f"tr_{label}"#out_name=tr_masks
    save_vis = vis_every > 0 and out_name is not None #vis_every=3000 save_vis=True
    # print("n_epochs",n_epochs)
    # exit(0)
    for _ in tqdm(range(n_epochs)): #n_epochs:188
        # print("test81")
        for batch in loader:#如果视频帧数比较多、似乎这个加载器会爆内存 #每个batch是一段视频
            # 一个[1,2,..9]和另一个[0,1,..9]
            # print("batch",type(batch))
            # print(batch.keys())
            # print(batch["idx"])
            # exit(0)
            # print("test83")
            # exit(0)
            model.optim.zero_grad() #清零梯度。确保每次计算梯度时不会受到之前计算的影响。
            batch = utils.move_to(batch, DEVICE) #batch的长度为8,device="cuda"
            out_dict = model(batch, **model_kwargs) #model_kwargs: {'ret_tex': False, 'ret_tform': False}
            # 输入原视频、输出的MASK是预测的光流图(因为是双通道)
            # 对于这个损失函数现在有两种理解：(感觉第2种更有可能)
            #       1.预测的MASK视频与RAFT光流一致
            #       2.预测的B样条参数与RAFT光流一致
            loss_dict = compute_losses(loss_fncs, batch, out_dict) # 计算所有损失函数的数值
            # print("loss_fncs",loss_fncs,type(loss_fncs))
            # 在第一阶段实际上有3个损失函数: 前/后向损失函数、MASK损失函数
            # {'f_warp': MaskWarpLoss(), 'b_warp': MaskWarpLoss(), 'epi': EpipolarLoss()}
            # exit(0)
            '''
                name f_warp weight 0.1
                name b_warp weight 0.1
                name epi weight 0.5
                前向扭曲损失
                后向扭曲损失
                另一个是什么？
            '''
            step_ct += len(batch["idx"])#统计处理图片的个数
            # print("batch[idx]",batch["idx"])
            # print("step_ct", step_ct)
            # print("len(loss_dict)",len(loss_dict))
            if len(loss_dict) < 1:
                continue
            sum(loss_dict.values()).backward() #计算梯度
            model.optim.step() #进行参数优化

            if writer is not None:
                for name, loss in loss_dict.items():
                    writer.add_scalar(f"loss/{name}", loss.item(), step_ct)

            # if save_vis and step_ct % vis_every < len(batch["idx"]) :
            #     save_dir = "{:08d}_{}".format(step_ct, out_name)
            #     if vis_grad:
            #         out_dict = get_vis_batch(
            #             batch, model, model_kwargs, loss_fncs, vis_grad
            #         )
            #     utils.save_vis_dict(save_dir, out_dict)
    # print("step_ct_last:",step_ct)
    # exit(0)
    return step_ct


def get_vis_batch(
    batch, model, model_kwargs={}, loss_fncs={}, vis_grad=False, **kwargs
):
    batch = utils.move_to(batch, DEVICE)
    out_dict = model(batch, vis=True, ret_inputs=True, **model_kwargs)

    # save mask gradients if loss functions
    if vis_grad and len(loss_fncs) > 0:
        grad_img, grad_max = get_loss_grad(batch, out_dict, loss_fncs, "pred")
        out_dict["pred_grad"] = grad_img

    return out_dict


def infer_model(
    step_ct,
    loader,
    model,
    model_kwargs={},
    loss_fncs={},
    label=None,
    skip_keys=[],
):
    """
    run the model on all data points
    """
    out_name = None if label is None else f"{step_ct:08d}_val_{label}"
    print("val step {:08d} saving to {}".format(step_ct, out_name))
    out_dicts = []
    for batch in loader:
        batch = utils.move_to(batch, DEVICE)
        with torch.no_grad():
            out_dict = get_vis_batch(batch, model, model_kwargs)
            out_dict = compute_multiple_iou(batch, out_dict)

        # out_dicts.append(
        #     {k: v.detach().cpu() for k, v in out_dict.items() if k not in skip_keys}
        # )
        out_dicts.append(
            {k: v.detach().cpu() for k, v in out_dict.items() if k =='masks'}
        )

    out_dict = utils.cat_tensor_dicts(out_dicts)
    if out_name is not None:
        if "texs" in out_dict:
            out_dict["texs"] = out_dict["texs"][:1]  # (n_batches, *) -> (1, *)
        utils.save_vis_dict(out_name, out_dict)
        # save the per-frame texture coords
        if "coords" in out_dict:
            torch.save(out_dict["coords"], f"{out_name}/coords.pth")
        save_metric(out_name, out_dict, "ious")
    # print(out_dict)
    return out_dict, out_name


def opt_infer_step(
    n_epochs,
    loader,
    val_loader,
    loss_fncs,
    model,
    model_kwargs={},
    start=0,
    val_every=0,
    batch_size=16,
    label="model",
    ckpt=None,
    save_grid=False,
    **kwargs,
): # 优化参数、推理验证、参数保存
    """
    optimizes model for n_epochs, then saves a checkpoint and validation visualizations
    为n_epochs优化模型，然后保存检查点和验证可视化
    """
    if ckpt is None: #ckpt None
        ckpt = "{}_latest_ckpt.pth".format(label)
        # ckpt = masks_latest_ckpt.pth

    step = start # 0
    steps_total = n_epochs * len(loader) * batch_size
    # 6624 = 6 * 69 * 16  # 共有77张图片，但为什么加载器的长度只有69
    val_epochs = max(1, steps_total // val_every)
    # 计算在训练过程中需要进行的验证轮次数量
    # 6624//6000=1
    #不能低于1,两者的整除
    n_epochs_per_val = max(1, n_epochs // val_epochs)
    # 计算在每个验证轮次中包含的训练轮次数量
    # 6//1=6
    print(f"running {val_epochs} train/val steps with {n_epochs_per_val} epochs each.")
    # running 1 train/val steps with 6 epochs each.
    # 运行1次训练/验证步，每次6个周期。(这句话表达了什么含义?)

    # print("label",label)
    # print("model_kwargs",model_kwargs)
    '''
    {
        label masks
        model_kwargs {'ret_tex': False, 'ret_tform': False}
    }
    {
        label planar
        model_kwargs {}
    }    
    '''
    for _ in range(val_epochs): #进行了1次循环
        step = optimize_model(#训练模型、优化参数
            n_epochs_per_val,
            loader,
            loss_fncs,
            model,
            model_kwargs,
            start=step,
            label=label,
            **kwargs,
        )

        utils.save_checkpoint(ckpt, step, model=model) # 用于保存模型的状态和相关信息
        # ckpt masks_latest_ckpt.pth
        # step 3572
       
        val_dict, val_out_dir = infer_model(# 使用模型进行推理验证
            step, val_loader, model, model_kwargs, loss_fncs, label=label
        )
        # print("程序中断位置: helper.py -- opt_infer_step()")
        # exit(0)
        print("label",label) # label=masks
        if label == 'refine': #在细化的时候存储生成视频的每帧图片
            save_res_img_dirs(val_out_dir, val_dict, ["masks"])
            print("val_out_dir",val_out_dir)
            # print("val_dict", val_dict)

        # if save_grid:
        #     save_grid_vis(val_out_dir, val_dict)

    return step, val_dict


def update_config(cfg, loader):
    """
    we provide a min number of iterations for each phase,
    need to update the config to reflect this
    """
    N = len(loader) * cfg.batch_size # len(loader)=0 cfg.batch_size=16
    # print("len(loader)",len(loader))
    print(len(loader) , cfg.batch_size,"len(loader) * cfg.batch_size")
    for phase, epochs in cfg.epochs_per_phase.items():
        n_iters = cfg.iters_per_phase[phase]
        cfg.epochs_per_phase[phase] = max(n_iters // N + 1, epochs)

    if cfg.n_layers <= 2:
        cfg.w_kmeans *= 0.1
        cfg.epochs_per_phase["kmeans"] = cfg.epochs_per_phase["kmeans"] // 10

    # also update the vis and val frequency in iterations
    cfg.vis_every = max(cfg.vis_every, cfg.vis_epochs * N)
    cfg.val_every = max(cfg.val_every, cfg.val_epochs * N)
    print("epochs_per_phase", cfg.epochs_per_phase)
    print("vis_every", cfg.vis_every)
    print("val_every", cfg.val_every)
    return cfg


def save_metric(out_dir, out_dict, name="ious"):
    os.makedirs(out_dir, exist_ok=True)
    if name not in out_dict:
        return

    vec = out_dict[name].detach().cpu()
    if len(vec.shape) > 2:
        return

    ok = (vec >= 0).all(dim=-1)
    vec = vec[ok]
    np.savetxt(os.path.join(out_dir, f"frame_{name}.txt"), vec)
    np.savetxt(os.path.join(out_dir, f"mean_{name}.txt"), vec.mean(dim=0))
    print(name, vec.mean(dim=0))


def compute_multiple_iou(batch_in, batch_out):
    """
    :param masks (B, M, *, H, W)
    :param gt (B, C, H, W)
    :returns iou (B, M) chooses the best iou for each mask
    """
    if "gt" not in batch_in:
        return batch_out

    gt, ok = batch_in["gt"]
    if ok.sum() < 1:
        return batch_out

    with torch.no_grad():
        masks = batch_out["masks"]

        B, C, H, W = gt.shape
        masks_bin = masks.view(B, -1, 1, H, W) > 0.5
        gt_bin = gt.view(B, 1, C, H, W) > 0.5
        ious = utils.compute_iou(masks_bin, gt_bin, dim=(-1, -2))  # (B, M, C)
        ious = ious.amax(dim=-1)  # (B, M)
        ious[~ok] = -1

    batch_out["ious"] = ious
    return batch_out


def save_grid_vis(out_dir, vis_dict, pad=4):
    grid_keys = ["rgb", "recons", "layers", "texs", "view_vis"]
    if not all(x in vis_dict for x in grid_keys):
        print(f"not all keys in vis_dict, cannot save to {out_dir}")
        return

    vis_dict = {k: v.detach().cpu() for k, v in vis_dict.items()}
    os.makedirs(out_dir, exist_ok=True)
    grid = make_grid_vis(vis_dict, pad=pad)
    grid_path = os.path.join(out_dir, "grid_vis.mp4")
    utils.save_vid(grid_path, grid)


def save_res_img_dirs(out_dir, vis_dict, save_keys):
    for save in save_keys:
        save_dir = os.path.join(out_dir, save)
        utils.save_batch_imgs(save_dir, vis_dict[save], True)


def make_grid_vis(vis_dict, pad=4):
    """
    make panel vis with input, layers, view_vis, textures, and recon
    :param rgb (B, 3, H, W)
    :param recons (B, 3, H, W)
    :param layers (B, M, 3, H, W)
    :param texs (1, M, 3, H, W)
    :param view_vis (B, M, 3, H, W)
    """
    required = ["rgb", "recons", "layers", "texs", "view_vis"]
    if not all(x in vis_dict for x in required):
        print(f"not all keys in vis_dict, cannot make grid vis")
        return

    rgb = vis_dict["rgb"]
    N, _, h, w = rgb.shape
    texs_rs = TF.resize(
        vis_dict["texs"][0], size=(h, w), antialias=True
    )  # (M, 3, h, w)
    texs_rs = texs_rs[None].repeat(N, 1, 1, 1, 1)  # (N, M, 3, h, w)

    texs_vert = pad_cat_groups_vert(texs_rs, pad=pad)
    layers_vert = pad_cat_groups_vert(vis_dict["layers"], pad=pad)
    tforms_vert = pad_cat_groups_vert(vis_dict["view_vis"], pad=pad)

    N, _, H, _ = texs_vert.shape
    diff = (H - h) // 2
    rgb_pad = TF.pad(rgb, (0, diff, pad, H - h - diff), fill=1)
    recon_pad = TF.pad(vis_dict["recons"], (pad, diff, 0, H - h - diff), fill=1)

    final = torch.cat([rgb_pad, texs_vert, tforms_vert, layers_vert, recon_pad], dim=-1)
    return final


def pad_cat_groups_vert(tensor, pad=4):
    """
    :param tensor (B, M, 3, h, w)
    :param pad (int)
    """
    padded = TF.pad(tensor, pad, fill=1)  # (B, M, 3, h+2*pad, w+2*pad)
    B, M, C, H, W = padded.shape
    catted = padded.transpose(1, 2).reshape(B, C, -1, W)
    return catted[..., pad:-pad, :]  # remove top-most and bottom-most padding
