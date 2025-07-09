import os
import glob
import imageio
import json
import matplotlib.pyplot as plt
import numpy as np
import torch

from argparse import Namespace

from .flow_viz import flow_to_image


def move_to(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    if isinstance(item, dict):
        return dict([(k, move_to(v, device)) for k, v in item.items()])
    if isinstance(item, (tuple, list)):
        return [move_to(x, device) for x in item]
    print(type(item))
    raise NotImplementedError


def get_unique_log_path(log_dir, resume):
    past_logs = sorted(glob.glob("{}/*".format(log_dir)))
    cur_version = len(past_logs)
    if resume:  # assume resuming the most recent run
        cur_version = max(0, cur_version - 1)
    return "{}/v{:03d}".format(log_dir, cur_version)


def load_checkpoint(path, **kwargs):
    if not os.path.isfile(path):
        print("{} DOES NOT EXIST!".format(path))
        return 0
    print("RESUMING FROM", path)
    ckpt = torch.load(path)
    start_iter = ckpt["i"]
    for name, module in kwargs.items():
        if name not in ckpt:
            print("{} not saved in checkpoint, skipping".format(name))
            continue
        module.load_state_dict(ckpt[name])
    return start_iter


def save_checkpoint(path, i, **kwargs):
    print("ITER {:6d} SAVING CHECKPOINT TO {}".format(i, path))
    save_dict = {name: module.state_dict() for name, module in kwargs.items()}
    save_dict["i"] = i
    torch.save(save_dict, path)


def save_args(args, path):
    with open(path, "w") as f:
        json.dump(vars(args), f, indent=1)


def load_args(path):
    with open(path, "r") as f:
        arg_dict = json.load(f)
    return Namespace(**arg_dict)


def cat_tensor_dicts(dict_list, dim=0):
    if len(dict_list) < 1:
        return {}
    keys = dict_list[0].keys()
    return {k: torch.cat([d[k] for d in dict_list], dim=dim) for k in keys}

def save_vis_dict(out_dir, vis_dict, save_keys=[], skip_keys=[], overwrite=False):
    """
    :param out_dir  # out_dir: 00000019_val_masks #输出路径
    :param vis_dict dict of 4+D tensors
    :param skip_keys (optional) list of keys to skip
    :return the paths each tensor is saved to
    """
    # print("out_dir:", out_dir)
    # print("vis_dict:", len(vis_dict), type(vis_dict),dir(vis_dict))
    # for i in vis_dict:
    #     print("i:", type(i),i)
    # # print("vis_dict[0]:", vis_dict[0].shape, type(vis_dict[0]))
    # items=vis_dict.items()
    # for i in items:
    #     print("i2:", type(i),dir(i))
    # print("items:",type(items),dir(items))
    # # print("items[0]:", type(items[0]))
    # exit(0)
    # 1.检查输出目录是否存在且不允许覆盖
    if os.path.isdir(out_dir) and not overwrite:
        print("{} exists already, skipping".format(out_dir)) #  避免覆盖已有结果
        return #  如果输出目录已存在且不允许覆盖(overwrite=False)，则打印提示信息并直接返回

    # 2.检查可视化字典是否为空
    if len(vis_dict) < 1: # 如果传入的可视化字典为空，返回空列表
        return [] # 提前退出避免不必要操作

    # 3.创建输出目录
    os.makedirs(out_dir, exist_ok=True) # 确保输出目录存在（如果不存在则创建）

    # 4.数据预处理
    vis_dict = {k: v.detach().cpu() for k, v in vis_dict.items()}
    '''
        k masks
        v <class 'torch.Tensor'> torch.Size([10, 2, 1, 128, 128])
        # 解除计算图关联(detach())，避免占用显存
        # 将字典中的张量数据从GPU转移到CPU
    '''

    # 5.确定需要保存的键
    if len(save_keys) < 1: # 如果没有指定保存键(save_keys)，默认保存所有键
        save_keys = vis_dict.keys()
    save_keys = set(save_keys) - set(skip_keys) # 从保存键中排除要跳过的键(skip_keys)

    # 6.批量保存可视化数据
    out_paths = {}
    for name, vis_batch in vis_dict.items():
        if name not in save_keys:
            continue
        if vis_batch is None:
            continue
        out_paths[name] = save_vis_batch(out_dir, name, vis_batch) #保存数据

    # 遍历过滤后的可视化字典
    # 调用save_vis_batch函数实际保存数据
    # 收集每个名称对应的输出路径
    return out_paths

def save_vis_batch(out_dir, name, vis_batch, rescale=False, save_dir=False):
    """
    :param out_dir
    :param name
    :param vis_batch (B, *, C, H, W) first dimension is time dimension
    """
    # 1.维度检查：
    if len(vis_batch.shape) < 4:
        return None # 确保输入至少是4D张量（排除标量或简单向量）

    # 2.通道数处理：
    C = vis_batch.shape[-3]
    if C > 3: # 只处理通道数 ≤3 的数据（灰度图、RGB图、光流图）
        return

    # 3.特殊处理光流图：
    if C == 2:  # is a flow map # 光流图
        vis_batch = flow_to_image(vis_batch) # 将2通道的光流向量转换为3通道的彩色图像

    # 4.归一化处理：# 保持宽高比不变，仅缩放强度值
    if rescale: # rescale=False #这里没有被执行
        vmax = vis_batch.amax(dim=(-1, -2), keepdim=True)# amax 计算每个图像的最大值（保留维度用于广播）
        vmax = torch.clamp_min(vmax, 1)
        vis_batch = vis_batch / vmax # 对每张图像独立归一化到[0,1]范围
        '''
        amax 方法
            功能：沿指定维度计算最大值
            等价操作：torch.max() 的简化版，专用于求最大值
            参数：
                dim=(-1, -2)：在最后两个维度（H和W）上求最大值
                keepdim=True：保持原始维度数（不压缩降维）
        '''

    # 5.保存图像：
    return save_batch_imgs(os.path.join(out_dir, name), vis_batch, save_dir)
def one_dim2three_dim(gray_images):
    frames = []
    rgb_images = np.concatenate((gray_images, gray_images, gray_images), axis=-1)
    # print(rgb_images.shape)    
    return rgb_images
def single_image(gray_images):
    rgb_images = np.concatenate((gray_images, gray_images, gray_images), axis=-1)
    # print(rgb_images.shape)
        
    return rgb_images
def save_batch_imgs(name, vis_batch, save_dir):
    """
    Saves a 4+D tensor of (B, *, 3, H, W) in separate image dirs of B files.
    :param out_dir_pre prefix of output image directories
    :param vis_batch (B, *, 3, H, W)
    """
    # 1.数据准备：
    vis_batch = vis_batch.detach().cpu() #将张量移至CPU并脱离计算图
    B, *dims, C, H, W = vis_batch.shape  #获取张量形状：B=批次大小，*dims=中间维度，C=通道数，H=高度，W=宽度
    vis_batch = vis_batch.view(B, -1, C, H, W) #将中间维度展平，简化处理

    # 2.数据格式转换：
    vis_batch = (255 * vis_batch.permute(0, 1, 3, 4, 2)).byte()
    '''
        缩放值范围：[0,1] → [0,255]
        调整维度顺序：从 (B, M, C, H, W) 变为 (B, M, H, W, C)
        转换为字节类型（8位整数），适合图像保存
    '''

    # 3.遍历所有图像组：
    M = vis_batch.shape[1]# 展平后的中间维度大小
    # print((vis_batch[:, 0]).shape)
    paths = []# 存储所有保存路径
    # current_directory = os.getcwd()
    # print("Current Directory:", current_directory)
    for m in range(M):
        # 4.三种保存模式：
        if B == 1:  # save single image # 保存单张静态图像（当批次大小B=1时）
            path = f"{name}_{m}.png"
            imageio.imwrite(path, vis_batch[0, m]) #保存为png文件
        elif save_dir:  # save directory of images # 保存为图像目录（当save_dir=True时）
            path = f"{name}_{m}"
            save_img_dir(path, vis_batch[:, m])
        else:  # save gif # 保存为GIF动画（默认情况）
            path = f"{name}_{m}.gif"
            if vis_batch[:, m].shape[3]==1:
                frames=one_dim2three_dim(vis_batch[:, m])
            else:
                frames=vis_batch[:, m]
            imageio.mimwrite(path, frames)
        paths.append(path)
    return paths #返回保存的路径

def save_img_dir(out, vis_batch):
    os.makedirs(out, exist_ok=True)
    print("----",out)
    for i in range(len(vis_batch)):
        path = f"{out}/{i:05d}.png" 
        if vis_batch[i].shape[2] == 1:
            frame=single_image(vis_batch[i])
        else:
            frame = vis_batch[i]
        frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
        imageio.imwrite(path,frame )

def save_vid(path, vis_batch):
    """
    :param vis_batch (B, 3, H, W)
    """
    vis_batch = vis_batch.detach().cpu()
    save = (255 * vis_batch.permute(0, 2, 3, 1)).byte()
    imageio.mimwrite(path, save)
