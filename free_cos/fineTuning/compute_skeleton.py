"""
compute_skeleton.py

计算原始分割掩码的骨架图，并评估骨架作为分割结果的指标。
使用方法：
    python compute_skeleton.py [--tag TAG] [--video_list LIST] [--save_skeleton] [--output_dir DIR]

参数说明：
    --tag: 预测掩码的标签（例如 A26-03），默认 'A26-03'
    --video_list: 可选，指定要处理的视频列表，格式如 "user1,video1 user2,video2" 或留空处理所有
    --save_skeleton: 是否保存骨架图（默认 True）
    --output_dir: 骨架图输出目录（默认为原始预测目录下的 skeletons 子目录）
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
import cv2
import yaml
from skimage.morphology import skeletonize

# 假设脚本位于项目根目录下的某个位置，根据实际调整
script_path = os.path.abspath(__file__)
ROOT1 = os.path.dirname(script_path)
file_path = os.path.join(ROOT1, "../..", 'confs/newConfig.yaml')  # 根据实际路径调整

# ---------- 读取工具（复用之前代码） ----------
def read_mask(path, normalize=False):
    """读取灰度掩码，可选归一化到 [0,1]"""
    img = Image.open(path).convert('L')
    mask = np.array(img, dtype=np.float32)
    if normalize:
        mask /= 255.0
    return mask

def compute_metrics(pred_bin, gt_bin, valid_mask=None):
    """计算 TP, FP, FN, Dice, Precision, Recall"""
    if valid_mask is None:
        valid_mask = np.ones_like(pred_bin, dtype=bool)
    tp = np.sum(pred_bin & gt_bin & valid_mask)
    fp = np.sum(pred_bin & (~gt_bin) & valid_mask)
    fn = np.sum((~pred_bin) & gt_bin & valid_mask)
    dice = 2 * tp / (2*tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return {'TP': tp, 'FP': fp, 'FN': fn,
            'Dice': dice, 'Precision': precision, 'Recall': recall}

# ---------- 骨架提取 ----------
def compute_skeleton(mask_bin):
    """
    输入二值掩码 (0/1 或 0/255)，返回骨架图 (0/1 二值)
    """
    # 确保输入是 0/1 的 bool 数组
    if mask_bin.dtype != bool:
        mask_bin = mask_bin > 0
    skeleton = skeletonize(mask_bin)  # 返回 bool 数组
    return skeleton

# ---------- 主处理 ----------
def main():
    parser = argparse.ArgumentParser(description='Compute skeleton of segmentation masks and evaluate as segmentation.')
    parser.add_argument('--tag', type=str, default='A26-03', help='Tag of prediction masks')
    parser.add_argument('--video_list', type=str, nargs='+', default=None,
                        help='List of video ids, format: userId,videoId (e.g., user1,video1 user2,video2). If not provided, process all videos.')
    parser.add_argument('--save_skeleton', action='store_true', default=True,
                        help='Save skeleton images')
    parser.add_argument('--output_dir', type=str, default="./log_27/outputs/skeletons_A26-03",
                        help='Directory to save skeleton images (default: {original_pred_dir}/skeletons)')
    parser.add_argument('--need_ana', action='store_true', default=True,
                        help='Compute evaluation metrics (need GT)')
    args = parser.parse_args()

    # 加载配置
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    datasetPath = config["my"]["datasetPath_rigid.in"]
    customPath = config["my"]["datasetPath_rigid.in_custom"]  # 可能不需要，但保留路径

    # 解析视频列表
    if args.video_list is not None:
        items = []
        for item in args.video_list:
            parts = item.split(',')
            if len(parts) == 2:
                items.append((parts[0], parts[1]))
            else:
                print(f"警告：视频列表格式错误 {item}，应为 userId,videoId")
        if not items:
            print("没有有效的视频列表，退出。")
            return
    else:
        # 收集所有视频
        items = []
        for userId in os.listdir(datasetPath):
            user_img_dir = os.path.join(datasetPath, userId, "images")
            if not os.path.isdir(user_img_dir):
                continue
            for videoId in os.listdir(user_img_dir):
                items.append((userId, videoId))

    # 初始化累加器
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_frames = 0

    # 遍历视频
    for userId, videoId in items:
        img_dir = os.path.join(datasetPath, userId, "images", videoId)
        if not os.path.isdir(img_dir):
            print(f"警告：图像目录不存在 {img_dir}，跳过")
            continue

        # 构造预测掩码路径
        outpath = os.path.join(config["my"]["datasetPath_rigid.out"], userId, "decouple", videoId)
        pred_dir = os.path.join(outpath, args.tag + ".orig_mask")
        if not os.path.isdir(pred_dir):
            print(f"警告：预测掩码目录不存在 {pred_dir}，跳过")
            continue

        # 构造骨架输出目录
        if args.save_skeleton:
            if args.output_dir is None:
                skeleton_dir = os.path.join(pred_dir, "skeletons")
            else:
                skeleton_dir = os.path.join(args.output_dir, videoId)
            os.makedirs(skeleton_dir, exist_ok=True)

        # 获取所有帧文件
        frames = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        frames.sort(key=lambda x: int(x.split('.')[0]))

        for idx, frame_file in enumerate(frames):
            frame_id = frame_file.split('.')[0]
            # 跳过第一帧和最后一帧（光流可能不存在，但这里只读预测，可以全部处理）
            # 但为了与之前一致，这里不跳过，因为不需要光流。不过GT需要与帧对应。
            # 如果要与之前一样忽略首尾，可以取消注释：
            # if int(frame_id) == 0 or int(frame_id) == len(frames)-1:
            #     continue

            # 读取预测掩码
            pred_path = os.path.join(pred_dir, frame_file)
            if not os.path.exists(pred_path):
                print(f"警告：预测掩码不存在 {pred_path}，跳过")
                continue
            pred_mask = read_mask(pred_path, normalize=False)  # 0-255
            pred_bin = pred_mask > 127  # 二值化（假设阈值0.5对应127）

            # 计算骨架
            skeleton = compute_skeleton(pred_bin)  # bool

            # 保存骨架图
            if args.save_skeleton:
                skeleton_img = Image.fromarray((skeleton * 255).astype(np.uint8))
                out_path = os.path.join(skeleton_dir, frame_file)
                skeleton_img.save(out_path)
                print("out_path",out_path)

            # 如果需要评估
            if args.need_ana:
                # 读取GT
                gt_path = os.path.join(datasetPath, userId, "ground_truth", videoId + "CATH", frame_file)
                if not os.path.exists(gt_path):
                    # print(f"\r警告：GT不存在 {gt_path}，跳过", end="")
                    continue
                gt_raw = read_mask(gt_path, normalize=False)  # 0-255
                # 有效区域排除白色导管（>=200）
                valid = gt_raw < 200  # 白色导管忽略
                # GT前景：灰色区域（50 < value < 200）
                gt_gray = (gt_raw > 50) & (gt_raw < 200)

                # 将骨架与GT比较
                metrics = compute_metrics(skeleton, gt_gray, valid)
                total_tp += metrics['TP']
                total_fp += metrics['FP']
                total_fn += metrics['FN']
                total_frames += 1

                # 可选：打印每帧信息
                # print(f"{userId}/{videoId}/{frame_id}: Precision={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}")

            total_frames += 1  # 没有评估时也计数，但这里上面已经计了，避免重复，调整逻辑

        # print("userId:", userId)
        # print("videoId", videoId)
        # exit(0)

    # 输出总体评估结果
    if args.need_ana and total_frames > 0:
        dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-8)
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        print("\n" + "="*50)
        print("骨架分割评估结果（所有帧聚合）")
        print(f"总帧数: {total_frames}")
        print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
        print(f"Dice: {dice:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print("="*50)
    else:
        print("未进行评估（未提供GT或没有有效帧）")

if __name__ == "__main__":
    main() 

