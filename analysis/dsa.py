#!/usr/bin/env python3
"""
XACV 数据集数字减影处理脚本（增强版 v2）
支持多种减影方法：
    mask_minus_live : Mask - 当前帧 (血管亮，常用于使暗血管变亮)
    live_minus_mask : 当前帧 - Mask (血管暗，数据集中血管区域更暗时推荐)
    abs_diff        : |当前帧 - Mask|
    division        : Mask / 当前帧 (像素值归一化后相除，再缩放至原深度)
默认路径：
    输入：../../DeNVeR_in/xca_dataset
    全部帧输出：../outputs/xca_dataset_copy/XCAD_out
    关键帧输出：../outputs/xca_dataset_copy/XCAD_out_sim
"""

import os
import sys
import argparse
from collections import defaultdict

import cv2
import numpy as np


def collect_critical_frames(input_dir):
    """
    遍历所有用户的 ground_truth 文件夹，收集每个 videoID 对应的关键帧 ID 集合。
    """
    critical_frames = defaultdict(set)

    if not os.path.isdir(input_dir):
        print(f"错误：输入目录不存在 -> {input_dir}")
        sys.exit(1)

    for user_id in sorted(os.listdir(input_dir)):
        user_path = os.path.join(input_dir, user_id)
        if not os.path.isdir(user_path):
            continue

        gt_path = os.path.join(user_path, "ground_truth")
        if not os.path.isdir(gt_path):
            continue

        for annot_dir in sorted(os.listdir(gt_path)):
            annot_path = os.path.join(gt_path, annot_dir)
            if not os.path.isdir(annot_path):
                continue

            # 确定视频 ID
            if annot_dir.endswith("CATH"):
                video_id = annot_dir[:-4]
            else:
                video_id = annot_dir

            # 收集帧 ID (去掉 .png 扩展名)
            for fname in sorted(os.listdir(annot_path)):
                if fname.lower().endswith(".png"):
                    frame_id = os.path.splitext(fname)[0]
                    critical_frames[video_id].add(frame_id)

    print(f"已收集 {len(critical_frames)} 个视频的关键帧信息")
    return critical_frames


def subtraction(frame, mask, method):
    """
    根据指定方法执行数字减影。
    参数:
        frame : np.ndarray (uint8 或 uint16)
        mask  : np.ndarray (同 frame 类型和尺寸)
        method: str, 可选 'mask_minus_live', 'live_minus_mask', 'abs_diff', 'division'
    返回:
        np.ndarray, 与输入相同 dtype 的减影结果
    """
    dtype = frame.dtype
    max_val = np.iinfo(dtype).max
    f = frame.astype(np.int32)
    m = mask.astype(np.int32)

    if method == 'sub':
        diff = m - f
        diff = max_val-np.clip(diff, 0, max_val)
        diff = np.clip(diff, 0, max_val)
    elif method == 'subAbs':
        diff = max_val-np.abs(m - f)
        diff = np.clip(diff, 0, max_val)
    elif method == 'div':
        # 转换为 float，避免除零，计算 mask / frame，再缩放至原深度范围
        eps = 1e-6
        f_float = frame.astype(np.float32)/max_val
        m_float = mask.astype(np.float32)/max_val
        # 除法结果可能很大，乘以 max_val 后钳位
        diff = (f_float / (m_float + eps)) * max_val
        diff = np.clip(diff, 0, max_val)
    else:
        raise ValueError(f"未知的减影方法: {method}")

    return diff.astype(dtype)


def process_videos(input_dir, out_all, out_sim, critical_frames, method, no_all_frames):
    """
    遍历所有用户下的 images 文件夹，对每个视频进行数字减影。
    """
    # 仅在需要时创建输出目录
    if not no_all_frames:
        os.makedirs(out_all, exist_ok=True)
    os.makedirs(out_sim, exist_ok=True)

    total_videos = 0
    total_all_frames = 0
    total_sim_frames = 0

    for user_id in sorted(os.listdir(input_dir)):
        user_path = os.path.join(input_dir, user_id)
        if not os.path.isdir(user_path):
            continue

        images_root = os.path.join(user_path, "images")
        if not os.path.isdir(images_root):
            continue

        for video_id in sorted(os.listdir(images_root)):
            video_path = os.path.join(images_root, video_id)
            if not os.path.isdir(video_path):
                continue

            all_frames = sorted(
                [f for f in os.listdir(video_path) if f.lower().endswith(".png")]
            )
            if not all_frames:
                print(f"警告：视频 {video_id} 中没有 PNG 文件，跳过")
                continue

            # 读取蒙片（第一帧，默认为 00000.png）
            first_frame_name = "00000.png"
            first_frame_path = os.path.join(video_path, first_frame_name)
            if not os.path.exists(first_frame_path):
                first_frame_name = all_frames[0]
                first_frame_path = os.path.join(video_path, first_frame_name)
                print(f"注意：{video_id} 中未找到 00000.png，使用 {first_frame_name} 作为蒙片")

            mask_img = cv2.imread(first_frame_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                print(f"错误：无法读取蒙片 -> {first_frame_path}，跳过该视频")
                continue

            has_critical = video_id in critical_frames
            if has_critical:
                out_video_sim = os.path.join(out_sim, video_id)
                os.makedirs(out_video_sim, exist_ok=True)
            else:
                print(f"注意：视频 {video_id} 没有关键帧标注，跳过关键帧输出")

            # 仅当需要输出全部帧时创建对应目录
            if not no_all_frames:
                out_video_all = os.path.join(out_all, video_id)
                os.makedirs(out_video_all, exist_ok=True)

            # 处理该视频的所有帧
            for fname in all_frames:
                frame_path = os.path.join(video_path, fname)
                frame_img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                if frame_img is None:
                    print(f"警告：无法读取帧 -> {frame_path}，跳过")
                    continue

                if frame_img.shape != mask_img.shape:
                    print(f"错误：帧 {fname} 尺寸 {frame_img.shape} 与蒙片 {mask_img.shape} 不符，跳过")
                    continue

                result = subtraction(frame_img, mask_img, method)

                # 保存全部帧（若不禁用）
                if not no_all_frames:
                    out_all_path = os.path.join(out_video_all, fname)
                    cv2.imwrite(out_all_path, result)
                    total_all_frames += 1

                # 保存关键帧
                if has_critical:
                    frame_id = os.path.splitext(fname)[0]
                    if frame_id in critical_frames[video_id]:
                        out_sim_path = os.path.join(out_video_sim, fname)
                        cv2.imwrite(out_sim_path, result)
                        total_sim_frames += 1

            total_videos += 1
            print(f"已处理视频：{video_id}  |  全部帧数：{len(all_frames)}")

    print("\n===== 处理完成 =====")
    print(f"减影方法：{method}")
    print(f"处理视频总数：{total_videos}")
    if not no_all_frames:
        print(f"全部帧减影图像保存至：{out_all}")
        print(f"全部帧总数：{total_all_frames}")
    else:
        print("未输出全部帧减影图像（--no_all_frames 激活）")
    print(f"关键帧减影图像保存至：{out_sim}")
    print(f"关键帧总数：{total_sim_frames}")


def main():
    parser = argparse.ArgumentParser(
        description="XACV 数据集数字减影处理 - 支持多种方法，可跳过全部帧输出"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../../DeNVeR_in/xca_dataset",
        help="输入数据集根目录，默认: ../../DeNVeR_in/xca_dataset",
    )
    parser.add_argument(
        "--out_all",
        type=str,
        default="../outputs/xca_dataset_copy/XCAD_out",
        help="全部帧减影输出目录，默认: ../outputs/xca_dataset_copy/XCAD_out",
    )
    parser.add_argument(
        "--out_sim",
        type=str,
        default="../outputs/xca_dataset_copy/XCAD_out_sim",
        help="关键帧减影输出目录，默认: ../outputs/xca_dataset_copy/XCAD_out_sim",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["sub", "subAbs", "div"],
        default="sub",
        help="减影方法 (默认: live_minus_mask)\n"
             "  sub            : 当前帧 - Mask (血管保持暗，推荐用于暗血管数据集)\n"
             "  subAbs        : |当前帧 - Mask|\n"
             "  div            : Mask / 当前帧 (归一化后相除，增强暗血管对比度)",
    )
    parser.add_argument(
        "--no_all_frames",
        action="store_true",
        help="禁用全部帧减影输出，仅生成关键帧结果",
    )
    args = parser.parse_args()

    # 收集关键帧信息
    critical_frames = collect_critical_frames(args.input_dir)

    # 处理视频
    process_videos(
        args.input_dir,
        args.out_all,
        args.out_sim,
        critical_frames,
        args.method,
        args.no_all_frames,
    )


if __name__ == "__main__":
    main()