import os
import struct
import numpy as np
from PIL import Image
import cv2
import yaml
from collections import defaultdict

# ==================== 辅助函数 ====================
def read_flo(path):
    with open(path, 'rb') as f:
        magic = struct.unpack('i', f.read(4))[0]
        if magic == 1234567890:
            byte_order = 'i'
        elif magic == 1212500304:
            byte_order = 'I'
        else:
            raise ValueError(f"Invalid .flo magic: {magic}")
        w = struct.unpack(byte_order, f.read(4))[0]
        h = struct.unpack(byte_order, f.read(4))[0]
        data = struct.unpack(f'{h*w*2}f', f.read(h*w*2*4))
        return np.array(data, dtype=np.float32).reshape((h, w, 2))

def read_mask(path, normalize=True):
    img = Image.open(path).convert('L')
    mask = np.array(img, dtype=np.float32)
    if normalize:
        mask /= 255.0
    return mask

def angle_between(v1, v2):
    """计算两个向量的夹角（弧度）"""
    norm_v1 = np.linalg.norm(v1) + 1e-8
    norm_v2 = np.linalg.norm(v2) + 1e-8
    cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.arccos(cos_theta)

def compute_metrics(pred_bin, gt_bin, valid_mask=None):
    if valid_mask is None:
        valid_mask = np.ones_like(pred_bin, dtype=bool)
    tp = np.sum(pred_bin & gt_bin & valid_mask)
    fp = np.sum(pred_bin & (~gt_bin) & valid_mask)
    fn = np.sum((~pred_bin) & gt_bin & valid_mask)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    dice = 2 * tp / (2*tp + fp + fn + 1e-8)
    return {'TP': tp, 'FP': fp, 'FN': fn,
            'Precision': precision, 'Recall': recall, 'Dice': dice}

def generate_angle_visualization(flow, mask, mean_vec, output_path):
    """
    生成方向差异可视化图：
      - 背景为绿色 (0,255,0)
      - 前景（mask）颜色根据方向偏移角度映射：
          偏移0° -> 红色 (255,0,0)
          偏移180° -> 黑色 (0,0,0)
          线性插值
    """
    H, W = flow.shape[:2]
    # 创建背景（绿色）
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :, 1] = 255  # 绿色背景

    # 获取 mask 内像素坐标
    indices = np.where(mask)
    if len(indices[0]) == 0:
        Image.fromarray(img).save(output_path)
        return

    vectors = flow[indices]  # (N,2)
    # 计算每个向量与平均向量的夹角（度）
    angles_deg = np.zeros(len(vectors))
    for i, v in enumerate(vectors):
        ang = angle_between(v, mean_vec)
        angles_deg[i] = np.degrees(ang)

    # 归一化到 [0,1]（偏移0->1，偏移180->0）
    norm = 1.0 - angles_deg / 180.0
    norm = np.clip(norm, 0, 1)

    # 颜色：红色分量 = norm, 绿色和蓝色为0
    red = (norm * 255).astype(np.uint8)
    for i, (y, x) in enumerate(zip(indices[0], indices[1])):
        img[y, x, 0] = red[i]   # R
        img[y, x, 1] = 0        # G
        img[y, x, 2] = 0        # B

    Image.fromarray(img).save(output_path)

# ==================== 主分析函数 ====================
def analyze_angle_extremes(tag, video_list=None, black_thresh=50, white_thresh=200):
    """
    找出：
        - 正确区域（TP）中偏移角度最大的帧（及具体像素）
        - 错误区域（FP）中偏移角度最小的帧（及具体像素）
    并生成可视化图像。
    """
    # 读取配置
    script_path = os.path.abspath(__file__)
    ROOT1 = os.path.dirname(script_path)
    config_path = os.path.join(ROOT1, "../..", 'confs/newConfig.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    datasetPath = config["my"]["datasetPath_rigid.in"]
    customPath = config["my"]["datasetPath_rigid.in_custom"]
    out_root = config["my"]["filePathRoot"]
    output_subdir = config["my"]["subPath"]["outputs"]
    vis_dir = os.path.join(out_root, output_subdir, "angle_vis")

    # 记录极值信息
    max_tp_angle = -1          # 全局最大TP角度
    max_tp_frame = None        # (userId, videoId, frame_id, gt_path, pred_path)
    max_tp_coords = []         # 该帧中达到该角度的像素坐标列表

    min_fp_angle = 181         # 全局最小FP角度
    min_fp_frame = None
    min_fp_coords = []

    # 确定要处理的视频列表
    if video_list is None:
        items = []
        for userId in os.listdir(datasetPath):
            user_img_dir = os.path.join(datasetPath, userId, "images")
            if not os.path.isdir(user_img_dir):
                continue
            for videoId in os.listdir(user_img_dir):
                items.append((userId, videoId))
    else:
        items = video_list

    total_frames = 0
    for userId, videoId in items:
        img_dir = os.path.join(datasetPath, userId, "images", videoId)
        if not os.path.isdir(img_dir):
            print(f"警告：{img_dir} 不存在，跳过")
            continue

        gt_dir = os.path.join(datasetPath, userId, "ground_truth", videoId + "CATH")
        if not os.path.isdir(gt_dir):
            print(f"警告：GT 目录 {gt_dir} 不存在，跳过")
            continue

        pred_dir = os.path.join(config["my"]["datasetPath_rigid.out"],
                                userId, "decouple", videoId, tag + ".orig_mask")

        frame_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        for frame_file in frame_files:
            frame_id = frame_file.split('.png')[0]
            if int(frame_id) == 0 or int(frame_id) == len(frame_files) - 1:
                continue

            fwd_flo = os.path.join(customPath, "raw_flows_gap1", videoId, frame_id + ".flo")
            bwd_flo = os.path.join(customPath, "raw_flows_gap-1", videoId, frame_id + ".flo")
            pred_mask = os.path.join(pred_dir, frame_id + ".png")
            gt_mask = os.path.join(gt_dir, frame_id + ".png")

            if not all(os.path.exists(p) for p in [fwd_flo, bwd_flo, pred_mask, gt_mask]):
                continue

            # 读取数据
            flow_f = read_flo(fwd_flo)
            pred = read_mask(pred_mask, normalize=True)
            gt_raw = read_mask(gt_mask, normalize=False)

            gt_gray = (gt_raw > black_thresh) & (gt_raw < white_thresh)
            valid = ~(gt_raw >= white_thresh)

            pred_bin = pred > 0.5

            # 只考虑有效区域内的前景像素（避免背景光流影响）
            valid_foreground = pred_bin & valid
            if np.sum(valid_foreground) < 2:
                continue  # 像素太少，无法计算平均方向

            # 计算整图平均方向
            vectors_all = flow_f[valid_foreground]
            mean_vec_all = np.mean(vectors_all, axis=0)

            # 正确区域（TP）
            tp_mask = pred_bin & gt_gray & valid
            if np.sum(tp_mask) > 0:
                tp_vectors = flow_f[tp_mask]
                # 计算每个TP像素的偏移角度
                for idx, vec in enumerate(tp_vectors):
                    ang = np.degrees(angle_between(vec, mean_vec_all))
                    if ang > max_tp_angle:
                        max_tp_angle = ang
                        max_tp_frame = (userId, videoId, frame_id, gt_mask, pred_mask)
                        # 记录该帧中所有达到该角度的像素（可能存在多个）
                        # 重新收集该帧所有TP像素的坐标和角度
                        # 注意：由于可能存在多个像素角度相同，这里重新收集
                        y, x = np.where(tp_mask)
                        # 重新计算该帧所有TP像素的角度
                        angles_frame = []
                        coords_frame = []
                        for (yy, xx), v in zip(zip(y, x), tp_vectors):
                            a = np.degrees(angle_between(v, mean_vec_all))
                            if a == max_tp_angle:  # 浮点比较，建议用容差
                                coords_frame.append((yy, xx))
                        max_tp_coords = coords_frame

            # 错误区域（FP）
            fp_mask = pred_bin & (~gt_gray) & valid
            if np.sum(fp_mask) > 0:
                fp_vectors = flow_f[fp_mask]
                for idx, vec in enumerate(fp_vectors):
                    ang = np.degrees(angle_between(vec, mean_vec_all))
                    if ang < min_fp_angle:
                        min_fp_angle = ang
                        min_fp_frame = (userId, videoId, frame_id, gt_mask, pred_mask)
                        y, x = np.where(fp_mask)
                        coords_frame = []
                        for (yy, xx), v in zip(zip(y, x), fp_vectors):
                            a = np.degrees(angle_between(v, mean_vec_all))
                            if a == min_fp_angle:
                                coords_frame.append((yy, xx))
                        min_fp_coords = coords_frame

            total_frames += 1
            if True:#total_frames % 100 == 0:
                print(f"\r已处理 {total_frames} 帧", end='')

    print(f"\n分析完成，共处理 {total_frames} 帧。")

    # ==================== 输出极值帧信息 ====================
    print("\n" + "="*60)
    print("【极值帧信息】")
    if max_tp_frame:
        print("\n1. 正确区域（TP）中偏移角度最大的帧:")
        print(f"   角度: {max_tp_angle:.2f}°")
        print(f"   userId: {max_tp_frame[0]}, videoId: {max_tp_frame[1]}, frameId: {max_tp_frame[2]}")
        print(f"   GT路径: {max_tp_frame[3]}")
        print(f"   预测路径: {max_tp_frame[4]}")
        print(f"   达到该角度的像素数: {len(max_tp_coords)}")
        print(f"   示例坐标（前10个）: {max_tp_coords[:10]}")
    else:
        print("\n1. 未找到正确的预测区域。")

    if min_fp_frame:
        print("\n2. 错误区域（FP）中偏移角度最小的帧:")
        print(f"   角度: {min_fp_angle:.2f}°")
        print(f"   userId: {min_fp_frame[0]}, videoId: {min_fp_frame[1]}, frameId: {min_fp_frame[2]}")
        print(f"   GT路径: {min_fp_frame[3]}")
        print(f"   预测路径: {min_fp_frame[4]}")
        print(f"   达到该角度的像素数: {len(min_fp_coords)}")
        print(f"   示例坐标（前10个）: {min_fp_coords[:10]}")
    else:
        print("\n2. 未找到错误的预测区域。")

    # ==================== 生成可视化 ====================
    # 为极值帧生成方向差异图
    if max_tp_frame:
        # 读取该帧的数据用于生成可视化
        userId, videoId, frame_id, gt_path, pred_path = max_tp_frame
        fwd_flo = os.path.join(customPath, "raw_flows_gap1", videoId, frame_id + ".flo")
        flow_f = read_flo(fwd_flo)
        pred = read_mask(pred_path, normalize=True)
        pred_bin = pred > 0.5
        valid_foreground = pred_bin  # 简化：全前景（因为该帧有TP像素）
        vectors_all = flow_f[valid_foreground]
        mean_vec_all = np.mean(vectors_all, axis=0)

        vis_output = os.path.join(vis_dir, f"{videoId}_{frame_id}_TP_max_angle_{max_tp_angle:.0f}.png")
        os.makedirs(os.path.dirname(vis_output), exist_ok=True)
        generate_angle_visualization(flow_f, pred_bin, mean_vec_all, vis_output)
        print(f"\n   可视化保存至: {vis_output}")

    if min_fp_frame:
        userId, videoId, frame_id, gt_path, pred_path = min_fp_frame
        fwd_flo = os.path.join(customPath, "raw_flows_gap1", videoId, frame_id + ".flo")
        flow_f = read_flo(fwd_flo)
        pred = read_mask(pred_path, normalize=True)
        pred_bin = pred > 0.5
        vectors_all = flow_f[pred_bin]
        mean_vec_all = np.mean(vectors_all, axis=0)

        vis_output = os.path.join(vis_dir, f"{videoId}_{frame_id}_FP_min_angle_{min_fp_angle:.0f}.png")
        generate_angle_visualization(flow_f, pred_bin, mean_vec_all, vis_output)
        print(f"\n   可视化保存至: {vis_output}")

    print("="*60)

# ==================== 使用示例 ====================
if __name__ == "__main__":
    tag = "A26-03"
    # 可选指定部分视频以加快分析
    # video_list = [("CVAI-1207", "CVAI-1207RAO2_CAU30")]
    video_list = None  # 处理全部

    analyze_angle_extremes(tag, video_list)


