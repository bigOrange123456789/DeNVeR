import os
import struct
import numpy as np
from PIL import Image
import cv2
import yaml

# ==================== 辅助函数 ====================
def read_flo(path):
    """读取 .flo 光流文件，返回 (H, W, 2) numpy 数组"""
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
    """读取灰度掩码，可选归一化到 [0,1]"""
    img = Image.open(path).convert('L')
    mask = np.array(img, dtype=np.float32)
    if normalize:
        mask /= 255.0
    return mask

def angle_between(v1, v2):
    """计算两个向量的夹角（弧度）"""
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.arccos(cos_theta)

def filter_by_direction(flow, mask, angle_threshold_deg=30):
    """
    根据平均方向剔除方向差异较大的像素
    flow: (H,W,2) 光流
    mask: (H,W) bool 二值掩码
    angle_threshold_deg: 方向差异阈值（度）
    返回新的二值掩码 (bool)
    """
    if np.sum(mask) < 2:
        return mask  # 像素太少，不处理

    # 获取掩码内所有光流向量
    vectors = flow[mask]  # (N,2)
    mean_vec = np.mean(vectors, axis=0)  # 平均向量

    # 计算每个向量与平均向量的夹角
    angles = np.array([angle_between(v, mean_vec) for v in vectors])
    # 转换为度
    angles_deg = np.degrees(angles)

    # 保留夹角小于阈值的像素
    keep = angles_deg <= angle_threshold_deg

    # 构建新掩码（在原掩码位置中保留符合条件的）
    new_mask = np.zeros_like(mask)
    indices = np.where(mask)
    new_mask[indices[0][keep], indices[1][keep]] = True
    return new_mask

def compute_metrics(pred_bin, gt_bin, valid_mask=None):
    """计算 TP, FP, FN, Precision, Recall, Dice"""
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

# ==================== 主处理函数 ====================
def process_video_with_direction_filter(tag, video_list=None, angle_threshold=30,
                                        black_thresh=50, white_thresh=200):
    """
    对整个数据集进行方向滤波处理并统计分析
    tag: 预测掩码的标签（如 'A26-03'）
    video_list: 要处理的视频列表，每个元素为 (userId, videoId)；None 则处理所有
    angle_threshold: 方向差异阈值（度）
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

    # 统计聚合变量（全数据集）
    total_removed_not_gt = 0   # 去除的区域中不属于GT的像素数
    total_removed_pixels = 0   # 去除的总像素数

    total_tp_orig = 0
    total_fp_orig = 0
    total_fn_orig = 0
    total_tp_refined = 0
    total_fp_refined = 0
    total_fn_refined = 0

    # 记录每帧的查准率变化，用于找出极值
    delta_precision_list = []  # 每个元素为 (frame_path, delta)

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

        # GT 目录
        gt_dir = os.path.join(datasetPath, userId, "ground_truth", videoId + "CATH")
        if not os.path.isdir(gt_dir):
            print(f"警告：GT 目录 {gt_dir} 不存在，跳过评估")
            continue

        # 预测掩码目录
        pred_dir = os.path.join(config["my"]["datasetPath_rigid.out"],
                                userId, "decouple", videoId, tag + ".orig_mask")

        # 遍历所有帧
        frame_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        for frame_file in frame_files:
            frame_id = frame_file.split('.png')[0]
            # 跳过首尾帧（因为光流需要前后帧）
            if int(frame_id) == 0 or int(frame_id) == len(frame_files) - 1:
                continue

            # 路径
            fwd_flo = os.path.join(customPath, "raw_flows_gap1", videoId, frame_id + ".flo")
            bwd_flo = os.path.join(customPath, "raw_flows_gap-1", videoId, frame_id + ".flo")
            pred_mask = os.path.join(pred_dir, frame_id + ".png")
            gt_mask = os.path.join(gt_dir, frame_id + ".png")
            output_mask = os.path.join(out_root, output_subdir, "direction_filter_refine", videoId, frame_id + ".png")

            # 检查文件存在性
            if not all(os.path.exists(p) for p in [fwd_flo, bwd_flo, pred_mask, gt_mask]):
                continue

            # 读取数据
            flow_f = read_flo(fwd_flo)
            pred = read_mask(pred_mask, normalize=True)   # [0,1]
            gt_raw = read_mask(gt_mask, normalize=False) # 0-255

            # 构建 GT 二值掩码（血管/导管区域，排除白色导管和黑色背景）
            gt_gray = (gt_raw > black_thresh) & (gt_raw < white_thresh)
            valid = ~(gt_raw >= white_thresh)  # 有效区域（忽略白色导管）

            # 原始预测二值化
            pred_bin = pred > 0.5

            # 应用方向滤波
            refined_bin = filter_by_direction(flow_f, pred_bin, angle_threshold)

            # 保存结果（可选）
            os.makedirs(os.path.dirname(output_mask), exist_ok=True)
            Image.fromarray((refined_bin.astype(np.uint8) * 255)).save(output_mask)

            # 计算指标
            metrics_orig = compute_metrics(pred_bin, gt_gray, valid)
            metrics_refined = compute_metrics(refined_bin, gt_gray, valid)

            # 聚合统计数据
            total_tp_orig += metrics_orig['TP']
            total_fp_orig += metrics_orig['FP']
            total_fn_orig += metrics_orig['FN']
            total_tp_refined += metrics_refined['TP']
            total_fp_refined += metrics_refined['FP']
            total_fn_refined += metrics_refined['FN']

            # 统计去除的区域中不属于GT的比例
            removed_pixels = np.sum(pred_bin & ~refined_bin)           # 被去除的像素
            removed_not_gt = np.sum(pred_bin & ~refined_bin & ~gt_gray) # 被去除且不属于GT的像素
            total_removed_pixels += removed_pixels
            total_removed_not_gt += removed_not_gt

            # 记录查准率变化
            delta_precision = metrics_refined['Precision'] - metrics_orig['Precision']
            delta_precision_list.append((frame_id, delta_precision, metrics_orig['Precision'], metrics_refined['Precision']))

            total_frames += 1
            if total_frames % 100 == 0:
                print(f"\r已处理 {total_frames} 帧", end='')

    print(f"\n处理完成，共处理 {total_frames} 帧。")

    # ==================== 输出分析结果 ====================
    print("\n" + "="*60)
    print("【分析结果】")

    # 1. 去除的运动异常区域中不为血管/导管的比例
    if total_removed_pixels > 0:
        ratio_not_gt = total_removed_not_gt / total_removed_pixels
        print(f"\n1. 去除区域中不属于血管/导管的比例: {ratio_not_gt:.4f} "
              f"({total_removed_not_gt} / {total_removed_pixels})")
    else:
        print("\n1. 没有像素被去除。")

    # 2. 整体查准率、查全率变化
    def precision(tp, fp):
        return tp / (tp + fp + 1e-8)
    def recall(tp, fn):
        return tp / (tp + fn + 1e-8)

    pre_orig = precision(total_tp_orig, total_fp_orig)
    pre_refined = precision(total_tp_refined, total_fp_refined)
    rec_orig = recall(total_tp_orig, total_fn_orig)
    rec_refined = recall(total_tp_refined, total_fn_refined)

    print("\n2. 整体指标变化（全数据集聚合）")
    print(f"   原始预测: Precision={pre_orig:.4f}, Recall={rec_orig:.4f}")
    print(f"   方向滤波后: Precision={pre_refined:.4f}, Recall={rec_refined:.4f}")
    print(f"   查准率变化: {pre_refined - pre_orig:+.4f}")
    print(f"   查全率变化: {rec_refined - rec_orig:+.4f}")

    # 3. 找出查准率提升最大和降低最多的图
    if delta_precision_list:
        # 按查准率变化排序
        sorted_list = sorted(delta_precision_list, key=lambda x: x[1])
        worst = sorted_list[0]      # 降低最多
        best = sorted_list[-1]      # 提升最大
        print("\n3. 单帧查准率变化极值")
        print(f"   查准率提升最大的帧: {best[0]}, ΔPrecision={best[1]:+.4f} "
              f"(原始={best[2]:.4f} → 修正={best[3]:.4f})")
        print(f"   查准率降低最多的帧: {worst[0]}, ΔPrecision={worst[1]:+.4f} "
              f"(原始={worst[2]:.4f} → 修正={worst[3]:.4f})")
    print("="*60)

    return {
        'ratio_removed_not_gt': ratio_not_gt if total_removed_pixels>0 else None,
        'precision_orig': pre_orig,
        'precision_refined': pre_refined,
        'recall_orig': rec_orig,
        'recall_refined': rec_refined,
        'best_frame': best if delta_precision_list else None,
        'worst_frame': worst if delta_precision_list else None
    }

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 配置参数
    tag = "A26-03"                # 预测掩码标签
    angle_threshold = 30         # 方向差异阈值（度）
    # 可选：指定处理部分视频，None 处理全部
    # video_list = [("CVAI-1207", "CVAI-1207RAO2_CAU30")]  # 示例
    video_list = None

    # 运行处理和分析
    process_video_with_direction_filter(tag, video_list, angle_threshold)