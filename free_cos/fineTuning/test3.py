import os
import struct
import numpy as np
from PIL import Image
import yaml

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
    norm1 = np.linalg.norm(v1) + 1e-8
    norm2 = np.linalg.norm(v2) + 1e-8
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.arccos(cos_theta)

def analyze_direction_offset(tag, video_list=None, black_thresh=50, white_thresh=200):
    """
    分析整个数据集中正确/错误区域的运动方向与平均方向的偏移角度
    """
    script_path = os.path.abspath(__file__)
    ROOT1 = os.path.dirname(script_path)
    config_path = os.path.join(ROOT1, "../..", 'confs/newConfig.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    datasetPath = config["my"]["datasetPath_rigid.in"]
    customPath = config["my"]["datasetPath_rigid.in_custom"]

    # 全局统计
    all_tp_angles = []   # 所有正确像素的偏移角度
    all_fp_angles = []   # 所有错误像素的偏移角度

    # 每帧的统计（可选）
    frame_stats = []   # 存储 (frame_id, tp_min, tp_max, fp_min, fp_max)

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
            print(f"警告：GT 目录 {gt_dir} 不存在，跳过评估")
            continue

        pred_dir = os.path.join(config["my"]["datasetPath_rigid.out"],
                                userId, "decouple", videoId, tag + ".orig_mask")

        frame_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        for frame_file in frame_files:
            frame_id = frame_file.split('.png')[0]
            # 跳过首尾帧（光流需要前后帧）
            if int(frame_id) == 0 or int(frame_id) == len(frame_files) - 1:
                continue

            fwd_flo = os.path.join(customPath, "raw_flows_gap1", videoId, frame_id + ".flo")
            pred_mask = os.path.join(pred_dir, frame_id + ".png")
            gt_mask = os.path.join(gt_dir, frame_id + ".png")

            if not all(os.path.exists(p) for p in [fwd_flo, pred_mask, gt_mask]):
                continue

            # 读取数据
            flow = read_flo(fwd_flo)
            pred = read_mask(pred_mask, normalize=True)  # [0,1]
            gt_raw = read_mask(gt_mask, normalize=False) # 0-255

            # 构建 GT 二值掩码（血管/导管区域，排除白色导管和黑色背景）
            gt_gray = (gt_raw > black_thresh) & (gt_raw < white_thresh)
            valid = ~(gt_raw >= white_thresh)   # 有效区域（忽略白色导管）

            # 原始预测二值化
            pred_bin = pred > 0.5

            # 有效预测前景（排除白色导管区域）
            valid_foreground = pred_bin & valid
            if np.sum(valid_foreground) < 2:
                # 前景像素太少，跳过此帧
                continue

            # 计算整图平均方向
            vectors_all = flow[valid_foreground]
            mean_vec = np.mean(vectors_all, axis=0)

            # 正确区域（TP）
            tp_mask = pred_bin & gt_gray & valid
            if np.sum(tp_mask) > 0:
                tp_vectors = flow[tp_mask]
                tp_angles = []
                for vec in tp_vectors:
                    ang = angle_between(vec, mean_vec)
                    tp_angles.append(np.degrees(ang))
                all_tp_angles.extend(tp_angles)
                tp_min = np.min(tp_angles)
                tp_max = np.max(tp_angles)
            else:
                tp_min = tp_max = None

            # 错误区域（FP）
            fp_mask = pred_bin & (~gt_gray) & valid
            if np.sum(fp_mask) > 0:
                fp_vectors = flow[fp_mask]
                fp_angles = []
                for vec in fp_vectors:
                    ang = angle_between(vec, mean_vec)
                    fp_angles.append(np.degrees(ang))
                all_fp_angles.extend(fp_angles)
                fp_min = np.min(fp_angles)
                fp_max = np.max(fp_angles)
            else:
                fp_min = fp_max = None

            # 记录每帧统计（可选）
            frame_stats.append({
                'frame': f"{userId}/{videoId}/{frame_id}",
                'tp_min': tp_min,
                'tp_max': tp_max,
                'fp_min': fp_min,
                'fp_max': fp_max
            })

            total_frames += 1
            if True:#total_frames % 100 == 0:
                print(f"\r已处理 {total_frames} 帧", end='')

    print(f"\n处理完成，共处理 {total_frames} 帧。")

    # 计算全局统计
    print("\n" + "="*60)
    print("【全局方向偏移统计】")

    if len(all_tp_angles) > 0:
        tp_arr = np.array(all_tp_angles)
        print("\n正确区域（TP）:")
        print(f"  总像素数: {len(tp_arr)}")
        print(f"  偏移角度范围: {np.min(tp_arr):.2f}° ~ {np.max(tp_arr):.2f}°")
        print(f"  平均值 ± 标准差: {np.mean(tp_arr):.2f}° ± {np.std(tp_arr):.2f}°")
        print(f"  中位数: {np.median(tp_arr):.2f}°")
        print(f"  分位数 (25%, 75%): {np.percentile(tp_arr, 25):.2f}°, {np.percentile(tp_arr, 75):.2f}°")
    else:
        print("\n正确区域（TP）: 无有效像素")

    if len(all_fp_angles) > 0:
        fp_arr = np.array(all_fp_angles)
        print("\n错误区域（FP）:")
        print(f"  总像素数: {len(fp_arr)}")
        print(f"  偏移角度范围: {np.min(fp_arr):.2f}° ~ {np.max(fp_arr):.2f}°")
        print(f"  平均值 ± 标准差: {np.mean(fp_arr):.2f}° ± {np.std(fp_arr):.2f}°")
        print(f"  中位数: {np.median(fp_arr):.2f}°")
        print(f"  分位数 (25%, 75%): {np.percentile(fp_arr, 25):.2f}°, {np.percentile(fp_arr, 75):.2f}°")
    else:
        print("\n错误区域（FP）: 无有效像素")

    # 汇总每帧的偏移范围
    print("\n【每帧偏移范围汇总】")
    # 提取所有帧的 tp_min 和 tp_max（忽略None）
    valid_tp_min = [s['tp_min'] for s in frame_stats if s['tp_min'] is not None]
    valid_tp_max = [s['tp_max'] for s in frame_stats if s['tp_max'] is not None]
    valid_fp_min = [s['fp_min'] for s in frame_stats if s['fp_min'] is not None]
    valid_fp_max = [s['fp_max'] for s in frame_stats if s['fp_max'] is not None]

    if valid_tp_min:
        print(f"  所有帧的 TP 最小偏移角度: 最小={np.min(valid_tp_min):.2f}°, 最大={np.max(valid_tp_min):.2f}°")
        print(f"  所有帧的 TP 最大偏移角度: 最小={np.min(valid_tp_max):.2f}°, 最大={np.max(valid_tp_max):.2f}°")
    if valid_fp_min:
        print(f"  所有帧的 FP 最小偏移角度: 最小={np.min(valid_fp_min):.2f}°, 最大={np.max(valid_fp_min):.2f}°")
        print(f"  所有帧的 FP 最大偏移角度: 最小={np.min(valid_fp_max):.2f}°, 最大={np.max(valid_fp_max):.2f}°")

    # 建议阈值
    print("\n【阈值建议】")
    if len(all_tp_angles) > 0 and len(all_fp_angles) > 0:
        tp_max_global = np.max(tp_arr)
        fp_min_global = np.min(fp_arr)
        if tp_max_global < fp_min_global:
            print(f"  理想分离阈值: 在 {tp_max_global:.2f}° 和 {fp_min_global:.2f}° 之间选择，例如 {(tp_max_global + fp_min_global)/2:.2f}°")
            print(f"  若阈值设为 {tp_max_global:.2f}°，则几乎所有正确区域被保留，但部分错误区域也可能被保留。")
            print(f"  若阈值设为 {fp_min_global:.2f}°，则能完全剔除错误区域，但可能丢失部分正确区域。")
        else:
            print(f"  正确区域最大偏移角度: {tp_max_global:.2f}°, 错误区域最小偏移角度: {fp_min_global:.2f}°")
            print(f"  两者有重叠区间，无法完全分离。建议阈值设置在 {tp_max_global:.2f}° ~ {fp_min_global:.2f}° 之间，根据实际需求权衡查准率和查全率。")
            # 也可以考虑分位数
            tp_75 = np.percentile(tp_arr, 75)
            fp_25 = np.percentile(fp_arr, 25)
            print(f"  例如，可考虑 TP 的 75% 分位数 {tp_75:.2f}° 或 FP 的 25% 分位数 {fp_25:.2f}°。")
    else:
        print("  数据不足，无法给出建议。")

    print("="*60)

    return {
        'tp_angles': all_tp_angles,
        'fp_angles': all_fp_angles,
        'frame_stats': frame_stats
    }

if __name__ == "__main__":
    tag = "A26-03"
    # 示例：只处理部分视频
    # video_list = [("CVAI-1207", "CVAI-1207RAO2_CAU30")]
    video_list = None  # 处理全部
    analyze_direction_offset(tag, video_list)