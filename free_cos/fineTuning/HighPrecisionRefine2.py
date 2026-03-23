import os
import numpy as np
from PIL import Image
import yaml

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def compute_metrics_from_files(refined_mask_path, gt_path, black_thresh=50, white_thresh=200):
    """
    根据修正后的掩码和GT计算评估指标（排除白色导管区域）。

    Args:
        refined_mask_path: 修正后的二值掩码路径（0或255）
        gt_path: GT图像路径（灰度图，0-255）
        black_thresh: 背景阈值（≤此值视为背景）
        white_thresh: 白色导管阈值（≥此值视为忽略区域）

    Returns:
        dict: 包含 TP, FP, FN, Precision, Recall, Dice 的字典；若文件缺失返回None
    """
    if not os.path.exists(refined_mask_path) or not os.path.exists(gt_path):
        return None

    # 读取修正掩码（已二值化，0或255）
    refined_img = Image.open(refined_mask_path).convert('L')
    refined = np.array(refined_img, dtype=np.uint8)
    refined_bin = (refined > 128)  # 确保二值化

    # 读取GT
    gt_img = Image.open(gt_path).convert('L')
    gt = np.array(gt_img, dtype=np.uint8)

    # 确保尺寸一致
    if refined_bin.shape != gt.shape:
        print(f"警告：尺寸不一致 {refined_mask_path} vs {gt_path}，跳过")
        return None

    # 根据阈值划分GT区域
    gt_black = gt <= black_thresh          # 背景
    gt_gray = (gt > black_thresh) & (gt < white_thresh)  # 目标导管
    gt_white = gt >= white_thresh           # 白色导管（忽略）
    valid = ~gt_white                       # 有效评估区域

    # 计算TP, FP, FN（仅在有效区域内）
    tp = np.sum(refined_bin & gt_gray & valid)
    fp = np.sum(refined_bin & ~gt_gray & valid)
    fn = np.sum(~refined_bin & gt_gray & valid)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    dice = 2 * tp / (2*tp + fp + fn + 1e-8)

    return {
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'Precision': float(precision),
        'Recall': float(recall),
        'Dice': float(dice)
    }

def collect_all_frame_metrics_no_reprocess(tag, config_path):
    """
    利用已存在的修正掩码和GT，遍历所有帧收集指标。
    """
    config = load_config(config_path)
    datasetPath = config["my"]["datasetPath_rigid.in"]
    customPath = config["my"]["datasetPath_rigid.in_custom"]  # 可能不需要，但保留
    out_root = config["my"]["datasetPath_rigid.out"]

    # 修正掩码根目录（与原脚本保存路径一致）
    refine_root = os.path.join(
        config["my"]["filePathRoot"],
        config["my"]["subPath"]["outputs"],
        "high_precision_refine"
    )

    results = []

    # 统计总帧数（用于进度）
    total_frames = 0
    for userId in os.listdir(datasetPath):
        user_img_dir = os.path.join(datasetPath, userId, "images")
        if not os.path.isdir(user_img_dir):
            continue
        for videoId in os.listdir(user_img_dir):
            img_dir = os.path.join(user_img_dir, videoId)
            frames = [f for f in os.listdir(img_dir) if f.endswith('.png')]
            total_frames += max(0, len(frames) - 2)  # 排除首尾帧

    processed = 0
    for userId in os.listdir(datasetPath):
        user_img_dir = os.path.join(datasetPath, userId, "images")
        user_gt_dir = os.path.join(datasetPath, userId, "ground_truth")
        if not os.path.isdir(user_img_dir) or not os.path.isdir(user_gt_dir):
            continue
        for videoId in os.listdir(user_img_dir):
            img_dir = os.path.join(user_img_dir, videoId)
            frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
            # 跳过首尾帧（与原脚本一致）
            for frame_file in frames[1:-1]:  # 排除第一帧和最后一帧
                frameId = frame_file.replace('.png', '')
                processed += 1
                print(f"\r正在处理: {processed}/{total_frames}", end="")

                # 构造修正掩码路径
                refined_mask_path = os.path.join(refine_root, videoId, frameId + ".png")
                # 构造GT路径
                gt_mask_path = os.path.join(datasetPath, userId, "ground_truth", videoId + "CATH", frameId + ".png")

                if not os.path.exists(refined_mask_path):
                    continue
                if not os.path.exists(gt_mask_path):
                    continue

                metrics = compute_metrics_from_files(refined_mask_path, gt_mask_path,
                                                     black_thresh=50, white_thresh=200)
                if metrics is not None:
                    results.append({
                        'user_id': userId,
                        'video_id': videoId,
                        'frame_id': frameId,
                        'precision': metrics['Precision'],
                        'recall': metrics['Recall'],
                        'dice': metrics['Dice'],
                        'tp': metrics['TP'],
                        'fp': metrics['FP'],
                        'fn': metrics['FN'],
                        'refined_path': refined_mask_path
                    })

    print(f"\n收集完成，共处理 {processed} 帧，成功获取 {len(results)} 帧的指标。")
    return results

def find_worst_k_cases_no_reprocess(tag, K, config_path):
    """
    找出修正后查准率最低的前K个案例（不重复后处理）。
    """
    all_metrics = collect_all_frame_metrics_no_reprocess(tag, config_path)

    if not all_metrics:
        print("未收集到任何有效指标，请检查修正掩码和GT是否存在。")
        return

    # 按查准率升序排序
    sorted_cases = sorted(all_metrics, key=lambda x: x['precision'])

    print("\n\n" + "=" * 70)
    print(f"修正后查准率最低的前 {K} 个案例")
    print("=" * 70)

    for i, case in enumerate(sorted_cases[:K]):
        print(f"\n【第 {i+1} 名】 查准率 = {case['precision']:.4f}")
        print(f"  用户ID: {case['user_id']}")
        print(f"  视频ID: {case['video_id']}")
        print(f"  帧号: {case['frame_id']}")
        print(f"  召回率: {case['recall']:.4f}")
        print(f"  Dice: {case['dice']:.4f}")
        print(f"  TP: {case['tp']}, FP: {case['fp']}, FN: {case['fn']}")
        print(f"  修正掩码路径: {case['refined_path']}")

    # 保存报告
    output_report = os.path.join(os.path.dirname(config_path), "worst_k_cases.txt")
    with open(output_report, 'w') as f:
        f.write(f"修正后查准率最低的前 {K} 个案例\n")
        f.write("=" * 50 + "\n")
        for i, case in enumerate(sorted_cases[:K]):
            f.write(f"\n【{i+1}】 Precision={case['precision']:.4f}\n")
            f.write(f"  {case['user_id']}/{case['video_id']}/frame_{case['frame_id']}\n")
            f.write(f"  Recall={case['recall']:.4f}, Dice={case['dice']:.4f}\n")
            f.write(f"  TP={case['tp']}, FP={case['fp']}, FN={case['fn']}\n")
            f.write(f"  path: {case['refined_path']}\n")
    print(f"\n报告已保存至: {output_report}")

if __name__ == "__main__":
    print("寻找topK个负面案例")
    # 配置参数
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../..", 'confs/newConfig.yaml')
    tag = "A26-03"   # 与之前一致，仅用于路径构造（实际上修正掩码路径不依赖tag？原脚本中output_path没有用tag，但tag用于原始预测路径，这里不需要）
    K = 10

    find_worst_k_cases_no_reprocess(tag, K, config_path)