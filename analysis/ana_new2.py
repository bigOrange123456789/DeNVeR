
import os
import json
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

def load_and_binarize_image(image_path, threshold=0.5):
    """加载图像并二值化"""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # 如果图像是RGB，转换为灰度
        if len(img_array.shape) == 3:
            img_array = np.mean(img_array, axis=2)
        
        # 二值化
        if not threshold is None:  # 阈值非空时进行二值化
            img_array = (img_array > 256*threshold).astype(np.uint8)
        else:
            img_array = img_array.astype(np.float32)/256
        img_array = img_array.astype(np.float32)  # /256
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def calculate_metrics(gt, pred):
    """计算Dice系数、查全率、查准率、IoU、准确率、特异性"""
    # 展平数组
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    
    # 计算TP, FP, FN, TN
    tp = np.sum((gt_flat == 1) & (pred_flat == 1))
    fp = np.sum((gt_flat == 0) & (pred_flat == 1))
    fn = np.sum((gt_flat == 1) & (pred_flat == 0))
    tn = np.sum((gt_flat == 0) & (pred_flat == 0))
    
    # 计算指标
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0   # 新增特异性
    
    return dice, recall, precision, iou, acc, specificity

import pandas as pd

def getNameList():
    # 读取 Excel 文件
    df = pd.read_excel('./outputs/namelist.xlsx', sheet_name='文件夹列表')
    # 构建字典便于快速检索
    video_dict = df.set_index('videoId')[['long', 'short']].to_dict('index')
    return video_dict

video_dict = getNameList()

def get_video_info(video_id):
    """
    根据 videoId 返回对应的 long 和 short 值
    返回格式：{'long': 值, 'short': 值} 或 None（如果未找到）
    """
    return video_dict.get(str(video_id))

def process_test_config(config, usedVideoId, DataFiltering=None):
    """处理单个测试配置"""
    name = config["name"]
    gt_path = config["gt_path"]
    pred_path = config["pred_path"]
    block_cath = config["block_cath"]  # 是否遮挡导管
    if block_cath:
        print(config)
        cath_path = config["cath_path"]  # 需要遮挡
    
    print(f"Processing {name}...")
    print(f"GT path: {gt_path}")
    print(f"Pred path: {pred_path}")
    if block_cath:
        print(f"cath_path: {cath_path}")
    
    # 收集所有图像的指标
    all_dice = []
    all_recall = []
    all_precision = []
    all_iou = []
    all_acc = []
    all_specificity = []   # 新增特异性列表
    
    for videoId in os.listdir(gt_path):
        if usedVideoId is None or (videoId in usedVideoId):
            long0 = get_video_info(videoId)["long"]
            short0 = get_video_info(videoId)["short"]
            for frameId in os.listdir(os.path.join(gt_path, videoId)):
                flag = True  # 使用全部数据
                if DataFiltering in ["T", "Move", "FrontBack"]:
                    flag = (long0 == DataFiltering)
                if flag:
                    filename = frameId
                    gt_img = load_and_binarize_image(os.path.join(gt_path, videoId, frameId), 0.5)
                    pred_img = load_and_binarize_image(os.path.join(pred_path, videoId, frameId), config["threshold"])
                    if block_cath:
                        cath_img = load_and_binarize_image(os.path.join(cath_path, videoId + "CATH", frameId), None)
                        mask_cath = np.zeros_like(cath_img)
                        mask_vessel = np.zeros_like(cath_img)
                        mask_cath[cath_img > 0.75] = 1
                        mask_vessel[(cath_img > 0.25) & (cath_img < 0.75)] = 1
                        gt_img = mask_vessel
                        pred_img = pred_img * (1 - mask_cath)
                    
                    if gt_img is not None and pred_img is not None:
                        # 确保图像尺寸相同
                        if gt_img.shape != pred_img.shape:
                            print(f"Warning: Image shape mismatch for {filename}, resizing pred to match gt")
                            pred_img = Image.fromarray(pred_img).resize((gt_img.shape[1], gt_img.shape[0]), Image.NEAREST)
                            pred_img = np.array(pred_img)
                        
                        # 现在calculate_metrics返回6个值
                        dice, recall, precision, iou, acc, specificity = calculate_metrics(gt_img, pred_img)
                        all_dice.append(dice)
                        all_recall.append(recall)
                        all_precision.append(precision)
                        all_iou.append(iou)
                        all_acc.append(acc)
                        all_specificity.append(specificity)
    
    # 计算均值和标准差
    if len(all_dice) > 0:
        avg_metrics = {
            "dice": {"mean": np.mean(all_dice), "std": np.std(all_dice)},
            "recall": {"mean": np.mean(all_recall), "std": np.std(all_recall)},
            "precision": {"mean": np.mean(all_precision), "std": np.std(all_precision)},
            "iou": {"mean": np.mean(all_iou), "std": np.std(all_iou)},
            "acc": {"mean": np.mean(all_acc), "std": np.std(all_acc)},
            "specificity": {"mean": np.mean(all_specificity), "std": np.std(all_specificity)}  # 新增特异性
        }
        print(f"Average metrics for {name}: Dice={avg_metrics['dice']['mean']:.4f}±{avg_metrics['dice']['std']:.4f}, "
              f"Recall={avg_metrics['recall']['mean']:.4f}±{avg_metrics['recall']['std']:.4f}, "
              f"Precision={avg_metrics['precision']['mean']:.4f}±{avg_metrics['precision']['std']:.4f}, "
              f"IoU={avg_metrics['iou']['mean']:.4f}±{avg_metrics['iou']['std']:.4f}, "
              f"Acc={avg_metrics['acc']['mean']:.4f}±{avg_metrics['acc']['std']:.4f}, "
              f"Specificity={avg_metrics['specificity']['mean']:.4f}±{avg_metrics['specificity']['std']:.4f}")
    else:
        print(f"Warning: No valid image pairs found for {name}")
        avg_metrics = {
            "dice": {"mean": 0, "std": 0},
            "recall": {"mean": 0, "std": 0},
            "precision": {"mean": 0, "std": 0},
            "iou": {"mean": 0, "std": 0},
            "acc": {"mean": 0, "std": 0},
            "specificity": {"mean": 0, "std": 0}
        }
    
    return name, avg_metrics

def create_bar_chart(results, colors, DataFiltering=None):
    """创建按指标分组的柱状图，并添加误差棒（标准差）"""
    # 现在包含6个指标
    metrics = ['Dice', 'Recall', 'Precision', 'IoU', 'Accuracy', 'Specificity']
    test_names = list(results.keys())
    test_names2 = ["tag:" + name for name in test_names]
    
    # 设置柱状图位置
    x = np.arange(len(metrics))
    width = 0.8 / len(test_names)  # 根据测试配置数量调整宽度
    
    # 创建图形（适当调大宽度以容纳更多柱子）
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 为每个测试配置绘制柱状图
    for i, name in enumerate(test_names):
        offset = (i - len(test_names)/2 + 0.5) * width
        means = [
            results[name]['dice']['mean'],
            results[name]['recall']['mean'],
            results[name]['precision']['mean'],
            results[name]['iou']['mean'],
            results[name]['acc']['mean'],
            results[name]['specificity']['mean']   # 新增特异性均值
        ]
        stds = [
            results[name]['dice']['std'],
            results[name]['recall']['std'],
            results[name]['precision']['std'],
            results[name]['iou']['std'],
            results[name]['acc']['std'],
            results[name]['specificity']['std']    # 新增特异性标准差
        ]
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                      label=test_names2[i], color=colors[name], ecolor='black')
        
        # 添加数值标签（均值）
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 设置图形属性
    ax.set_title('Segmentation Performance Metrics Comparison (DataFiltering=' + str(DataFiltering) + ")")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)  # 显示指标名称
    ax.legend()  # 显示每个颜色的含义
    ax.grid(True, alpha=0.3)
    
    # 设置y轴范围
    ax.set_ylim(0, 1.1)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像（可选）
    if False:
        plt.savefig('segmentation_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

from analysis.json0 import config_data

def main():
    # 处理每个测试配置
    results = {}
    colors = {}
    
    for config in config_data["experiments"]:
        name, metrics = process_test_config(config,
                                            config_data["usedVideoId"],
                                            DataFiltering=config_data["DataFiltering"])
        results[name] = metrics
        colors[name] = config["color"]
        print("name:", name)
        print("-" * 50)
    
    # 打印汇总结果（包含均值和标准差）
    print("\n" + "=" * 60)
    print("SUMMARY RESULTS (mean ± std)")
    print("=" * 60)
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Dice:        {metrics['dice']['mean']:.4f} ± {metrics['dice']['std']:.4f}")
        print(f"  Recall:      {metrics['recall']['mean']:.4f} ± {metrics['recall']['std']:.4f}")
        print(f"  Precision:   {metrics['precision']['mean']:.4f} ± {metrics['precision']['std']:.4f}")
        print(f"  IoU:         {metrics['iou']['mean']:.4f} ± {metrics['iou']['std']:.4f}")
        print(f"  Accuracy:    {metrics['acc']['mean']:.4f} ± {metrics['acc']['std']:.4f}")
        print(f"  Specificity: {metrics['specificity']['mean']:.4f} ± {metrics['specificity']['std']:.4f}")
        print()
    
    # 生成柱状图（含误差棒）
    create_bar_chart(results, colors, DataFiltering=config_data["DataFiltering"])
    print("Bar chart with error bars (std) displayed.")

if __name__ == "__main__":
    main()
