import os
import json
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def load_and_binarize_image(image_path, threshold=0.5):
    """加载图像并二值化"""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # 如果图像是RGB，转换为灰度
        if len(img_array.shape) == 3:
            img_array = np.mean(img_array, axis=2)
        
        # 二值化
        if not threshold is None:#阈值非空的时候进行二值化
            img_array = (img_array > 256*threshold).astype(np.uint8)
        else:
            img_array = img_array.astype(np.float32)/256
        img_array = img_array.astype(np.float32)#/256
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def calculate_metrics(gt, pred, pred_prob=None):
    """计算Dice系数、查全率、查准率"""
    # 展平数组
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    
    # 计算TP, FP, FN
    tp = np.sum((gt_flat == 1) & (pred_flat == 1))
    fp = np.sum((gt_flat == 0) & (pred_flat == 1))
    fn = np.sum((gt_flat == 1) & (pred_flat == 0))
    tn = np.sum((gt_flat == 0) & (pred_flat == 0))
    
    # 计算基础指标
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # 计算AUC, AP, maxDice（如果有概率预测）
    auc_score = 0
    ap_score = 0
    max_dice = dice
    
    if pred_prob is not None:
        pred_prob_flat = pred_prob.flatten()
        gt_flat_binary = (gt_flat > 0.5).astype(int)
        
        # 计算AUC
        if len(np.unique(gt_flat_binary)) > 1:
            fpr, tpr, _ = roc_curve(gt_flat_binary, pred_prob_flat)
            auc_score = auc(fpr, tpr)
        else:
            auc_score = 0
            
        # 计算AP
        ap_score = average_precision_score(gt_flat_binary, pred_prob_flat)
        
        # 计算maxDice
        thresholds = np.linspace(0, 1, 101)
        dice_scores = []
        
        for thresh in thresholds:
            pred_thresh = (pred_prob_flat > thresh).astype(int)
            tp_t = np.sum((gt_flat_binary == 1) & (pred_thresh == 1))
            fp_t = np.sum((gt_flat_binary == 0) & (pred_thresh == 1))
            fn_t = np.sum((gt_flat_binary == 1) & (pred_thresh == 0))
            dice_t = (2 * tp_t) / (2 * tp_t + fp_t + fn_t) if (2 * tp_t + fp_t + fn_t) > 0 else 0
            dice_scores.append(dice_t)
        
        max_dice = np.max(dice_scores)
    
    return dice, recall, precision, auc_score, ap_score, max_dice

def load_prediction_probability(pred_path, videoId, frameId):
    """加载预测概率图（未二值化的原始预测）"""
    try:
        pred_img = load_and_binarize_image(os.path.join(pred_path, videoId, frameId), None)
        return pred_img
    except:
        return None

def process_test_config(config):
    """处理单个测试配置"""
    name = config["name"]
    gt_path = config["gt_path"]
    pred_path = config["pred_path"]
    block_cath = config["block_cath"]#是否遮挡导管
    if block_cath: 
        print(config)
        cath_path = config["cath_path"]#需要遮挡
    
    print(f"Processing {name}...")
    print(f"GT path: {gt_path}")
    # exit(0)
    print(f"Pred path: {pred_path}")
    if block_cath: print(f"cath_path: {cath_path}")
    
    # 计算所有图像的指标
    all_dice = []
    all_recall = []
    all_precision = []
    all_auc = []
    all_ap = []
    all_max_dice = []
    
    # 用于存储每个图片的详细结果
    image_results = []
    
    # 用于绘制曲线的数据
    all_gt_probs = []
    all_pred_probs = []
    
    for videoId in os.listdir(gt_path):
        for frameId in os.listdir(gt_path+"/"+videoId):
            filename = frameId
            gt_img = load_and_binarize_image(os.path.join(gt_path, videoId, frameId), 0.5)
            pred_img = load_and_binarize_image(os.path.join(pred_path, videoId, frameId), config["threshold"])
            pred_prob = load_prediction_probability(pred_path, videoId, frameId)
            
            if block_cath: 
                # print(cath_path, videoId+"CATH", frameId)
                cath_img = load_and_binarize_image(os.path.join(cath_path, videoId+"CATH", frameId), None)
                # print("cath_img",cath_img)
                mask_cath = np.zeros_like(cath_img)
                mask_vessel = np.zeros_like(cath_img)
                mask_cath[cath_img>0.75]=1
                mask_vessel[(cath_img>0.25) & (cath_img<0.75)]=1 
                gt_img = mask_vessel
                pred_img = pred_img * ( 1 - mask_cath )
                if pred_prob is not None:
                    pred_prob = pred_prob * ( 1 - mask_cath )

            if gt_img is not None and pred_img is not None:
                # 确保图像尺寸相同
                if gt_img.shape != pred_img.shape:
                    print(f"Warning: Image shape mismatch for {filename}, resizing pred to match gt")
                    pred_img = Image.fromarray(pred_img).resize((gt_img.shape[1], gt_img.shape[0]), Image.NEAREST)
                    pred_img = np.array(pred_img)
                    if pred_prob is not None:
                        pred_prob = Image.fromarray(pred_prob).resize((gt_img.shape[1], gt_img.shape[0]), Image.NEAREST)
                        pred_prob = np.array(pred_prob)
                
                # 计算指标
                dice, recall, precision, auc_score, ap_score, max_dice = calculate_metrics(gt_img, pred_img, pred_prob)
                
                all_dice.append(dice)
                all_recall.append(recall)
                all_precision.append(precision)
                all_auc.append(auc_score)
                all_ap.append(ap_score)
                all_max_dice.append(max_dice)
                
                # 保存单个图片结果
                image_results.append({
                    'videoId': videoId,
                    'frameId': frameId,
                    'dice': dice,
                    'recall': recall,
                    'precision': precision,
                    'auc': auc_score,
                    'ap': ap_score,
                    'maxDice': max_dice
                })
                
                # 收集用于绘制曲线的数据
                if pred_prob is not None:
                    all_gt_probs.extend(gt_img.flatten())
                    all_pred_probs.extend(pred_prob.flatten())
    
    # 计算平均指标
    if len(all_dice) > 0:
        avg_metrics = {
            "dice": np.mean(all_dice),
            "recall": np.mean(all_recall),
            "precision": np.mean(all_precision),
            "auc": np.mean(all_auc),
            "ap": np.mean(all_ap),
            "maxDice": np.mean(all_max_dice)
        }
        print(f"Average metrics for {name}: "
              f"Dice={avg_metrics['dice']:.4f}, "
              f"Recall={avg_metrics['recall']:.4f}, "
              f"Precision={avg_metrics['precision']:.4f}, "
              f"AUC={avg_metrics['auc']:.4f}, "
              f"AP={avg_metrics['ap']:.4f}, "
              f"maxDice={avg_metrics['maxDice']:.4f}")
    else:
        print(f"Warning: No valid image pairs found for {name}")
        avg_metrics = {
            "dice": 0, "recall": 0, "precision": 0, 
            "auc": 0, "ap": 0, "maxDice": 0
        }
    
    return name, avg_metrics, image_results, (all_gt_probs, all_pred_probs)

def create_bar_chart(results, colors):
    """创建按指标分组的柱状图"""
    metrics = ['Dice', 'Recall', 'Precision', 'AUC', 'AP', 'maxDice']
    test_names = list(results.keys())
    test_names2 = ["tag:" + name for name in test_names]
    
    # 设置柱状图位置
    x = np.arange(len(metrics))
    width = 0.8 / len(test_names)  # 根据测试配置数量调整宽度
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 为每个测试配置绘制柱状图
    for i, name in enumerate(test_names):
        offset = (i - len(test_names)/2 + 0.5) * width
        values = [
            results[name]['dice'], 
            results[name]['recall'], 
            results[name]['precision'],
            results[name]['auc'],
            results[name]['ap'],
            results[name]['maxDice']
        ]
        bars = ax.bar(x + offset, values, width, label=test_names2[i], color=colors[name])
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 设置图形属性
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Segmentation Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 设置y轴范围
    ax.set_ylim(0, 1.1)
    
    # 调整布局
    plt.tight_layout()
    plt.show()

def plot_roc_curves(roc_data, colors):
    """绘制ROC曲线"""
    plt.figure(figsize=(10, 8))
    
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, color=colors[name], lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_pr_curves(pr_data, colors):
    """绘制Precision-Recall曲线"""
    plt.figure(figsize=(10, 8))
    
    for name, (precision, recall, ap) in pr_data.items():
        plt.plot(recall, precision, color=colors[name], lw=2,
                label=f'{name} (AP = {ap:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def save_to_excel(image_results, experiment_name):
    """将结果保存到Excel文件"""
    # 创建DataFrame
    df = pd.DataFrame(image_results)
    
    # 确保列的顺序正确
    columns = ['videoId', 'frameId', 'dice', 'recall', 'precision', 'auc', 'ap', 'maxDice']
    df = df[columns]
    
    # 保存到Excel
    filename = f"./outputs/{experiment_name}_results.xlsx"
    df.to_excel(filename, index=False)
    print(f"Results saved to {filename}")

def compute_curves(gt_probs, pred_probs):
    """计算ROC和PR曲线数据"""
    if len(gt_probs) == 0 or len(pred_probs) == 0:
        return None, None, None, None, None, None
    
    gt_binary = (np.array(gt_probs) > 0.5).astype(int)
    pred_array = np.array(pred_probs)
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(gt_binary, pred_array)
    roc_auc = auc(fpr, tpr)
    
    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(gt_binary, pred_array)
    ap_score = average_precision_score(gt_binary, pred_array)
    
    return fpr, tpr, roc_auc, precision, recall, ap_score

from analysis.json0 import config_data 

def main():
    # 处理每个测试配置
    results = {}
    colors = {}
    all_image_results = {}
    roc_data = {}
    pr_data = {}
    
    for config in config_data["experiments"]:
        name, metrics, image_results, curve_data = process_test_config(config)
        results[name] = metrics
        colors[name] = config["color"]
        all_image_results[name] = image_results
        
        # 计算曲线数据
        gt_probs, pred_probs = curve_data
        fpr, tpr, roc_auc, precision, recall, ap_score = compute_curves(gt_probs, pred_probs)
        
        if fpr is not None:
            roc_data[name] = (fpr, tpr, roc_auc)
            pr_data[name] = (precision, recall, ap_score)
        
        # 保存到Excel
        save_to_excel(image_results, name)
        
        print("name:", name)
        print("-" * 50)
    
    # 打印汇总结果
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Dice:      {metrics['dice']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  AP:        {metrics['ap']:.4f}")
        print(f"  maxDice:   {metrics['maxDice']:.4f}")
        print()
    
    # 生成柱状图
    create_bar_chart(results, colors)
    
    # 生成ROC曲线
    if roc_data:
        plot_roc_curves(roc_data, colors)
    
    # 生成PR曲线
    if pr_data:
        plot_pr_curves(pr_data, colors)

if __name__ == "__main__":
    main()