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
        if not threshold is None:#阈值非空的时候进行二值化
            img_array = (img_array > 256*threshold).astype(np.uint8)
        else:
            img_array = img_array.astype(np.float32)/256
        img_array = img_array.astype(np.float32)#/256
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def calculate_metrics(gt, pred):
    """计算Dice系数、查全率、查准率"""
    # 展平数组
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    
    # 计算TP, FP, FN
    tp = np.sum((gt_flat == 1) & (pred_flat == 1))
    fp = np.sum((gt_flat == 0) & (pred_flat == 1))
    fn = np.sum((gt_flat == 1) & (pred_flat == 0))
    tn = np.sum((gt_flat == 0) & (pred_flat == 0))
    
    # 计算指标
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return dice, recall, precision


import pandas as pd
def getNameList():
    # 读取 Excel 文件
    df = pd.read_excel('./outputs/namelist.xlsx', sheet_name='文件夹列表')

    # # 确保 videoId 列是字符串类型（避免格式问题）
    # df['videoId'] = df['videoId'].astype(str)

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

def process_test_config(config, usedVideoId,DataFiltering=None):
    
    # print(video_dict)
    

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
    print(f"Pred path: {pred_path}")
    if block_cath: print(f"cath_path: {cath_path}")
    
    # 计算所有图像的指标
    all_dice = []
    all_recall = []
    all_precision = []
    
    for videoId in os.listdir(gt_path):
     if usedVideoId==None or(videoId in usedVideoId):
    #  print(videoId,get_video_info(videoId))
      long0 = get_video_info(videoId)["long"]
      short0 = get_video_info(videoId)["short"]
    #  exit(0)
      for frameId in os.listdir(gt_path+"/"+videoId):
       flag =True#使用全部数据
       if DataFiltering in ["T","Move","FrontBack"]:
        flag= (long0==DataFiltering)
        # print(long0,DataFiltering,flag)
       if flag:
      #if long0=="T" and short0=="T":
      #if long0=="Move" and short0=="Move":
        # print(long0, short0)
        filename = frameId
        gt_img = load_and_binarize_image(os.path.join(gt_path, videoId, frameId),0.5)
        # print("pred_path",pred_path)
        # print("os.path.join(pred_path, videoId, frameId)",os.path.join(pred_path, videoId, frameId))
        # exit(0)
        pred_img = load_and_binarize_image(os.path.join(pred_path, videoId, frameId),config["threshold"])
        if block_cath: 
            cath_img = load_and_binarize_image(os.path.join(cath_path, videoId+"CATH", frameId),None)
            mask_cath = np.zeros_like(cath_img)#torch.zeros_like(gt_tensor)
            mask_vessel = np.zeros_like(cath_img)
            mask_cath[cath_img>0.75]=1
            mask_vessel[(cath_img>0.25) & (cath_img<0.75)]=1 
            gt_img = mask_vessel
            pred_img = pred_img * ( 1 - mask_cath )

        
        if gt_img is not None and pred_img is not None:
            # 确保图像尺寸相同
            if gt_img.shape != pred_img.shape:
                print(f"Warning: Image shape mismatch for {filename}, resizing pred to match gt")
                pred_img = Image.fromarray(pred_img).resize((gt_img.shape[1], gt_img.shape[0]), Image.NEAREST)
                pred_img = np.array(pred_img)
            
            dice, recall, precision = calculate_metrics(gt_img, pred_img)
            all_dice.append(dice)
            all_recall.append(recall)
            all_precision.append(precision)
    
    # 计算平均指标
    if len(all_dice) > 0:
        avg_metrics = {
            "dice": np.mean(all_dice),
            "recall": np.mean(all_recall),
            "precision": np.mean(all_precision)
        }
        print(f"Average metrics for {name}: Dice={avg_metrics['dice']:.4f}, "
              f"Recall={avg_metrics['recall']:.4f}, Precision={avg_metrics['precision']:.4f}")
    else:
        print(f"Warning: No valid image pairs found for {name}")
        avg_metrics = {"dice": 0, "recall": 0, "precision": 0}
    
    return name, avg_metrics

def create_bar_chart(results, colors,DataFiltering=None):
    """创建按指标分组的柱状图"""
    metrics = ['Dice', 'Recall', 'Precision']
    print("results.keys()",results.keys())
    test_names = list(results.keys())
    # print("test_names",test_names)
    test_names2=[]
    for i in test_names:
        test_names2.append("tag:"+i)
    
    # 准备数据
    # dice_values = [results[name]['dice'] for name in test_names]
    # recall_values = [results[name]['recall'] for name in test_names]
    # precision_values = [results[name]['precision'] for name in test_names]
    
    # 设置柱状图位置
    x = np.arange(len(metrics))
    width = 0.8 / len(test_names)  # 根据测试配置数量调整宽度
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 为每个测试配置绘制柱状图
    for i, name in enumerate(test_names):
        offset = (i - len(test_names)/2 + 0.5) * width
        values = [results[name]['dice'], results[name]['recall'], results[name]['precision']]
        bars = ax.bar(x + offset, values, width, label=test_names2[i], color=colors[name])
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 设置图形属性
    # ax.set_xlabel('Metrics')
    # ax.set_ylabel('Scores')
    ax.set_title('Segmentation Performance Metrics Comparison.(DataFiltering='+str(DataFiltering)+")")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics) #显示指标名称
    ax.legend()#显示每个颜色的含义
    ax.grid(True, alpha=0.3)
    
    # 设置y轴范围
    ax.set_ylim(0, 1.1)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    if False: plt.savefig('segmentation_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

from analysis.json0 import config_data 
def main():
    # 处理每个测试配置
    results = {}
    colors = {}
    
    for config in config_data["experiments"]:
        name, metrics = process_test_config(config, 
                                            config_data["usedVideoId"],
                                            DataFiltering=config_data["DataFiltering"],
                                            
                                            )
        results[name] = metrics
        colors[name] = config["color"]
        print("name:",name)
        print("-" * 50)
    print("results",results.keys())
    # exit(0)
    
    # 打印汇总结果
    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Dice:      {metrics['dice']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print()
    
    # 生成柱状图
    create_bar_chart(results, colors,DataFiltering=config_data["DataFiltering"])
    print("Bar chart saved as 'segmentation_metrics_comparison.png'")

if __name__ == "__main__":
    print("test:2020.11.26-12:31")
    main()



