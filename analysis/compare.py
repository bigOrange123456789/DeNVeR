# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from PIL import Image
# import argparse

# def load_images_for_comparison(videoId, frameId, original_path, gt_path, pred_path1, pred_path2):
#     """
#     加载用于比较的四张图像：原始图像、真实标签、方法1预测、方法2预测
#     """
#     images = {}
    
#     # 原始图像
#     try:
#         orig_img_path = os.path.join(original_path, videoId, frameId)
#         images['original'] = np.array(Image.open(orig_img_path))
#     except:
#         images['original'] = None
    
#     # 真实标签
#     try:
#         gt_img_path = os.path.join(gt_path, videoId, frameId)
#         gt_img = Image.open(gt_img_path)
#         gt_img = np.array(gt_img)
#         if len(gt_img.shape) == 3:
#             gt_img = np.mean(gt_img, axis=2)
#         images['gt'] = (gt_img > 128).astype(np.uint8)  # 二值化
#     except:
#         images['gt'] = None
    
#     # 方法1预测
#     try:
#         pred1_img_path = os.path.join(pred_path1, videoId, frameId)
#         pred1_img = Image.open(pred1_img_path)
#         pred1_img = np.array(pred1_img)
#         if len(pred1_img.shape) == 3:
#             pred1_img = np.mean(pred1_img, axis=2)
#         images['pred1'] = (pred1_img > 128).astype(np.uint8)  # 二值化
#     except:
#         images['pred1'] = None
    
#     # 方法2预测
#     try:
#         pred2_img_path = os.path.join(pred_path2, videoId, frameId)
#         pred2_img = Image.open(pred2_img_path)
#         pred2_img = np.array(pred2_img)
#         if len(pred2_img.shape) == 3:
#             pred2_img = np.mean(pred2_img, axis=2)
#         images['pred2'] = (pred2_img > 128).astype(np.uint8)  # 二值化
#     except:
#         images['pred2'] = None
    
#     return images

# def create_comparison_figure(cases, title, original_path, gt_path, pred_path1, pred_path2, k):
#     """
#     创建比较图：4行×K列，显示原始图像、GT、方法1预测、方法2预测
#     """
#     fig, axes = plt.subplots(4, k, figsize=(3*k, 12))
    
#     # 如果只有一列，确保axes是二维数组
#     if k == 1:
#         axes = axes.reshape(4, 1)
    
#     for col, (_, row) in enumerate(cases.iterrows()):
#         videoId = row['videoId']
#         frameId = row['frameId']
#         metric_value1 = row['metric1']
#         metric_value2 = row['metric2']
#         diff = row['diff']
        
#         # 加载图像
#         images = load_images_for_comparison(videoId, frameId, original_path, gt_path, pred_path1, pred_path2)
        
#         # 显示原始图像
#         if images['original'] is not None:
#             if len(images['original'].shape) == 2:
#                 axes[0, col].imshow(images['original'], cmap='gray')
#             else:
#                 axes[0, col].imshow(images['original'])
#         axes[0, col].set_title(f'Original\n{frameId}', fontsize=10)
#         axes[0, col].axis('off')
        
#         # 显示真实标签
#         if images['gt'] is not None:
#             axes[1, col].imshow(images['gt'], cmap='gray')
#         axes[1, col].set_title('Ground Truth', fontsize=10)
#         axes[1, col].axis('off')
        
#         # 显示方法1预测
#         if images['pred1'] is not None:
#             axes[2, col].imshow(images['pred1'], cmap='gray')
#         axes[2, col].set_title(f'Method1: {metric_value1:.3f}', fontsize=10)
#         axes[2, col].axis('off')
        
#         # 显示方法2预测
#         if images['pred2'] is not None:
#             axes[3, col].imshow(images['pred2'], cmap='gray')
#         axes[3, col].set_title(f'Method2: {metric_value2:.3f}', fontsize=10)
#         axes[3, col].axis('off')
    
#     plt.suptitle(title, fontsize=16, y=0.95)
#     plt.tight_layout()
#     return fig

# from analysis.json0 import config_data 
# def main():
#     # 设置参数解析
#     parser = argparse.ArgumentParser(description='Compare two experiment results and visualize differences')
#     parser.add_argument('--excel1', 
#                         default='_011_continuity_01-orig-CATH_results.xlsx',
#                         type=str, help='Path to first Excel file')
#     parser.add_argument('--excel2', 
#                         default='_011_continuity_02-orig-CATH_results.xlsx',
#                         type=str, help='Path to second Excel file')
#     parser.add_argument('--metric', 
#                         default='recall',
#                         type=str, help='Metric column name (e.g., dice, recall, precision)')
#     parser.add_argument('--k', 
#                         default=10,
#                         type=int, help='Number of top cases to display for each category')
#     parser.add_argument('--original_path', 
#                         default='./outputs/xca_dataset_sim2_copy/images',
#                         type=str, help='Path to original images')
#     parser.add_argument('--gt_path', type=str, help='Path to ground truth masks')
#     parser.add_argument('--pred_path1', type=str, help='Path to method1 prediction masks')
#     parser.add_argument('--pred_path2', type=str, help='Path to method2 prediction masks')
#     parser.add_argument('--output_dir', type=str, 
#                         default='./outputs', 
#                         help='Output directory for results')
    
#     # name1=args
#     # config1=config_data[""]
#     args = parser.parse_args()
#     name1 = args.excel1.split("_results.xlsx")[0]
#     name2=args.excel2.split("_results.xlsx")[0]
#     args.excel1 = "outputs/"+args.excel1
#     args.excel2 = "outputs/"+args.excel2
#     config1={}
#     config2={}
#     for i in config_data["experiments"]:
#         print("i name",i["name"])
#         if i["name"]==name1: 
#             config1=i
#         if i["name"]==name2: 
#             config2=i
#     args.gt_path=config1["gt_path"]
#     args.pred_path1=config1["pred_path"]
#     args.pred_path2=config2["pred_path"]

    
#     # 创建输出目录
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     # 读取Excel文件
#     print(f"Loading {args.excel1} and {args.excel2}...")
#     df1 = pd.read_excel(args.excel1)
#     df2 = pd.read_excel(args.excel2)
    
#     # 确保两个DataFrame有相同的行数且对应相同的图片
#     if len(df1) != len(df2):
#         print("Warning: The two Excel files have different numbers of rows!")
    
#     # 合并两个DataFrame
#     merged_df = pd.merge(
#         df1[['videoId', 'frameId', args.metric]], 
#         df2[['videoId', 'frameId', args.metric]], 
#         on=['videoId', 'frameId'], 
#         suffixes=('_1', '_2')
#     )
    
#     # 计算差异
#     merged_df['metric1'] = merged_df[f'{args.metric}_1']
#     merged_df['metric2'] = merged_df[f'{args.metric}_2']
#     merged_df['diff'] = merged_df['metric1'] - merged_df['metric2']
    
#     # 找出差异最大的案例
#     # 方法1优于方法2最多的K个案例
#     method1_better = merged_df.nlargest(args.k, 'diff')
#     # 方法2优于方法1最多的K个案例（即差异最小的K个）
#     method2_better = merged_df.nsmallest(args.k, 'diff')
    
#     print(f"Found {len(method1_better)} cases where method1 is better than method2")
#     print(f"Found {len(method2_better)} cases where method2 is better than method1")
    
#     # 创建第一张图：方法1优于方法2的案例
#     if len(method1_better) > 0:
#         fig1 = create_comparison_figure(
#             method1_better, 
#             f'Top {args.k} Cases: Method1 > Method2 (by {args.metric})',
#             args.original_path,
#             args.gt_path,
#             args.pred_path1,
#             args.pred_path2,
#             min(args.k, len(method1_better))
#         )
#         fig1.savefig(os.path.join(args.output_dir, f'method1_better_{args.metric}.png'), dpi=150, bbox_inches='tight')
#         print(f"Saved method1_better_{args.metric}.png")
    
#     # 创建第二张图：方法2优于方法1的案例
#     if len(method2_better) > 0:
#         fig2 = create_comparison_figure(
#             method2_better,
#             f'Top {args.k} Cases: Method2 > Method1 (by {args.metric})',
#             args.original_path,
#             args.gt_path,
#             args.pred_path1,
#             args.pred_path2,
#             min(args.k, len(method2_better))
#         )
#         fig2.savefig(os.path.join(args.output_dir, f'method2_better_{args.metric}.png'), dpi=150, bbox_inches='tight')
#         print(f"Saved method2_better_{args.metric}.png")
    
#     # 保存差异分析结果到CSV
#     comparison_summary = pd.concat([
#         method1_better.assign(category='method1_better'),
#         method2_better.assign(category='method2_better')
#     ])
#     comparison_summary.to_csv(os.path.join(args.output_dir, f'comparison_summary_{args.metric}.csv'), index=False)
#     print(f"Saved comparison_summary_{args.metric}.csv")
    
#     # 打印统计信息
#     print(f"\nComparison Summary for {args.metric}:")
#     print(f"Average {args.metric} - Method1: {merged_df['metric1'].mean():.4f}")
#     print(f"Average {args.metric} - Method2: {merged_df['metric2'].mean():.4f}")
#     print(f"Average difference (Method1 - Method2): {merged_df['diff'].mean():.4f}")
#     print(f"Method1 better in {len(method1_better)} cases")
#     print(f"Method2 better in {len(method2_better)} cases")
    
#     plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import argparse
from scipy import stats
import seaborn as sns

def load_images_for_comparison(videoId, frameId, original_path, gt_path, pred_path1, pred_path2):
    """
    加载用于比较的四张图像：原始图像、真实标签、方法1预测、方法2预测
    """
    images = {}
    
    # 原始图像
    try:
        orig_img_path = os.path.join(original_path, videoId, frameId)
        images['original'] = np.array(Image.open(orig_img_path))
    except:
        images['original'] = None
    
    # 真实标签
    try:
        gt_img_path = os.path.join(gt_path, videoId, frameId)
        gt_img = Image.open(gt_img_path)
        gt_img = np.array(gt_img)
        if len(gt_img.shape) == 3:
            gt_img = np.mean(gt_img, axis=2)
        images['gt'] = (gt_img > 128).astype(np.uint8)  # 二值化
    except:
        images['gt'] = None
    
    # 方法1预测
    try:
        pred1_img_path = os.path.join(pred_path1, videoId, frameId)
        pred1_img = Image.open(pred1_img_path)
        pred1_img = np.array(pred1_img)
        if len(pred1_img.shape) == 3:
            pred1_img = np.mean(pred1_img, axis=2)
        images['pred1'] = (pred1_img > 128).astype(np.uint8)  # 二值化
    except:
        images['pred1'] = None
    
    # 方法2预测
    try:
        pred2_img_path = os.path.join(pred_path2, videoId, frameId)
        pred2_img = Image.open(pred2_img_path)
        pred2_img = np.array(pred2_img)
        if len(pred2_img.shape) == 3:
            pred2_img = np.mean(pred2_img, axis=2)
        images['pred2'] = (pred2_img > 128).astype(np.uint8)  # 二值化
    except:
        images['pred2'] = None
    
    return images

def create_comparison_figure(cases, title, original_path, gt_path, pred_path1, pred_path2, k, metric, method1_name="Method1", method2_name="Method2"):
    """
    创建比较图：4行×K列，显示原始图像、GT、方法1预测、方法2预测
    """
    fig, axes = plt.subplots(4, k, figsize=(3*k, 12))
    
    # 如果只有一列，确保axes是二维数组
    if k == 1:
        axes = axes.reshape(4, 1)
    
    for col, (_, row) in enumerate(cases.iterrows()):
        videoId = row['videoId']
        frameId = row['frameId']
        metric_value1 = row['metric1']
        metric_value2 = row['metric2']
        diff = row['diff']
        
        # 加载图像
        images = load_images_for_comparison(videoId, frameId, original_path, gt_path, pred_path1, pred_path2)
        
        # 显示原始图像
        if images['original'] is not None:
            if len(images['original'].shape) == 2:
                axes[0, col].imshow(images['original'], cmap='gray')
            else:
                axes[0, col].imshow(images['original'])
        # 第一行同时显示videoId和frameId
        axes[0, col].set_title(f'{videoId}/{frameId}', fontsize=7) #=10)
        axes[0, col].axis('off')
        
        # 显示真实标签
        if images['gt'] is not None:
            axes[1, col].imshow(images['gt'], cmap='gray')
        axes[1, col].set_title('Ground Truth', fontsize=7) #10)
        axes[1, col].axis('off')
        
        # 显示方法1预测
        if images['pred1'] is not None:
            axes[2, col].imshow(images['pred1'], cmap='gray')
        # 第三行显示指标名称和方法名称
        axes[2, col].set_title(f'{method1_name}\n{metric}: {metric_value1:.3f}', fontsize=7)
        axes[2, col].axis('off')
        
        # 显示方法2预测
        if images['pred2'] is not None:
            axes[3, col].imshow(images['pred2'], cmap='gray')
        # 第四行显示指标名称和方法名称
        axes[3, col].set_title(f'{method2_name}\n{metric}: {metric_value2:.3f}', fontsize=7)
        axes[3, col].axis('off')
    
    plt.suptitle(title, fontsize=10,#16,
                  y=0.5)#y=0.95)
    plt.tight_layout()
    return fig

def calculate_significance_matrix(df1, df2, metrics):
    """
    计算任意两个指标之间的显著性差异矩阵
    """
    # 合并两个DataFrame
    merged_df = pd.merge(
        df1[['videoId', 'frameId'] + metrics], 
        df2[['videoId', 'frameId'] + metrics], 
        on=['videoId', 'frameId'], 
        suffixes=('_1', '_2')
    )
    
    # 初始化显著性矩阵
    n_metrics = len(metrics)
    p_value_matrix = np.zeros((n_metrics, n_metrics))
    
    # 计算每对指标的配对t检验p值
    for i, metric1 in enumerate(metrics):
        for j, metric2 in enumerate(metrics):
            if i == j:
                # 对角线：同一指标在两个方法间的差异
                diff = merged_df[f'{metric1}_1'] - merged_df[f'{metric1}_2']
                _, p_value = stats.ttest_rel(merged_df[f'{metric1}_1'], merged_df[f'{metric1}_2'])
            else:
                # 非对角线：不同指标在同一方法内的差异
                _, p_value = stats.ttest_rel(merged_df[f'{metric1}_1'], merged_df[f'{metric2}_1'])
            
            p_value_matrix[i, j] = p_value
    
    # 创建DataFrame
    p_value_df = pd.DataFrame(p_value_matrix, index=metrics, columns=metrics)
    
    return p_value_df

def create_comparison_summary(merged_df, metric, method1_name="Method1", method2_name="Method2"):
    """
    创建比较摘要，包括方法1比方法2更好/更差的图像数量和比例
    """
    # 计算差异
    merged_df['diff'] = merged_df[f'{metric}_1'] - merged_df[f'{metric}_2']
    
    # 统计
    method1_better_count = len(merged_df[merged_df['diff'] > 0])
    method2_better_count = len(merged_df[merged_df['diff'] < 0])
    equal_count = len(merged_df[merged_df['diff'] == 0])
    total_count = len(merged_df)
    
    # 创建摘要DataFrame
    summary_df = pd.DataFrame({
        'Comparison': [
            f'{method1_name} better than {method2_name}',
            f'{method2_name} better than {method1_name}',
            'Equal',
            'Total'
        ],
        'Count': [
            method1_better_count,
            method2_better_count,
            equal_count,
            total_count
        ],
        'Percentage': [
            method1_better_count / total_count * 100,
            method2_better_count / total_count * 100,
            equal_count / total_count * 100,
            100
        ]
    })
    
    return summary_df

from analysis.json0 import config_data 
def main():
    if not len(config_data["experiments"])==2:
        print("analysis/json0.py 中参数列表中实验的个数必须为2!!!!")
        exit(0)
    # 设置参数解析
    parser = argparse.ArgumentParser(description='Compare two experiment results and visualize differences')
    # parser.add_argument('--excel1', type=str, required=True, help='Path to first Excel file')
    # parser.add_argument('--excel2', type=str, required=True, help='Path to second Excel file')
    # parser.add_argument('--metric', type=str, required=True, help='Metric column name (e.g., dice, recall, precision)')
    # parser.add_argument('--k', type=int, required=True, help='Number of top cases to display for each category')
    # parser.add_argument('--original_path', type=str, required=True, help='Path to original images')
    # parser.add_argument('--gt_path', type=str, required=True, help='Path to ground truth masks')
    # parser.add_argument('--pred_path1', type=str, required=True, help='Path to method1 prediction masks')
    # parser.add_argument('--pred_path2', type=str, required=True, help='Path to method2 prediction masks')
    # parser.add_argument('--output_dir', type=str, default='./comparison_results', help='Output directory for results')
    # parser.add_argument('--method1_name', type=str, default='Method1', help='Name of first method')
    # parser.add_argument('--method2_name', type=str, default='Method2', help='Name of second method')
    # name1="_015_01_noRigid1(b4000)-CATH"
    # name2="_015_02_noRigid1(b2000)-CATH"

    name1=config_data["experiments"][0]["name"]#"_017_02_nr(b2000)"#"_017_05_rigid.non(doubleStage)"
    # name1 = "_016_01_noRigid1(b1000)[smooth]"+"-CATH"
    name2=config_data["experiments"][1]["name"]#"_017_07_orig(sub2)"
    # name2 = "_016_01_noAllRigid1(b1000)[smooth]"+"-CATH"
    # name1 = name1+"-CATH"
    # name2 = name2+"-CATH"
    print("name1",name1)
    print("name2",name2)
    parser.add_argument('--excel1', 
                        default=name1+'_results.xlsx',
                        type=str, help='Path to first Excel file')
    parser.add_argument('--excel2', 
                        default=name2+'_results.xlsx',
                        type=str, help='Path to second Excel file')
    parser.add_argument('--metric', 
                        default='dice',
                        type=str, help='Metric column name (e.g., dice, recall, precision)')
    parser.add_argument('--k', 
                        default=10,
                        type=int, help='Number of top cases to display for each category')
    parser.add_argument('--original_path', 
                        # default='./outputs/xca_dataset_sub1_copy/images',
                        default='./outputs/xca_dataset_sub2_copy/images',
                        type=str, help='Path to original images')
    parser.add_argument('--gt_path', type=str, help='Path to ground truth masks')
    parser.add_argument('--pred_path1', type=str, help='Path to method1 prediction masks')
    parser.add_argument('--pred_path2', type=str, help='Path to method2 prediction masks')
    parser.add_argument('--output_dir', type=str, 
                        default='./outputs/compare', 
                        help='Output directory for results')
    parser.add_argument('--method1_name', type=str, default='Method1', help='Name of first method')
    parser.add_argument('--method2_name', type=str, default='Method2', help='Name of second method')
    
    args = parser.parse_args()
    name1 = args.excel1.split("_results.xlsx")[0]
    name2 = args.excel2.split("_results.xlsx")[0]
    args.method1_name = name1
    args.method2_name = name2
    args.excel1 = "outputs/"+args.excel1
    args.excel2 = "outputs/"+args.excel2
    config1={}
    config2={}
    for i in config_data["experiments"]:
        print("i name",i["name"],i["name"]==name1,i["name"],name1)
        if i["name"]==name1: 
            config1=i
        if i["name"]==name2: 
            config2=i
        i["name"]=i["name"]+"_results.xlsx"

    args.gt_path=config1["gt_path"]
    args.pred_path1=config1["pred_path"]
    args.pred_path2=config2["pred_path"]
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取Excel文件
    print(f"Loading {args.excel1} and {args.excel2}...")
    df1 = pd.read_excel(args.excel1)
    df2 = pd.read_excel(args.excel2)
    
    # 确保两个DataFrame有相同的行数且对应相同的图片
    if len(df1) != len(df2):
        print("Warning: The two Excel files have different numbers of rows!")
    
    # 合并两个DataFrame
    merged_df = pd.merge(
        df1[['videoId', 'frameId', args.metric]], 
        df2[['videoId', 'frameId', args.metric]], 
        on=['videoId', 'frameId'], 
        suffixes=('_1', '_2')
    )
    
    # 计算差异
    merged_df['metric1'] = merged_df[f'{args.metric}_1']
    merged_df['metric2'] = merged_df[f'{args.metric}_2']
    merged_df['diff'] = merged_df['metric1'] - merged_df['metric2']
    
    # 找出差异最大的案例
    # 方法1优于方法2最多的K个案例
    method1_better = merged_df.nlargest(args.k, 'diff')
    # 方法2优于方法1最多的K个案例（即差异最小的K个）
    method2_better = merged_df.nsmallest(args.k, 'diff')
    
    print(f"Found {len(method1_better)} cases where {args.method1_name} is better than {args.method2_name}")
    print(f"Found {len(method2_better)} cases where {args.method2_name} is better than {args.method1_name}")
    
    # 创建第一张图：方法1优于方法2的案例
    if len(method1_better) > 0:
        fig1 = create_comparison_figure(
            method1_better, 
            f'Top {args.k} Cases: {args.method1_name} > {args.method2_name} (by {args.metric})',
            args.original_path,
            args.gt_path,
            args.pred_path1,
            args.pred_path2,
            min(args.k, len(method1_better)),
            args.metric,
            args.method1_name,
            args.method2_name
        )
        fig1.savefig(os.path.join(args.output_dir, f'{args.method1_name}_better_{args.metric}.png'), dpi=150, bbox_inches='tight')
        print(f"Saved {args.method1_name}_better_{args.metric}.png")
    
    # 创建第二张图：方法2优于方法1的案例
    if len(method2_better) > 0:
        fig2 = create_comparison_figure(
            method2_better,
            f'Top {args.k} Cases: {args.method2_name} > {args.method1_name} (by {args.metric})',
            args.original_path,
            args.gt_path,
            args.pred_path1,
            args.pred_path2,
            min(args.k, len(method2_better)),
            args.metric,
            args.method1_name,
            args.method2_name
        )
        fig2.savefig(os.path.join(args.output_dir, f'{args.method2_name}_better_{args.metric}.png'), dpi=150, bbox_inches='tight')
        print(f"Saved {args.method2_name}_better_{args.metric}.png")
    
    # 保存差异分析结果到CSV
    comparison_summary = pd.concat([
        method1_better.assign(category=f'{args.method1_name}_better'),
        method2_better.assign(category=f'{args.method2_name}_better')
    ])
    comparison_summary.to_csv(os.path.join(args.output_dir, f'comparison_summary_{args.metric}.csv'), index=False)
    print(f"Saved comparison_summary_{args.metric}.csv")
    
    # 创建显著性差异矩阵和比较摘要
    # 获取所有可能的指标列
    common_columns = set(df1.columns) & set(df2.columns)
    metric_columns = [col for col in common_columns if col not in ['videoId', 'frameId']]
    
    print(f"Found common metrics: {metric_columns}")
    
    # 计算显著性矩阵
    significance_matrix = calculate_significance_matrix(df1, df2, metric_columns)
    
    # 创建比较摘要
    comparison_stats = create_comparison_summary(
        merged_df, 
        args.metric, 
        args.method1_name, 
        args.method2_name
    )
    
    # 保存到Excel文件
    with pd.ExcelWriter(os.path.join(args.output_dir, f'statistical_analysis_{args.metric}.xlsx')) as writer:
        significance_matrix.to_excel(writer, sheet_name='Significance Matrix')
        comparison_stats.to_excel(writer, sheet_name='Comparison Summary', index=False)
        
        # 添加详细的配对比较
        detailed_comparison = merged_df.copy()
        detailed_comparison['Comparison'] = np.where(
            detailed_comparison['diff'] > 0, 
            f'{args.method1_name} better', 
            np.where(
                detailed_comparison['diff'] < 0, 
                f'{args.method2_name} better', 
                'Equal'
            )
        )
        detailed_comparison.to_excel(writer, sheet_name='Detailed Comparison', index=False)
    
    print(f"Saved statistical_analysis_{args.metric}.xlsx")
    
    # 打印统计信息
    print(f"\nComparison Summary for {args.metric}:")
    print(f"Average {args.metric} - {args.method1_name}: {merged_df['metric1'].mean():.4f}")
    print(f"Average {args.metric} - {args.method2_name}: {merged_df['metric2'].mean():.4f}")
    print(f"Average difference ({args.method1_name} - {args.method2_name}): {merged_df['diff'].mean():.4f}")
    
    method1_better_count = len(merged_df[merged_df['diff'] > 0])
    method2_better_count = len(merged_df[merged_df['diff'] < 0])
    equal_count = len(merged_df[merged_df['diff'] == 0])
    total_count = len(merged_df)
    
    print(f"{args.method1_name} better in {method1_better_count} cases ({method1_better_count/total_count*100:.1f}%)")
    print(f"{args.method2_name} better in {method2_better_count} cases ({method2_better_count/total_count*100:.1f}%)")
    print(f"Equal in {equal_count} cases ({equal_count/total_count*100:.1f}%)")
    
    # 执行配对t检验
    t_stat, p_value = stats.ttest_rel(merged_df['metric1'], merged_df['metric2'])
    print(f"\nPaired t-test results:")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Statistically significant difference (p < 0.05)")
    else:
        print("No statistically significant difference (p >= 0.05)")
    
    plt.show()

if __name__ == "__main__":
    main()

