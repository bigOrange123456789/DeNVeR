import os
import yaml
import json
from nir.new import startDecouple1,startDecouple3

from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torch
from free_cos.newTrain import initCSV, save2CVS, getIndicators
from nir.analysisOri import getModel
import numpy as np

script_path = os.path.abspath(__file__)
ROOT = os.path.dirname(script_path)
# datasetPath=os.path.join(ROOT,"../","../DeNVeR_in/xca_dataset")

# 打开并读取 YAML 文件
ROOT1 = os.path.dirname(script_path)
file_path = os.path.join(ROOT1, "../",'confs/newConfig.yaml')
with open(file_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
    rootPath = config["my"]["filePathRoot"]
    datasetPath0 = config["my"]["datasetPath"] #真值标签的路径
    datasetPath = os.path.join(rootPath, "dataset_decouple") #解耦数据的路径
    datasetPath = os.path.join("..","DeNVeR.008","log_8", "dataset_decouple") #解耦数据的路径
    print("真值标签的路径:",datasetPath0)
    print("解耦数据的路径:",datasetPath)
    # outpath = os.path.join(rootPath, "result", patientID, "decouple", videoId) #本地路径

def getImg(TAG,transform, patientID, videoId, frameId):
    if TAG=="orig": #orig f1: (tensor(0.7558),) pr: (tensor(0.7176),) sn: tensor(0.8118) 
        img_path = os.path.join(datasetPath, patientID, "decouple", videoId, "orig", frameId)
        img = Image.open(img_path).convert('L') 
        img = transform(img).unsqueeze(0).cuda()
    elif TAG=="fluid":
        img_path = os.path.join(datasetPath, patientID, "decouple", videoId, "recon_non2", frameId)
        img = Image.open(img_path).convert('L') 
        img = transform(img).unsqueeze(0).cuda()
    elif TAG=="fluid2":#fluid2 f1: (tensor(0.7668),) pr: (tensor(0.7599),) sn: tensor(0.7875)
        img_path = os.path.join(datasetPath, patientID, "decouple", videoId, "C.recon_non2", frameId)
        img = Image.open(img_path).convert('L') 
        img = transform(img).unsqueeze(0).cuda()
    elif TAG=="noRigid1": #soft_fluid_1 f1: (tensor(0.7707),) pr: (tensor(0.7516),) sn: tensor(0.8020)
        img_path = os.path.join(datasetPath, patientID, "decouple", videoId, "A.rigid.main_non1", frameId)
        img = Image.open(img_path).convert('L') 
        img = transform(img).unsqueeze(0).cuda()
    elif TAG=="noRigid2":#noRigid2 f1: (tensor(0.7772),) pr: (tensor(0.7564),) sn: tensor(0.8091)
        img_path = os.path.join(datasetPath, patientID, "decouple", videoId, "A.rigid.main_non2", frameId) # A.mask.main_nr2
        img = Image.open(img_path).convert('L') 
        img = transform(img).unsqueeze(0).cuda()
    elif TAG=="mix": #mix f1: (tensor(0.7630),) pr: (tensor(0.7348),) sn: tensor(0.8062)
        img_path1 = os.path.join(datasetPath, patientID, "decouple", videoId, "recon_non2", frameId)
        img1 = Image.open(img_path1).convert('L') 
        img1 = transform(img1).unsqueeze(0).cuda()
        img_path2 = os.path.join(datasetPath, patientID, "decouple", videoId, "orig", frameId)
        img2 = Image.open(img_path2).convert('L') 
        img2 = transform(img2).unsqueeze(0).cuda()
        img=(img1+img2)/2
    elif TAG=="mix2":
        img_path1 = os.path.join(datasetPath, patientID, "decouple", videoId, "recon_non2", frameId)#流体
        img1 = Image.open(img_path1).convert('L') 
        img1 = transform(img1).unsqueeze(0).cuda()
        img_path2 = os.path.join(datasetPath, patientID, "decouple", videoId, "A.rigid.main_non1", frameId)#非刚体1
        img2 = Image.open(img_path2).convert('L') 
        img2 = transform(img2).unsqueeze(0).cuda()
        img=(img1+img2)/2
    elif TAG=="mix4": #mix4 f1: (tensor(0.7702),) pr: (tensor(0.7544),) sn: tensor(0.7988)
        img_path1 = os.path.join(datasetPath, patientID, "decouple", videoId, "C.recon_non2", frameId)#流体
        img1 = Image.open(img_path1).convert('L') 
        img1 = transform(img1).unsqueeze(0).cuda()
        img_path2 = os.path.join(datasetPath, patientID, "decouple", videoId, "A.rigid.main_non1", frameId)#非刚体1
        img2 = Image.open(img_path2).convert('L') 
        img2 = transform(img2).unsqueeze(0).cuda()
        img=(img1+img2)/2
    elif TAG=="mix5": #mix5 f1: (tensor(0.7705),) pr: (tensor(0.7125),) sn: tensor(0.8461)
        # t=0.85 mix5 f1: (tensor(0.7783),) pr: (tensor(0.7446),) sn: tensor(0.8223)
        img_path1 = os.path.join(datasetPath, patientID, "decouple", videoId, "C.recon_non2", frameId)#流体2
        img1 = Image.open(img_path1).convert('L') 
        img1 = transform(img1).unsqueeze(0).cuda()
        img_path2 = os.path.join(datasetPath, patientID, "decouple", videoId, "A.rigid.main_non1", frameId)#非刚体1
        img2 = Image.open(img_path2).convert('L') 
        img2 = transform(img2).unsqueeze(0).cuda()
        img=img1*img2 # 流体*非刚体
    elif TAG=="mix6": #mix6 f1: (tensor(0.0472),) pr: (tensor(0.1330),) sn: tensor(0.0308)
        img_path1 = os.path.join(datasetPath, patientID, "decouple", videoId, "C.recon_non", frameId)#流体1
        img1 = Image.open(img_path1).convert('L') 
        img1 = transform(img1).unsqueeze(0).cuda()
        img_path2 = os.path.join(datasetPath, patientID, "decouple", videoId, "A.rigid.main_non1", frameId)#非刚体1
        img2 = Image.open(img_path2).convert('L') 
        img2 = transform(img2).unsqueeze(0).cuda()
        img=img1*img2 # 流体*非刚体
    return img
def calculate_mean_variance(TAG,transform, patientID, videoId):
    img_list = []
    path0 = os.path.join(datasetPath, patientID, "decouple", videoId, "orig")
    # print(path0,len(os.path.join(path0)))
    for frameId in os.listdir(os.path.join(path0)):
        # frameId = str(i).zfill(5)+".png"
        img = getImg(TAG,transform, patientID, videoId, frameId)
        img_list.append(img)
    imgAll = torch.cat(img_list, dim=0)
    if False: #使用分位数（5%和95%）来过滤掉离群值
        import numpy as np
        flat = imgAll.detach().cpu().numpy().flatten()
        # 计算 5% 和 95% 分位数
        q_min = np.percentile(flat, 5)
        q_max = np.percentile(flat, 95)

        # 构建 mask
        mask = (imgAll >= q_min) & (imgAll <= q_max)
        imgAll = imgAll[mask] #只保留在5%至95%分位数之间的像素值

    mean0 = imgAll.mean()
    std0 = imgAll.std()
    return mean0, std0
def calculate_mean_varianceOld(TAG,transform, patientID, videoId): #基于原始数据的均值和方差，这肯定不对啊 #处理去噪数据就要使用去噪后的均值
    image_folder = os.path.join(datasetPath, patientID, "decouple", videoId, "orig")
    # 初始化变量
    total_pixels = 0
    sum_pixels = 0.0
    sum_squared_pixels = 0.0

    # 获取文件夹中所有图片文件
    image_files = [f for f in os.listdir(image_folder) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not image_files:
        print("文件夹中没有找到图片文件")
        return None, None

    # 遍历所有图片
    for img_file in image_files:
        try:
            img_path = os.path.join(image_folder, img_file)
            img = Image.open(img_path).convert('L')  # 确保是灰度图
            img_array = np.array(img).astype(np.float32)/255.0
            # print("img_array",np.max(img_array),np.min(img_array))
            # exit(0)

            # 更新统计量
            num_pixels = img_array.size
            total_pixels += num_pixels
            sum_pixels += np.sum(img_array)
            sum_squared_pixels += np.sum(img_array.astype(np.float64) ** 2)

        except Exception as e:
            print(f"处理图片 {img_file} 时出错: {e}")
            continue

    if total_pixels == 0:
        print("没有有效的像素数据")
        return None, None

    # 计算均值和方差
    mean = sum_pixels / total_pixels
    variance = (sum_squared_pixels / total_pixels) - (mean ** 2)

    return mean, variance**0.5

def initModel(pathParam):
    import torch.backends.cudnn as cudnn
    from free_cos.ModelSegment import ModelSegment
    os.environ['MASTER_PORT'] = '169711' #“master_port”的意思是主端口
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    cudnn.benchmark = True #benchmark的意思是基准
    n_channels = 1
    num_classes =  1
    Segment_model = ModelSegment(n_channels, num_classes)
    if torch.cuda.is_available():
        Segment_model = Segment_model.cuda() # 分割模型
    checkpoint = torch.load(pathParam, map_location=torch.device('cpu'))  # 如果模型是在GPU上训练的，这里指定为'cpu'以确保兼容性
    Segment_model.load_state_dict(checkpoint['state_dict'])  # 提取模型状态字典并赋值给模型
    return Segment_model.eval()
from free_cos.main import mainFreeCOS
# def evaluate(TAG="raft", threshold=0.5):
def evaluate(TAG="raft", threshold=0.85):
    print("TAG:",TAG)
    paramPath = "../DeNVeR_in/models_config/freecos_Seg.pt"
    model = initModel(paramPath) # getModel(paramPath)
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
    ])
    patient_names = [name for name in os.listdir(datasetPath)
                 if os.path.isdir(os.path.join(datasetPath, name))]
    CountSum = 0
    for patientID in patient_names:
        patient_path = os.path.join(datasetPath, patientID, "ground_truth")
        video_names = [name for name in os.listdir(patient_path)
                       if os.path.isdir(os.path.join(patient_path, name))]
        for videoId in video_names:
            if len(videoId.split("CATH")) == 1:
                video_path = os.path.join(datasetPath, patientID, "ground_truth", videoId)
                CountSum = CountSum + len(os.listdir(video_path))

    sum_recall = 0
    sum_precision = 0
    sum_f1 = 0
    with tqdm(total=CountSum) as pbar:
     for patientID in patient_names:
        patient_path = os.path.join(datasetPath, patientID, "ground_truth")
        video_names = [name for name in os.listdir(patient_path)
                     if os.path.isdir(os.path.join(patient_path, name))]
        for videoId in video_names:
            pbar.set_postfix(videoId=f"{videoId}")
            if len(videoId.split("CATH"))==1:
                video_path = os.path.join(datasetPath, patientID, "ground_truth", videoId)
                outpath = os.path.join(datasetPath, patientID, "decouple", videoId)
                if  not TAG=="pred":
                    mean0, std0 = calculate_mean_variance(TAG,transform, patientID, videoId)
                else:
                    mainFreeCOS(paramPath, os.path.join(outpath, "A.rigid.main_non2"), os.path.join(outpath, "A.mask.main_nr2.cf"),needConnect=False)
                for frameId in os.listdir(video_path):
                    # pred_path = os.path.join(datasetPath, patientID, "decouple", videoId, "A.mask_nr2", TAG,frameId)
                    if  TAG=="pred":
                        # pred_path = os.path.join(datasetPath, patientID, "decouple", videoId, "A.mask_nr2", "filter",frameId) # f1:0.7646
                        pred_path = os.path.join(datasetPath, patientID, "decouple", videoId, "A.mask.main_nr2.cf", "filter",frameId) # f1:0.7809
                        pred = Image.open(pred_path).convert('L')
                        pred = transform(pred).unsqueeze(0).cuda()
                    else:
                        img = getImg(TAG,transform, patientID, videoId, frameId)
                        if False:
                            img = (img - img.mean() ) / img.std()
                        else:
                            img = (img - mean0 ) / std0
                        pred = model(img)["pred"] #输出是0-1

                    pred[pred > threshold] = 1 #是0~1与0~255的原因吗？
                    pred[pred <=  threshold] = 0
                    # path_gt = os.path.join(pathGt, name)
                    # gt = Image.open(path_gt).convert('L')
                    # gt = transform(gt).unsqueeze(0).cuda()
                    # gt[gt >= 0.5] = 1
                    # gt[gt < 0.5] = 0
                    # ind=getIndicators(
                    #     pred[0,0].detach().cpu()*255,
                    #     gt[0,0].detach().cpu()*255
                    # )

                    if False:
                        from preprocess.mySkeleton import getCatheter
                        catheter = getCatheter(pred[0,0].detach().cpu().numpy())
                        catheter = torch.from_numpy(catheter).unsqueeze(0).unsqueeze(0)
                        # pred = (1-catheter) * pred
                        pred[catheter]=0#导管处为背景

                    gt_path = os.path.join(datasetPath, patientID, "ground_truth", videoId,frameId)
                    gt = Image.open(gt_path).convert('L')
                    gt = transform(gt).unsqueeze(0).cuda()
                    gt[gt > 0.5] = 1
                    gt[gt <= 0.5] = 0

                    ind = getIndicators(
                        pred[0, 0].detach().cpu() * 255,
                        gt[0, 0].detach().cpu() * 255
                    )
                    sum_recall += ind["recall"]
                    sum_precision += ind["precision"]
                    sum_f1 += ind["f1"]

                    pbar.update(1)  # 每次增加 1
    if not sum_f1.dim() == 0:
        print("sum_f1",sum_f1,sum_f1[0])
        sum_f1=sum_f1[0]
    if not sum_precision.dim() == 0:
        print("sum_precision",sum_precision,sum_precision[0])
        sum_precision=sum_precision[0]
    f1 = sum_f1 / CountSum,
    pr = sum_precision / CountSum,
    sn = sum_recall / CountSum
    print(TAG,"f1:", f1, "pr:", pr, "sn:", sn )
    return f1, pr, sn
def evaluateNew(TAG="raft"):
    if False:
        f1, pr, sn = evaluate(TAG)
        print(TAG,"f1:",f1, "pr:",pr, "sn:",sn)
        return
    f1_last = -1
    pr_last = -1
    sn_last = -1
    t_last = -1
    for i in range(20):
        f1, pr, sn = evaluate(TAG, threshold= i/20)
        f1 = f1[0]
        if f1<=0 or f1 > f1_last:
            f1_last = f1
            pr_last = pr
            sn_last = sn
            t_last = i/20
    print("最佳阈值:",TAG,f1_last, pr_last, sn_last, t_last)

#################################################################################################

import pandas as pd
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import datetime

def get_image_path(tag, patientID, videoId, frameId):
    """根据TAG获取输入图片的路径"""
    if tag == "orig":
        return os.path.join(datasetPath, patientID, "decouple", videoId, "orig", frameId)
    elif tag == "fluid":
        return os.path.join(datasetPath, patientID, "decouple", videoId, "recon_non2", frameId)
    elif tag == "fluid2":
        return os.path.join(datasetPath, patientID, "decouple", videoId, "C.recon_non2", frameId)
    elif tag == "noRigid1":
        return os.path.join(datasetPath, patientID, "decouple", videoId, "A.rigid.main_non1", frameId)
    elif tag == "noRigid2":
        return os.path.join(datasetPath, patientID, "decouple", videoId, "A.rigid.main_non2", frameId)
    elif tag == "mix":
        # 对于混合方法，返回主要成分的路径
        return os.path.join(datasetPath, patientID, "decouple", videoId, "recon_non2", frameId)
    elif tag == "mix2":
        return os.path.join(datasetPath, patientID, "decouple", videoId, "recon_non2", frameId)
    elif tag == "mix4":
        return os.path.join(datasetPath, patientID, "decouple", videoId, "C.recon_non2", frameId)
    elif tag == "mix5":
        return os.path.join(datasetPath, patientID, "decouple", videoId, "C.recon_non2", frameId)
    elif tag == "mix6":
        return os.path.join(datasetPath, patientID, "decouple", videoId, "C.recon_non", frameId)
    elif tag == "pred":
        return os.path.join(datasetPath, patientID, "decouple", videoId, "A.mask.main_nr2.cf", "filter", frameId)
    else:
        return f"未知TAG: {tag}"

def perform_paired_t_test_on_dataframe(df, tag1, tag2):
    """
    对DataFrame中的两种方法的指标进行配对t检验
    """
    print(f"\n执行配对t检验: {tag1} vs {tag2}")
    
    # 找出所有指标列
    tag1_columns = [col for col in df.columns if col.startswith(tag1 + '_')]
    tag2_columns = [col for col in df.columns if col.startswith(tag2 + '_')]
    
    # 提取指标名称（去掉标签前缀）
    metrics = [col.replace(tag1 + '_', '') for col in tag1_columns]
    
    # 确保两个标签的指标名称相同
    tag2_metrics = [col.replace(tag2 + '_', '') for col in tag2_columns]
    if set(metrics) != set(tag2_metrics):
        print("警告：两种方法的指标不匹配")
        # 使用交集
        metrics = list(set(metrics) & set(tag2_metrics))
    
    results = {}
    
    for metric in metrics:
        tag1_metric_col = f"{tag1}_{metric}"
        tag2_metric_col = f"{tag2}_{metric}"
        
        # 提取数据
        data1 = df[tag1_metric_col].values
        data2 = df[tag2_metric_col].values
        
        # 移除NaN值
        mask = ~(np.isnan(data1) | np.isnan(data2))
        data1_clean = data1[mask]
        data2_clean = data2[mask]
        
        if len(data1_clean) < 2:
            print(f"警告：{metric} 指标的有效样本数不足，无法进行t检验")
            continue
        
        # 执行配对t检验
        t_stat, p_value = stats.ttest_rel(data1_clean, data2_clean)
        
        # 计算描述性统计
        mean1 = np.mean(data1_clean)
        mean2 = np.mean(data2_clean)
        mean_diff = mean2 - mean1
        std1 = np.std(data1_clean, ddof=1)
        std2 = np.std(data2_clean, ddof=1)
        std_diff = np.std(data2_clean - data1_clean, ddof=1)
        
        # 计算置信区间
        n = len(data1_clean)
        se = std_diff / np.sqrt(n)
        ci_lower = mean_diff - 1.96 * se
        ci_upper = mean_diff + 1.96 * se
        
        # 判断显著性
        if p_value < 0.001:
            significance = "***"
            significance_text = "非常显著"
        elif p_value < 0.01:
            significance = "**"
            significance_text = "很显著"
        elif p_value < 0.05:
            significance = "*"
            significance_text = "显著"
        else:
            significance = "ns"
            significance_text = "不显著"
        
        results[metric] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_' + tag1: mean1,
            'mean_' + tag2: mean2,
            'mean_difference': mean_diff,
            'std_' + tag1: std1,
            'std_' + tag2: std2,
            'std_difference': std_diff,
            'sample_size': n,
            'confidence_interval_lower': ci_lower,
            'confidence_interval_upper': ci_upper,
            'significance': significance,
            'significance_text': significance_text
        }
        
        print(f"\n=== {metric.upper()} 指标配对t检验结果 ===")
        print(f"{tag1}均值: {mean1:.4f} ± {std1:.4f}")
        print(f"{tag2}均值: {mean2:.4f} ± {std2:.4f}")
        print(f"均值差异: {mean_diff:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
        print(f"t 统计量: {t_stat:.4f}")
        print(f"p 值: {p_value:.4f} {significance}")
        print(f"统计显著性: {significance_text}")
    
    return results

def save_comparison_figure_optimized(results, K, tag1, tag2, save_path, result_type="Best"):
    """
    保存对比图为JPG文件：5行K列
    行1: tag1输入图像
    行2: tag1分割结果
    行3: tag2输入图像  
    行4: tag2分割结果
    行5: ground truth
    并在每列底部添加文件名
    
    result_type: "Best" 或 "Worst"，表示是最好还是最差的结果
    """
    print(f"创建{result_type}对比图，包含 {K} 张图像...")
    
    # 创建图形 - 根据K的大小调整图形尺寸
    fig_width = max(4 * K, 12)  # 最小宽度为12
    fig_height = 16  # 增加高度以容纳文件名
    
    # 创建图形和子图 - 6行K列（第6行用于显示文件名）
    fig, axes = plt.subplots(6, K, figsize=(fig_width, fig_height))
    
    # 如果只有一张图，调整axes的形状
    if K == 1:
        axes = axes.reshape(6, 1)
    
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 10})
    
    for i, result in enumerate(results):
        # 第1行: tag1输入图像
        axes[0, i].imshow(result['img_tag1'], cmap='gray')
        axes[0, i].set_title(f"{tag1} Input", fontsize=10)
        axes[0, i].axis('off')
        
        # 第2行: tag1分割结果
        axes[1, i].imshow(result['pred_tag1'], cmap='gray')
        axes[1, i].set_title(f"{tag1} Seg\nF1: {result['f1_tag1']:.4f}", fontsize=10)
        axes[1, i].axis('off')
        
        # 第3行: tag2输入图像
        axes[2, i].imshow(result['img_tag2'], cmap='gray')
        axes[2, i].set_title(f"{tag2} Input", fontsize=10)
        axes[2, i].axis('off')
        
        # 第4行: tag2分割结果
        axes[3, i].imshow(result['pred_tag2'], cmap='gray')
        axes[3, i].set_title(f"{tag2} Seg\nF1: {result['f1_tag2']:.4f}", fontsize=10)
        axes[3, i].axis('off')
        
        # 第5行: ground truth
        axes[4, i].imshow(result['gt'], cmap='gray')
        axes[4, i].set_title(f"Ground Truth", fontsize=10)
        axes[4, i].axis('off')
        
        # 第6行: 文件名和F1差异
        f1_diff_text = f"F1差异: {result['f1_diff']:.4f}"
        axes[5, i].text(0.5, 0.7, result["videoId"]+"/"+result['frameId'], 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[5, i].transAxes, fontsize=8)
        axes[5, i].text(0.5, 0.3, f1_diff_text,
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[5, i].transAxes, fontsize=8)
        axes[5, i].set_title("文件名和F1差异", fontsize=10)
        axes[5, i].axis('off')
    
    # 添加总标题
    if result_type == "Best":
        title = f'Top {K} Images Where {tag1} Outperforms {tag2} (Best F1 Difference)'
    else:
        title = f'Top {K} Images Where {tag2} Outperforms {tag1} (Worst F1 Difference)'
    
    plt.suptitle(title, fontsize=16, y=0.98)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # 为总标题留出空间
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='jpg')
    print(f"✓ {result_type}对比图像已保存: {save_path}")
    
    # 关闭图形以释放内存
    plt.close(fig)

def save_results_to_excel(df, t_test_results, tag1, tag2, save_path):
    """
    将数据和统计检验结果保存到Excel文件
    """
    print(f"保存统计结果到Excel: {save_path}")
    
    # 创建Excel写入器
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        # 1. 保存原始数据
        df.to_excel(writer, sheet_name='原始数据', index=False)
        
        # 2. 创建统计摘要表
        summary_data = []
        for metric, result in t_test_results.items():
            summary_data.append({
                '指标': metric.upper(),
                f'{tag1}均值': result[f'mean_{tag1}'],
                f'{tag1}标准差': result[f'std_{tag1}'],
                f'{tag2}均值': result[f'mean_{tag2}'],
                f'{tag2}标准差': result[f'std_{tag2}'],
                '均值差异': result['mean_difference'],
                '差异标准差': result['std_difference'],
                't统计量': result['t_statistic'],
                'p值': result['p_value'],
                '显著性': result['significance'],
                '样本量': result['sample_size'],
                '95%置信区间下限': result['confidence_interval_lower'],
                '95%置信区间上限': result['confidence_interval_upper']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='统计摘要', index=False)
        
        # 3. 创建差异数据表
        diff_data = df[['patientID', 'videoId', 'frameId', 'image_id', 
                       'tag1_image_path', 'tag2_image_path', 'ground_truth_path']].copy()
        
        for metric in t_test_results.keys():
            tag1_col = f"{tag1}_{metric}"
            tag2_col = f"{tag2}_{metric}"
            diff_col = f"{metric}_difference"
            
            if tag1_col in df.columns and tag2_col in df.columns:
                diff_data[tag1_col] = df[tag1_col]
                diff_data[tag2_col] = df[tag2_col]
                diff_data[diff_col] = df[tag2_col] - df[tag1_col]
        
        diff_data.to_excel(writer, sheet_name='差异数据', index=False)
        
        # 4. 创建方法比较表
        comparison_data = []
        for metric, result in t_test_results.items():
            comparison_data.append({
                '指标': metric.upper(),
                '检验方法': '配对t检验',
                f'{tag1}均值±标准差': f"{result[f'mean_{tag1}']:.4f} ± {result[f'std_{tag1}']:.4f}",
                f'{tag2}均值±标准差': f"{result[f'mean_{tag2}']:.4f} ± {result[f'std_{tag2}']:.4f}",
                '均值差异(95% CI)': f"{result['mean_difference']:.4f} ({result['confidence_interval_lower']:.4f} to {result['confidence_interval_upper']:.4f})",
                't值': f"{result['t_statistic']:.4f}",
                'p值': f"{result['p_value']:.4f}",
                '显著性': result['significance_text']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_excel(writer, sheet_name='方法比较', index=False)
        
        # 5. 创建图像路径表
        path_data = df[['patientID', 'videoId', 'frameId', 'image_id', 
                       'tag1_image_path', 'tag2_image_path', 'ground_truth_path']].copy()
        path_data.to_excel(writer, sheet_name='图像路径', index=False)
    
    print(f"✓ 统计结果已保存: {save_path}")
    print(f"Excel文件包含以下工作表:")
    print(f"  1. 原始数据 - 所有图像的详细指标")
    print(f"  2. 统计摘要 - 详细的统计检验结果")
    print(f"  3. 差异数据 - 两种方法的差异值")
    print(f"  4. 方法比较 - 简明的比较结果")
    print(f"  5. 图像路径 - 所有输入图像和ground truth的路径")


##########################################################################################################

def comprehensive_comparison_analysis(config1, config2, K=5, threshold=0.85, save_path="", only_annotated=True):
    """
    对两种配置进行全面的比较分析
    
    参数:
        config1: 第一种配置的JSON对象
        config2: 第二种配置的JSON对象
        K: 对比图中显示的图像数量
        threshold: 分割阈值
        save_path: 保存路径
        only_annotated: 是否只分割有人工标注的图片
    """
    tag1 = config1["name"]
    tag2 = config2["name"]
    
    print(f"开始全面的比较分析: {tag1} vs {tag2}")
    print(f"模式: {'仅处理有标注的图片' if only_annotated else '处理所有图片'}")
    print("=" * 60)
    
    # 设置默认保存路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_annotated" if only_annotated else "_all"
    save_img_path_best = f"{save_path}{tag1}_vs_{tag2}_best{mode_suffix}_{timestamp}.jpg"
    save_img_path_worst = f"{save_path}{tag1}_vs_{tag2}_worst{mode_suffix}_{timestamp}.jpg"
    save_excel_path = f"{save_path}{tag1}_vs_{tag2}_statistical_analysis{mode_suffix}_{timestamp}.xlsx"
    
    # 步骤1: 收集所有图像的指标数据
    print("步骤1/4: 收集所有图像的指标数据...")
    df, best_results, worst_results = collect_all_metrics_optimized(config1, config2, K, threshold, only_annotated)
    
    # 步骤2: 执行统计检验
    print("步骤2/4: 执行统计检验...")
    t_test_results = perform_paired_t_test_on_dataframe(df, tag1, tag2)
    
    # 步骤3: 保存对比图像
    print("步骤3/4: 保存对比图像...")
    save_comparison_figure_optimized(best_results, K, tag1, tag2, save_img_path_best, "Best")
    save_comparison_figure_optimized(worst_results, K, tag1, tag2, save_img_path_worst, "Worst")
    
    # 步骤4: 保存Excel统计结果
    print("步骤4/4: 保存Excel统计结果...")
    save_results_to_excel(df, t_test_results, tag1, tag2, save_excel_path)
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print(f"✓ 最佳对比图像已保存: {save_img_path_best}")
    print(f"✓ 最差对比图像已保存: {save_img_path_worst}")
    print(f"✓ 统计结果已保存: {save_excel_path}")
    
    return df, t_test_results, best_results, worst_results

def collect_all_metrics_optimized(config1, config2, K=5, threshold=0.85, only_annotated=True):
    """
    收集所有图像的指标数据（优化版）
    
    参数:
        config1: 第一种配置的JSON对象
        config2: 第二种配置的JSON对象
    """
    # 初始化模型和转换（只有需要模型推理的配置才需要）
    paramPath = "../DeNVeR_in/models_config/freecos_Seg.pt"
    
    # 检查哪些配置需要模型
    need_model1 = not config1.get("precomputed", False)
    need_model2 = not config2.get("precomputed", False)
    
    if need_model1 or need_model2:
        model = initModel(paramPath)
    else:
        model = None
        
    transform = transforms.Compose([transforms.ToTensor()])
    
    # 获取配置信息
    tag1 = config1["name"]
    tag2 = config2["name"]
    
    # 获取所有患者和视频
    patient_names = [name for name in os.listdir(datasetPath)
                     if os.path.isdir(os.path.join(datasetPath, name))]
    
    # 存储所有图像的指标结果
    all_results = []
    comparison_results = []
    
    # 计算总图像数量用于进度条
    total_images = 0
    if only_annotated:
        # 只处理有标注的图片
        for patientID in patient_names:
            patient_path = os.path.join(datasetPath0, patientID, "ground_truth")
            if not os.path.exists(patient_path):
                continue
            video_names = [name for name in os.listdir(patient_path)
                           if os.path.isdir(os.path.join(patient_path, name))]
            for videoId in video_names:
                if len(videoId.split("CATH")) == 1:
                    video_path = os.path.join(datasetPath0, patientID, "ground_truth", videoId)
                    total_images += len(os.listdir(video_path))
    else:
        # 处理所有图片
        for patientID in patient_names:
            patient_path = os.path.join(datasetPath, patientID, "images")
            if not os.path.exists(patient_path):
                continue
            video_names = [name for name in os.listdir(patient_path)
                           if os.path.isdir(os.path.join(patient_path, name))]
            for videoId in video_names:
                video_path = os.path.join(datasetPath, patientID, "images", videoId)
                total_images += len(os.listdir(video_path))
    
    print(f"总图像数量: {total_images}")
    
    # 创建进度条
    with tqdm(total=total_images, desc="处理图像") as pbar:
        for patientID in patient_names:
            if only_annotated:
                # 只处理有标注的图片
                patient_gt_path = os.path.join(datasetPath0, patientID, "ground_truth")
                if not os.path.exists(patient_gt_path):
                    continue
                video_names = [name for name in os.listdir(patient_gt_path)
                               if os.path.isdir(os.path.join(patient_gt_path, name))]
                
                for videoId in video_names:
                    if len(videoId.split("CATH")) == 1:
                        video_gt_path = os.path.join(datasetPath0, patientID, "ground_truth", videoId)
                        if not os.path.exists(video_gt_path):
                            continue
                            
                        pbar.set_postfix(patient=patientID, video=videoId, mode="annotated")
                        
                        # 计算均值和标准差（只有需要模型推理的配置才需要）
                        mean_tag1, std_tag1 = get_normalization_params(config1, transform, patientID, videoId)
                        mean_tag2, std_tag2 = get_normalization_params(config2, transform, patientID, videoId)
                        
                        for frameId in os.listdir(video_gt_path):
                            try:
                                process_single_image_final(patientID, videoId, frameId, 
                                                         config1, config2, tag1, tag2,
                                                         mean_tag1, std_tag1, mean_tag2, std_tag2,
                                                         model, transform, threshold, 
                                                         all_results, comparison_results, True)
                            except Exception as e:
                                print(f"处理错误 {patientID}/{videoId}/{frameId}: {str(e)}")
                            pbar.update(1)
            else:
                # 处理所有图片
                patient_img_path = os.path.join(datasetPath, patientID, "images")
                if not os.path.exists(patient_img_path):
                    continue
                video_names = [name for name in os.listdir(patient_img_path)
                               if os.path.isdir(os.path.join(patient_img_path, name))]
                
                for videoId in video_names:
                    video_img_path = os.path.join(datasetPath, patientID, "images", videoId)
                    if not os.path.exists(video_img_path):
                        continue
                        
                    pbar.set_postfix(patient=patientID, video=videoId, mode="all")
                    
                    # 计算均值和标准差（只有需要模型推理的配置才需要）
                    mean_tag1, std_tag1 = get_normalization_params(config1, transform, patientID, videoId)
                    mean_tag2, std_tag2 = get_normalization_params(config2, transform, patientID, videoId)
                    
                    for frameId in os.listdir(video_img_path):
                        try:
                            # 检查是否有对应的ground truth
                            gt_path = os.path.join(datasetPath0, patientID, "ground_truth", videoId, frameId)
                            has_gt = os.path.exists(gt_path)
                            
                            process_single_image_final(patientID, videoId, frameId,
                                                     config1, config2, tag1, tag2,
                                                     mean_tag1, std_tag1, mean_tag2, std_tag2,
                                                     model, transform, threshold,
                                                     all_results, comparison_results, has_gt)
                        except Exception as e:
                            print(f"处理错误 {patientID}/{videoId}/{frameId}: {str(e)}")
                        pbar.update(1)
    
    # 转换为DataFrame
    df = pd.DataFrame(all_results)
    
    if only_annotated or any('f1_diff' in result for result in comparison_results):
        # 找出config1优于config2的前K张图像
        valid_results = [r for r in comparison_results if 'f1_diff' in r]
        best_results = sorted(valid_results, key=lambda x: x['f1_diff'], reverse=True)[:K]
        
        # 找出config2优于config1的前K张图像
        worst_results = sorted(valid_results, key=lambda x: x['f1_diff'])[:K]
    else:
        best_results = []
        worst_results = []
    
    return df, best_results, worst_results

def get_normalization_params(config, transform, patientID, videoId):
    """
    根据配置获取归一化参数
    """
    if config.get("precomputed", False):
        # 预计算的方法不需要归一化
        return 0, 1
    else:
        # 需要模型推理的方法需要归一化
        input_mode = config["input_mode"]
        norm_method = config["norm_method"]
        return norm_method(input_mode, transform, patientID, videoId)

def process_single_image_final(patientID, videoId, frameId, 
                              config1, config2, tag1, tag2,
                              mean_tag1, std_tag1, mean_tag2, std_tag2,
                              model, transform, threshold,
                              all_results, comparison_results, has_gt=True):
    """
    处理单张图片的通用函数（最终版）
    """
    # 获取两种配置的输入图像
    img_tag1 = get_input_image(config1, transform, patientID, videoId, frameId)
    img_tag2 = get_input_image(config2, transform, patientID, videoId, frameId)
    
    # 归一化处理（只有需要模型推理的配置才需要）
    img_tag1_norm = normalize_image(config1, img_tag1, mean_tag1, std_tag1)
    img_tag2_norm = normalize_image(config2, img_tag2, mean_tag2, std_tag2)
    
    # 获取预测结果
    pred_tag1 = get_prediction_final(config1, model, img_tag1_norm, threshold, patientID, videoId, frameId)
    pred_tag2 = get_prediction_final(config2, model, img_tag2_norm, threshold, patientID, videoId, frameId)
    
    # 获取输入图片路径
    tag1_img_path = get_image_path_final(config1, patientID, videoId, frameId)
    tag2_img_path = get_image_path_final(config2, patientID, videoId, frameId)
    
    # 存储基本结果
    result = {
        'patientID': patientID,
        'videoId': videoId,
        'frameId': frameId,
        'image_id': f"{patientID}_{videoId}_{frameId}",
        'tag1_image_path': tag1_img_path,
        'tag2_image_path': tag2_img_path,
        'has_ground_truth': has_gt
    }
    
    if has_gt:
        # 获取ground truth
        gt_path = os.path.join(datasetPath0, patientID, "ground_truth", videoId, frameId)
        gt = Image.open(gt_path).convert('L')
        gt_tensor = transform(gt).unsqueeze(0).cuda()
        gt_tensor[gt_tensor > 0.5] = 1
        gt_tensor[gt_tensor <= 0.5] = 0
        
        # 计算指标
        ind_tag1 = getIndicators(
            pred_tag1[0, 0].detach().cpu() * 255,
            gt_tensor[0, 0].detach().cpu() * 255
        )
        ind_tag2 = getIndicators(
            pred_tag2[0, 0].detach().cpu() * 255,
            gt_tensor[0, 0].detach().cpu() * 255
        )
        
        f1_tag1 = ind_tag1["f1"]
        f1_tag2 = ind_tag2["f1"]
        
        # 转换为标量值
        if not f1_tag1.dim() == 0:
            f1_tag1 = f1_tag1[0]
        if not f1_tag2.dim() == 0:
            f1_tag2 = f1_tag2[0]
        
        f1_diff = float(f1_tag1 - f1_tag2)
        
        result['ground_truth_path'] = gt_path
        result['f1_diff'] = f1_diff
        
        # 添加tag1的指标
        for metric_name, metric_value in ind_tag1.items():
            result[f'{tag1}_{metric_name}'] = float(metric_value[0] if not metric_value.dim() == 0 else metric_value)
        
        # 添加tag2的指标
        for metric_name, metric_value in ind_tag2.items():
            result[f'{tag2}_{metric_name}'] = float(metric_value[0] if not metric_value.dim() == 0 else metric_value)
        
        # 存储用于图像对比的数据
        comparison_result = {
            'patientID': patientID,
            'videoId': videoId,
            'frameId': frameId,
            'f1_tag1': float(f1_tag1),
            'f1_tag2': float(f1_tag2),
            'f1_diff': f1_diff,
            'img_tag1': img_tag1.detach().cpu().numpy()[0, 0],
            'img_tag2': img_tag2.detach().cpu().numpy()[0, 0],
            'pred_tag1': pred_tag1.detach().cpu().numpy()[0, 0],
            'pred_tag2': pred_tag2.detach().cpu().numpy()[0, 0],
            'gt': gt_tensor.detach().cpu().numpy()[0, 0],
            'tag1_image_path': tag1_img_path,
            'tag2_image_path': tag2_img_path,
            'has_ground_truth': True
        }
        comparison_results.append(comparison_result)
    else:
        # 对于没有ground truth的图片，只保存预测结果
        comparison_result = {
            'patientID': patientID,
            'videoId': videoId,
            'frameId': frameId,
            'img_tag1': img_tag1.detach().cpu().numpy()[0, 0],
            'img_tag2': img_tag2.detach().cpu().numpy()[0, 0],
            'pred_tag1': pred_tag1.detach().cpu().numpy()[0, 0],
            'pred_tag2': pred_tag2.detach().cpu().numpy()[0, 0],
            'tag1_image_path': tag1_img_path,
            'tag2_image_path': tag2_img_path,
            'has_ground_truth': False
        }
        comparison_results.append(comparison_result)
    
    all_results.append(result)

def normalize_image(config, img, mean, std):
    """
    根据配置对图像进行归一化
    """
    if config.get("precomputed", False):
        # 预计算的方法不需要归一化
        return img
    else:
        # 需要模型推理的方法需要归一化
        return (img - mean) / std

##########################################################################################################

def get_prediction_final(config, model, img_norm, threshold, patientID, videoId, frameId):
    """
    获取预测结果的通用函数（最终版）
    """
    if config.get("precomputed", False):
        # 对于预计算方法，直接读取预先计算的分割结果
        result_path_template = config["result_path_template"]
        pred_path = result_path_template.format(
            patientID=patientID, 
            videoId=videoId, 
            frameId=frameId
        )
        if os.path.exists(pred_path):
            pred = Image.open(pred_path).convert('L')
            pred_tensor = transforms.ToTensor()(pred).unsqueeze(0).cuda()
            if config.get("binarize", True):
                # 根据阈值进行二值化
                pred_tensor[pred_tensor > threshold] = 1
                pred_tensor[pred_tensor <= threshold] = 0
            return pred_tensor
        else:
            raise FileNotFoundError(f"Precomputed prediction not found: {pred_path}")
    else:
        # 对于需要模型推理的方法，使用模型推理
        pred = model(img_norm)["pred"]
        if config.get("binarize", True):
            pred[pred > threshold] = 1
            pred[pred <= threshold] = 0
        return pred

def get_input_image(config, transform, patientID, videoId, frameId):
    """
    根据配置获取输入图像（用于显示）
    """
    # 所有方法都使用相同的输入图像用于显示
    # 对于预计算方法，使用config中指定的input_mode_for_display
    # 对于模型推理方法，使用自己的input_mode
    if config.get("precomputed", False):
        input_mode = config.get("input_mode_for_display", "orig")
    else:
        input_mode = config["input_mode"]
    
    return getImg(input_mode, transform, patientID, videoId, frameId)

def get_image_path_final(config, patientID, videoId, frameId):
    """
    根据配置获取图像路径（最终版）
    """
    if config.get("precomputed", False):
        # 预计算方法的结果路径
        result_path_template = config["result_path_template"]
        return result_path_template.format(
            patientID=patientID, 
            videoId=videoId, 
            frameId=frameId
        )
    else:
        # 需要模型推理的方法的输入图像路径
        input_mode = config["input_mode"]
        return get_image_path(input_mode, patientID, videoId, frameId)

##########################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
from scipy import stats
import datetime
from tqdm import tqdm

def precompute_all_metrics(configs, threshold=0.85, blockCath=False):
    """
    预计算所有方法的指标
    
    返回:
        all_metrics: 字典，键为方法名，值为该方法的指标DataFrame
    """
    # 初始化模型和转换（只有需要模型推理的配置才需要）
    paramPath = "../DeNVeR_in/models_config/freecos_Seg.pt"
    
    # 检查哪些配置需要模型
    need_model = any(not config.get("precomputed", False) for config in configs)
    
    if need_model:
        model = initModel(paramPath)
    else:
        model = None
        
    transform = transforms.Compose([transforms.ToTensor()])
    
    # 获取所有患者和视频
    patient_names = [name for name in os.listdir(datasetPath)
                     if os.path.isdir(os.path.join(datasetPath, name))]
    
    # 计算总图像数量用于进度条
    total_images = 0
    if True:#if only_annotated:
        # 只处理有标注的图片
        for patientID in patient_names:
            patient_path = os.path.join(datasetPath0, patientID, "ground_truth")
            if not os.path.exists(patient_path):
                continue
            video_names = [name for name in os.listdir(patient_path)
                           if os.path.isdir(os.path.join(patient_path, name))]
            for videoId in video_names:
                if len(videoId.split("CATH")) == 1:
                    video_path = os.path.join(datasetPath0, patientID, "ground_truth", videoId)
                    total_images += len(os.listdir(video_path))
    # else:
    #     # 处理所有图片
    #     for patientID in patient_names:
    #         patient_path = os.path.join(datasetPath, patientID, "images")
    #         if not os.path.exists(patient_path):
    #             continue
    #         video_names = [name for name in os.listdir(patient_path)
    #                        if os.path.isdir(os.path.join(patient_path, name))]
    #         for videoId in video_names:
    #             video_path = os.path.join(datasetPath, patientID, "images", videoId)
    #             total_images += len(os.listdir(video_path))
    
    print(f"总图像数量: {total_images}")
    
    # 初始化结果字典
    all_results = {config["name"]: [] for config in configs}
    
    # 创建进度条
    with tqdm(total=total_images, desc="处理图像") as pbar:
        for patientID in patient_names:
            # if only_annotated:
                # 只处理有标注的图片
                patient_gt_path = os.path.join(datasetPath0, patientID, "ground_truth")
                if not os.path.exists(patient_gt_path):
                    continue
                video_names = [name for name in os.listdir(patient_gt_path)
                               if os.path.isdir(os.path.join(patient_gt_path, name))]
                
                for videoId in video_names:
                    if len(videoId.split("CATH")) == 1:
                        video_gt_path = os.path.join(datasetPath0, patientID, "ground_truth", videoId)
                        if not os.path.exists(video_gt_path):
                            continue
                            
                        pbar.set_postfix(patient=patientID, video=videoId, mode="annotated")
                        
                        # 为每个需要模型推理的方法计算均值和标准差
                        norm_params = {}
                        for config in configs:
                            if not config.get("precomputed", False):
                                method_name = config["name"]
                                input_mode = config["input_mode"]
                                norm_method = config["norm_method"]
                                mean, std = norm_method(input_mode, transform, patientID, videoId)
                                norm_params[method_name] = (mean, std)
                        
                        for frameId in os.listdir(video_gt_path):
                            try:
                                # 处理每个方法
                                for config in configs:
                                    method_name = config["name"]
                                    result = process_single_method(
                                        config, patientID, videoId, frameId, 
                                        model, transform, threshold, norm_params.get(method_name, (0, 1)),
                                        blockCath
                                    )
                                    all_results[method_name].append(result)
                                
                            except Exception as e:
                                print(f"处理错误 {patientID}/{videoId}/{frameId}: {str(e)}")
                            pbar.update(1)
            # else:
            #     # 处理所有图片
            #     patient_img_path = os.path.join(datasetPath, patientID, "images")
            #     if not os.path.exists(patient_img_path):
            #         continue
            #     video_names = [name for name in os.listdir(patient_img_path)
            #                    if os.path.isdir(os.path.join(patient_img_path, name))]
                
            #     for videoId in video_names:
            #         video_img_path = os.path.join(datasetPath, patientID, "images", videoId)
            #         if not os.path.exists(video_img_path):
            #             continue
                        
            #         pbar.set_postfix(patient=patientID, video=videoId, mode="all")
                    
            #         # 为每个需要模型推理的方法计算均值和标准差
            #         norm_params = {}
            #         for config in configs:
            #             if not config.get("precomputed", False):
            #                 method_name = config["name"]
            #                 input_mode = config["input_mode"]
            #                 norm_method = config["norm_method"]
            #                 mean, std = norm_method(input_mode, transform, patientID, videoId)
            #                 norm_params[method_name] = (mean, std)
                    
            #         for frameId in os.listdir(video_img_path):
            #             try:
            #                 # 检查是否有对应的ground truth
            #                 gt_path = os.path.join(datasetPath0, patientID, "ground_truth", videoId, frameId)
            #                 has_gt = os.path.exists(gt_path)
                            
            #                 # 处理每个方法
            #                 for config in configs:
            #                     method_name = config["name"]
            #                     result = process_single_method(
            #                         config, patientID, videoId, frameId, 
            #                         model, transform, threshold, norm_params.get(method_name, (0, 1)), has_gt
            #                     )
            #                     all_results[method_name].append(result)
                                
            #             except Exception as e:
            #                 print(f"处理错误 {patientID}/{videoId}/{frameId}: {str(e)}")
            #             pbar.update(1)
    
    # 转换为DataFrame
    all_metrics = {}
    for method_name, results in all_results.items():
        df = pd.DataFrame(results)
        # 只保留有ground truth的数据用于后续分析
        if 'has_ground_truth' in df.columns:
            df = df[df['has_ground_truth'] == True]
        all_metrics[method_name] = df
    
    return all_metrics

def process_single_method(config, patientID, videoId, frameId, model, transform, threshold, norm_params=(0, 1), has_gt=True,blockCath=False):
    """
    处理单个方法在单张图片上的结果
    """
    method_name = config["name"]
    mean, std = norm_params
    
    # 获取输入图像
    input_img = get_input_image(config, transform, patientID, videoId, frameId)
    
    # 归一化处理（只有需要模型推理的配置才需要）
    if not config.get("precomputed", False):
        input_img_norm = (input_img - mean) / std
    else:
        input_img_norm = input_img
    
    # 获取预测结果
    pred = get_prediction_final(config, model, input_img_norm, threshold, patientID, videoId, frameId)
    
    # 获取图像路径
    img_path = get_image_path_final(config, patientID, videoId, frameId)
    
    # 存储结果
    result = {
        'patientID': patientID,
        'videoId': videoId,
        'frameId': frameId,
        'image_id': f"{patientID}_{videoId}_{frameId}",
        'image_path': img_path,
        'has_ground_truth': has_gt
    }
    
    if has_gt:
        # 获取ground truth
        if blockCath: 
            gt_path = os.path.join(datasetPath0, patientID, "ground_truth", videoId+"CATH", frameId)
            gt = Image.open(gt_path).convert('L')
            gt_tensor = transform(gt).unsqueeze(0).cuda()
            # mask_cath = (gt_tensor == 1).to(gt_tensor.dtype)  
            # mask_vessel = (gt_tensor == 0.5).to(gt_tensor.dtype)  
            # mask_cath = (gt_tensor>0.75).to(gt_tensor.dtype)  
            # mask_vessel = (gt_tensor>0.25 and gt_tensor<0.75).to(gt_tensor.dtype)  
            mask_cath = torch.zeros_like(gt_tensor)
            mask_vessel = torch.zeros_like(gt_tensor)
            mask_cath[gt_tensor>0.75]=1
            mask_vessel[(gt_tensor>0.25) & (gt_tensor<0.75)]=1 
            gt_tensor = mask_vessel
            pred_tag1 = pred_tag1 * ( 1 - mask_cath )
            pred_tag2 = pred_tag2 * ( 1 - mask_cath )
        else:
            gt_path = os.path.join(datasetPath0, patientID, "ground_truth", videoId, frameId)
            gt = Image.open(gt_path).convert('L')
            gt_tensor = transform(gt).unsqueeze(0).cuda()
            gt_tensor[gt_tensor > 0.5] = 1
            gt_tensor[gt_tensor <= 0.5] = 0
        
        # 计算指标
        indicators = getIndicators(
            pred[0, 0].detach().cpu() * 255,
            gt_tensor[0, 0].detach().cpu() * 255
        )
        
        result['ground_truth_path'] = gt_path
        
        # 添加指标
        for metric_name, metric_value in indicators.items():
            # 转换为标量值
            if not metric_value.dim() == 0:
                metric_value = metric_value[0]
            result[metric_name] = float(metric_value)
        
        # 存储用于图像对比的数据
        result.update({
            'input_img': input_img.detach().cpu().numpy()[0, 0],
            'pred_img': pred.detach().cpu().numpy()[0, 0],
            'gt_img': gt_tensor.detach().cpu().numpy()[0, 0]
        })
    
    return result

##########################################################################################################

def generate_comparison_matrix(configs, all_metrics, K=1, save_path=""):
    """
    生成M×M对比矩阵图
    每个格子显示两种方法MASK结果的重叠图（彩色）
    """
    M = len(configs)
    method_names = [config["name"] for config in configs]
    
    # 创建图形 - M×M矩阵
    fig, axes = plt.subplots(M, M, figsize=(6*M, 6*M))
    
    # 如果只有一种方法，调整axes的形状
    if M == 1:
        axes = np.array([[axes]])
    
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 10})
    
    # 定义颜色映射
    colors = ['red', 'yellow', 'blue', 'green', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan']
    
    for i in range(M):
        for j in range(M):
            method_i = method_names[i]
            method_j = method_names[j]
            
            if i == j:
                # 对角线：显示方法名称
                axes[i, j].text(0.5, 0.5, method_i, 
                               horizontalalignment='center', verticalalignment='center',
                               fontsize=16, fontweight='bold')
                axes[i, j].set_title(f"{method_i}", fontsize=14)
                axes[i, j].axis('off')
            else:
                # 非对角线：比较两种方法
                df_i = all_metrics[method_i]
                df_j = all_metrics[method_j]
                
                # 合并两个方法的指标
                merged = pd.merge(
                    df_i, df_j, 
                    on=['patientID', 'videoId', 'frameId', 'image_id'], 
                    suffixes=(f'_{method_i}', f'_{method_j}')
                )
                
                # 计算F1差异
                f1_col_i = f'f1_{method_i}'
                f1_col_j = f'f1_{method_j}'
                
                if f1_col_i in merged.columns and f1_col_j in merged.columns:
                    merged['f1_diff'] = merged[f1_col_i] - merged[f1_col_j]
                    
                    # 找出F1差异最大的K张图像
                    top_k = merged.nlargest(K, 'f1_diff')
                    
                    if len(top_k) > 0:
                        # 显示差异最大的图像
                        for k_idx, (_, row) in enumerate(top_k.iterrows()):
                            # 获取预测结果
                            pred_img_i = row[f'pred_img_{method_i}']
                            pred_img_j = row[f'pred_img_{method_j}']
                            
                            # 创建彩色重叠图
                            # 初始化RGB图像
                            h, w = pred_img_i.shape
                            overlap_img = np.zeros((h, w, 3))
                            
                            # 仅方法i的区域：红色
                            mask_i_only = (pred_img_i > 0.5) & (pred_img_j <= 0.5)
                            overlap_img[mask_i_only] = [1, 0, 0]  # 红色
                            
                            # 仅方法j的区域：黄色
                            mask_j_only = (pred_img_i <= 0.5) & (pred_img_j > 0.5)
                            overlap_img[mask_j_only] = [1, 1, 0]  # 黄色
                            
                            # 两个方法都预测的区域：蓝色
                            mask_both = (pred_img_i > 0.5) & (pred_img_j > 0.5)
                            overlap_img[mask_both] = [0, 0, 1]  # 蓝色
                            
                            # 显示重叠图
                            axes[i, j].imshow(overlap_img)
                            
                            # 添加标题和信息
                            video_frame = f"{row['videoId']}/{row['frameId']}"
                            title = f"{method_i} vs {method_j}\nF1 diff: {row['f1_diff']:.3f}\n{video_frame}"
                            axes[i, j].set_title(title, fontsize=10)
                            axes[i, j].axis('off')
                            
                            # 添加图例
                            if k_idx == 0:  # 只在第一张图添加图例
                                # 创建图例元素
                                from matplotlib.patches import Patch
                                legend_elements = [
                                    Patch(facecolor='red', label=f'Only {method_i}'),
                                    Patch(facecolor='yellow', label=f'Only {method_j}'),
                                    Patch(facecolor='blue', label='Both')
                                ]
                                axes[i, j].legend(handles=legend_elements, loc='upper right', fontsize=8)
                    else:
                        axes[i, j].text(0.5, 0.5, "No data", 
                                       horizontalalignment='center', verticalalignment='center')
                        axes[i, j].set_title(f"{method_i} vs {method_j}", fontsize=10)
                        axes[i, j].axis('off')
                else:
                    axes[i, j].text(0.5, 0.5, "No F1 data", 
                                   horizontalalignment='center', verticalalignment='center')
                    axes[i, j].set_title(f"{method_i} vs {method_j}", fontsize=10)
                    axes[i, j].axis('off')
    
    plt.suptitle(f'Method Comparison Matrix (Top {K} Images by F1 Difference)', fontsize=20, y=0.95)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 对比矩阵图已保存: {save_path}")
    plt.close()

def generate_significance_matrix(configs, all_metrics, save_path):
    """
    生成显著性差异矩阵并保存到Excel
    添加一个表格页显示每个方法每个指标的均值和标准差
    """
    M = len(configs)
    method_names = [config["name"] for config in configs]
    
    # 获取所有指标名称
    metrics = []
    for method_name in method_names:
        df = all_metrics[method_name]
        method_metrics = [col for col in df.columns if col not in [
            'patientID', 'videoId', 'frameId', 'image_id', 'image_path', 
            'ground_truth_path', 'has_ground_truth', 'input_img', 'pred_img', 'gt_img'
        ]]
        metrics.extend(method_metrics)
    
    # 去重
    metrics = list(set(metrics))
    
    # 创建Excel写入器
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        # 1. 创建方法指标统计表（均值和标准差）
        stats_data = []
        for method_name in method_names:
            df = all_metrics[method_name]
            for metric in metrics:
                if metric in df.columns:
                    values = df[metric].dropna()
                    if len(values) > 0:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        stats_data.append({
                            'Method': method_name,
                            'Metric': metric,
                            'Mean': mean_val,
                            'Std': std_val,
                            'Count': len(values)
                        })
        
        stats_df = pd.DataFrame(stats_data)
        if not stats_df.empty:
            # 重新排列数据，使每个方法一行
            pivot_stats = stats_df.pivot_table(
                index='Method', 
                columns='Metric', 
                values=['Mean', 'Std'],
                aggfunc='first'
            )
            # 扁平化多级列索引
            pivot_stats.columns = [f'{col[0]}_{col[1]}' for col in pivot_stats.columns]
            pivot_stats.reset_index(inplace=True)
            pivot_stats.to_excel(writer, sheet_name='Method_Statistics', index=False)
        else:
            # 如果没有数据，创建空表
            empty_stats = pd.DataFrame(columns=['Method', 'Metric', 'Mean', 'Std', 'Count'])
            empty_stats.to_excel(writer, sheet_name='Method_Statistics', index=False)
        
        # 2. 为每个指标创建显著性矩阵
        for metric in metrics:
            # 创建M×M矩阵
            pvalue_matrix = pd.DataFrame(index=method_names, columns=method_names)
            mean_diff_matrix = pd.DataFrame(index=method_names, columns=method_names)
            significance_matrix = pd.DataFrame(index=method_names, columns=method_names)
            
            for i, method_i in enumerate(method_names):
                for j, method_j in enumerate(method_names):
                    if i == j:
                        pvalue_matrix.loc[method_i, method_j] = np.nan
                        mean_diff_matrix.loc[method_i, method_j] = np.nan
                        significance_matrix.loc[method_i, method_j] = "-"
                    else:
                        df_i = all_metrics[method_i]
                        df_j = all_metrics[method_j]
                        
                        # 合并数据
                        merged = pd.merge(
                            df_i, df_j, 
                            on=['patientID', 'videoId', 'frameId', 'image_id'], 
                            suffixes=(f'_{method_i}', f'_{method_j}')
                        )
                        
                        metric_col_i = f'{metric}_{method_i}'
                        metric_col_j = f'{metric}_{method_j}'
                        
                        if metric_col_i in merged.columns and metric_col_j in merged.columns:
                            data_i = merged[metric_col_i].values
                            data_j = merged[metric_col_j].values
                            
                            # 移除NaN值
                            mask = ~(np.isnan(data_i) | np.isnan(data_j))
                            data_i_clean = data_i[mask]
                            data_j_clean = data_j[mask]
                            
                            if len(data_i_clean) >= 2:
                                # 执行配对t检验
                                t_stat, p_value = stats.ttest_rel(data_i_clean, data_j_clean)
                                mean_diff = np.mean(data_i_clean) - np.mean(data_j_clean)
                                
                                pvalue_matrix.loc[method_i, method_j] = p_value
                                mean_diff_matrix.loc[method_i, method_j] = mean_diff
                                
                                # 判断显著性
                                if p_value < 0.001:
                                    significance_matrix.loc[method_i, method_j] = "***"
                                elif p_value < 0.01:
                                    significance_matrix.loc[method_i, method_j] = "**"
                                elif p_value < 0.05:
                                    significance_matrix.loc[method_i, method_j] = "*"
                                else:
                                    significance_matrix.loc[method_i, method_j] = "ns"
                            else:
                                pvalue_matrix.loc[method_i, method_j] = np.nan
                                mean_diff_matrix.loc[method_i, method_j] = np.nan
                                significance_matrix.loc[method_i, method_j] = "NA"
                        else:
                            pvalue_matrix.loc[method_i, method_j] = np.nan
                            mean_diff_matrix.loc[method_i, method_j] = np.nan
                            significance_matrix.loc[method_i, method_j] = "NA"
            
            # 保存到Excel的不同sheet
            pvalue_matrix.to_excel(writer, sheet_name=f'{metric}_pvalue')
            mean_diff_matrix.to_excel(writer, sheet_name=f'{metric}_mean_diff')
            significance_matrix.to_excel(writer, sheet_name=f'{metric}_significance')
    
    print(f"✓ 显著性差异矩阵已保存: {save_path}")

def comprehensive_multi_comparison_analysis(configs, K=1, threshold=0.85, save_path="", blockCath=True):
    """
    对多个方法进行全面的两两比较分析
    
    参数:
        configs: 方法配置列表
        K: 每个对比中显示的图像数量
        threshold: 分割阈值
        save_path: 保存路径
        only_annotated: 是否只分割有人工标注的图片
    """
    M = len(configs)
    method_names = [config["name"] for config in configs]
    
    print(f"开始全面的多方法比较分析: {M}个方法")
    print(f"方法列表: {method_names}")
    print(f"模式: {'遮挡标签中的导管' if blockCath else '不遮挡导管'}")
    print("=" * 60)
    
    # 设置默认保存路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_hasCath" if blockCath else "_nonCath"
    
    # 步骤1: 预计算所有方法的指标
    print("步骤1/3: 预计算所有方法的指标...")
    all_metrics = precompute_all_metrics(configs, threshold, blockCath)
    
    # 步骤2: 生成M×M对比矩阵图
    print("步骤2/3: 生成M×M对比矩阵图...")
    matrix_img_path = f"{save_path}method_comparison_matrix{mode_suffix}_{timestamp}.jpg"
    generate_comparison_matrix(configs, all_metrics, K, matrix_img_path)
    
    # 步骤3: 生成显著性差异矩阵并保存到Excel
    print("步骤3/3: 生成显著性差异矩阵...")
    excel_path = f"{save_path}method_comparison_significance{mode_suffix}_{timestamp}.xlsx"
    generate_significance_matrix(configs, all_metrics, excel_path)
    
    # 步骤4: 打印方法统计摘要
    print_method_statistics(configs, all_metrics)
    
    print("\n" + "=" * 60)
    print("多方法比较分析完成!")
    print(f"✓ 对比矩阵图已保存: {matrix_img_path}")
    print(f"✓ 显著性差异矩阵已保存: {excel_path}")
    
    return all_metrics

def print_method_statistics(configs, all_metrics):
    """
    打印方法统计摘要
    """
    print("\n方法统计摘要:")
    print("-" * 50)
    
    for config in configs:
        method_name = config["name"]
        df = all_metrics[method_name]
        
        print(f"\n{method_name}:")
        print(f"  有效图像数量: {len(df)}")
        
        # 计算主要指标的均值和标准差
        metrics_to_show = ['f1', 'precision', 'recall', 'dice', 'iou']
        for metric in metrics_to_show:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")

# # 修改主函数调用
# if __name__ == "__main__":
#     topK = 1  # 每个对比显示1张图像
#     threshold = 0.5
#     only_annotated = True
    
#     # 打开并读取 YAML 文件
#     with open(file_path, 'r', encoding='utf-8') as file:
#         config = yaml.safe_load(file)
#         print("notes:",config["my"]["notes"])
#         rootPath = config["my"]["filePathRoot"]
    
#     # 定义多个方法的配置
#     configs = [
#         {
#             "name": "noRigid1_NewNorm",
#             "precomputed": False,
#             "input_mode": "noRigid1",
#             "norm_method": calculate_mean_variance,
#             "binarize": True
#         },
#         {
#             "name": "planar", 
#             "precomputed": True,
#             "result_path_template": os.path.join("output", "{videoId}", "planar", "{frameId}"),
#             "input_mode_for_display": "orig",
#             "binarize": True
#         },
#         {
#             "name": "orig",
#             "precomputed": False,
#             "input_mode": "orig", 
#             "norm_method": calculate_mean_variance,
#             "binarize": True
#         },
#         # 可以继续添加更多方法...
#     ]
    
#     # 执行多方法比较
#     all_metrics = comprehensive_multi_comparison_analysis(
#         configs=configs, K=topK, threshold=threshold, 
#         save_path=rootPath+"/", only_annotated=only_annotated
#     )
    
#     print(f"\n多方法比较完成!")
#     print(f"参与比较的方法: {[config['name'] for config in configs]}")

##########################################################################################################

# 可用的TAG列表
AVAILABLE_TAGS = [
    "orig", "fluid", "fluid2", "noRigid1", "noRigid2", 
    "mix", "mix2", "mix4", "mix5", "mix6", "pred"
]

def denoising():
# if False:#
# if __name__ == "__main__":
    ROOT1 = os.path.dirname(script_path)
    file_path = os.path.join(ROOT1, "../",'confs/newConfig.yaml')
    
    # 进度文件路径
    progress_file = os.path.join(ROOT1, "progress_newBatch.json")
    # 加载进度文件
    processed_videos = set()
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                processed_videos = set(progress_data.get("processed_videos", []))
                print(f"加载进度文件，已处理 {len(processed_videos)} 个视频")
        except Exception as e:
            print(f"加载进度文件失败: {e}，将重新开始处理")
    
    # 打开并读取 YAML 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        print("notes:",config["my"]["notes"])
        rootPath = config["my"]["filePathRoot"]
        datasetPath0=config["my"]["datasetPath"]

        # datasetPath="../DeNVeR_in/xca_dataset"
        paramPath = "../DeNVeR_in/models_config/freecos_Seg.pt"
        patient_names = [name for name in os.listdir(datasetPath0)
                    if os.path.isdir(os.path.join(datasetPath0, name))]
        CountSum = 0
        for patientID in patient_names:
            patient_path = os.path.join(datasetPath0, patientID, "images")
            video_names = [name for name in os.listdir(patient_path)
                        if os.path.isdir(os.path.join(patient_path, name))]
            CountSum = CountSum + len(video_names)
        CountI = len(processed_videos)  # 从已处理的数量开始计数
        
        print(f"总视频数: {CountSum}, 已处理: {CountI}, 待处理: {CountSum - CountI}")
        
        for patientID in patient_names:
            patient_path = os.path.join(datasetPath0, patientID, "images")
            video_names = [name for name in os.listdir(patient_path)
                        if os.path.isdir(os.path.join(patient_path, name))]
            for videoId in video_names:
                # 生成唯一标识符
                video_key = f"{patientID}/{videoId}"
                
                # 如果已经处理过，则跳过
                if video_key in processed_videos:
                    continue
                
                inpath = os.path.join(datasetPath0, patientID, "images", videoId)
                outpath = os.path.join(datasetPath0, patientID, "decouple", videoId)#数据集路径
                outpath = os.path.join(datasetPath0, patientID, "decouple", videoId)#本地路径
                print("outpath:",outpath)
                os.makedirs(outpath, exist_ok=True)
                
                startDecouple1(videoId, paramPath, inpath, outpath)  # 去除刚体层
                # startDecouple1(videoId, paramPath, inpath, outpath)  # 去除刚体层
                # startDecouple3(videoId, paramPath, inpath, outpath)  # 获取流体层
                    
                # 处理成功，更新进度
                CountI += 1
                processed_videos.add(video_key)
                    
                # 更新进度文件
                progress_data = {"processed_videos": list(processed_videos)}
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(progress_data, f, ensure_ascii=False, indent=2)    
                print(f"{CountI}/{CountSum} {videoId} - 已完成")
                    
# 修改主函数调用
if __name__ == "__main__":
    topK = 1  # 每个对比显示1张图像
    threshold = 0.5
    blockCath = True
    
    # 打开并读取 YAML 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        print("notes:",config["my"]["notes"])
        rootPath = config["my"]["filePathRoot"]
    
    # 定义多个方法的配置
    configs = [
        {
            "name": "1.masks", 
            "precomputed": True,  # 预计算方法
            "result_path_template": os.path.join("../DeNVeR.006/log_6/outputs", "{videoId}", "1.masks", "{frameId}"),
            # 1.masks   2.2.planar  3.parallel  4.deform    5.refine
            "input_mode_for_display": "orig",  # 用于显示输入图像的模式
            "binarize": True  # 对分割结果进行二值化
        },
        {
            "name": "2.planar", 
            "precomputed": True,  # 预计算方法
            "result_path_template": os.path.join("../DeNVeR.006/log_6/outputs", "{videoId}", "2.2.planar", "{frameId}"),
            # 1.masks   2.2.planar  3.parallel  4.deform    5.refine
            "input_mode_for_display": "orig",  # 用于显示输入图像的模式
            "binarize": True  # 对分割结果进行二值化
        },
        {
            "name": "3.parallel", 
            "precomputed": True,  # 预计算方法
            "result_path_template": os.path.join("../DeNVeR.006/log_6/outputs", "{videoId}", "3.parallel", "{frameId}"),
            # 1.masks   2.2.planar  3.parallel  4.deform    5.refine
            "input_mode_for_display": "orig",  # 用于显示输入图像的模式
            "binarize": True  # 对分割结果进行二值化
        },
        {
            "name": "4.deform", 
            "precomputed": True,  # 预计算方法
            "result_path_template": os.path.join("../DeNVeR.006/log_6/outputs", "{videoId}", "4.deform", "{frameId}"),
            # 1.masks   2.2.planar  3.parallel  4.deform    5.refine
            "input_mode_for_display": "orig",  # 用于显示输入图像的模式
            "binarize": True  # 对分割结果进行二值化
        },
        {
            "name": "5.refine", 
            "precomputed": True,  # 预计算方法
            "result_path_template": os.path.join("../DeNVeR.006/log_6/outputs", "{videoId}", "5.refine", "{frameId}"),
            # 1.masks   2.2.planar  3.parallel  4.deform    5.refine
            "input_mode_for_display": "orig",  # 用于显示输入图像的模式
            "binarize": True  # 对分割结果进行二值化
        },
        # 可以继续添加更多方法...
    ]
    
    # 执行多方法比较
    all_metrics = comprehensive_multi_comparison_analysis(
        configs=configs, K=topK, threshold=threshold, 
        save_path=rootPath+"/", 
        blockCath=blockCath # only_annotated => blockCath 是否屏蔽导管区域
    )
    
    print(f"\n多方法比较完成!")
    print(f"参与比较的方法: {[config['name'] for config in configs]}")

'''
    source activate DNVR
    python -m nir.new_batch_topK
'''