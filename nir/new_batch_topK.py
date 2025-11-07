import os
from nir.new import startDecouple1,startDecouple3

from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torch
from free_cos.newTrain import initCSV, save2CVS, getIndicators
from nir.analysisOri import getModel
script_path = os.path.abspath(__file__)
ROOT = os.path.dirname(script_path)
datasetPath=os.path.join(ROOT,"../","../DeNVeR_in/xca_dataset")
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
def calculate_mean_varianceOld(TAG,transform, patientID, videoId):
    img_list = []
    path0 = os.path.join(datasetPath, patientID, "decouple", videoId, "orig")
    # print(path0,len(os.path.join(path0)))
    for frameId in os.listdir(os.path.join(path0)):
        # frameId = str(i).zfill(5)+".png"
        img = getImg(TAG,transform, patientID, videoId, frameId)
        img_list.append(img)
    imgAll = torch.cat(img_list, dim=0)
    if True:

        import numpy as np
        # 假设 tensor 是 [10, 3, 256, 256]
        flat = imgAll.detach().cpu().numpy().flatten()
        # 计算 5% 和 95% 分位数
        q_min = np.percentile(flat, 5)
        q_max = np.percentile(flat, 95)

        # 构建 mask
        mask = (imgAll >= q_min) & (imgAll <= q_max)
        imgAll = imgAll[mask]

    mean0 = imgAll.mean()
    std0 = imgAll.std()
    return mean0, std0
import numpy as np
def calculate_mean_variance(TAG,transform, patientID, videoId):
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


def save_comparison_figure(results, K, tag1, tag2, save_path):
    """
    保存对比图为JPG文件：5行K列
    行1: tag1输入图像
    行2: tag1分割结果
    行3: tag2输入图像  
    行4: tag2分割结果
    行5: ground truth
    """
    print(f"创建对比图，包含 {K} 张图像...")
    
    # 创建图形 - 根据K的大小调整图形尺寸
    fig_width = max(4 * K, 12)  # 最小宽度为12
    fig_height = 15  # 增加高度以容纳第5行
    
    # 创建图形和子图 - 5行K列
    fig, axes = plt.subplots(5, K, figsize=(fig_width, fig_height))
    
    # 如果只有一张图，调整axes的形状
    if K == 1:
        axes = axes.reshape(5, 1)
    
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
    
    # 添加总标题
    plt.suptitle(f'Top {K} Images Where {tag1} Outperforms {tag2} (F1 Difference)', fontsize=16, y=0.98)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # 为总标题留出空间
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='jpg')
    print(f"✓ 对比图像已保存: {save_path}")
    
    # 关闭图形以释放内存
    plt.close(fig)
    
    # # 打印详细信息
    # print(f"\n{tag1} 优于 {tag2} 的前 {K} 张图像:")
    # print("=" * 80)
    # for i, result in enumerate(results):
    #     print(f"{i+1}. {result['patientID']}/{result['videoId']}/{result['frameId']}")
    #     print(f"   {tag1} F1: {result['f1_tag1']:.4f}, {tag2} F1: {result['f1_tag2']:.4f}, "
    #           f"差异: {result['f1_diff']:.4f}")
    # print("=" * 80)

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
        diff_data = df[['patientID', 'videoId', 'frameId', 'image_id']].copy()
        
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
    
    print(f"✓ 统计结果已保存: {save_path}")
    print(f"Excel文件包含以下工作表:")
    print(f"  1. 原始数据 - 所有图像的详细指标")
    print(f"  2. 统计摘要 - 详细的统计检验结果")
    print(f"  3. 差异数据 - 两种方法的差异值")
    print(f"  4. 方法比较 - 简明的比较结果")


def comprehensive_comparison_analysis(tag1, tag2, K=5, threshold=0.85, 
                                      save_path=""):
                                    #  save_img_path=None, save_excel_path=None):
    """
    对两种方法进行全面的比较分析，同时保存对比图像和Excel统计结果
    
    参数:
        tag1: 第一种方法标签
        tag2: 第二种方法标签
        K: 对比图中显示的图像数量
        threshold: 分割阈值
        save_img_path: 对比图像保存路径
        save_excel_path: Excel文件保存路径
    """
    print(f"开始全面的比较分析: {tag1} vs {tag2}")
    print("=" * 60)
    
    # 设置默认保存路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # if save_img_path is None:
    #     save_img_path = f"{tag1}_vs_{tag2}_comparison_{timestamp}.jpg"
    # if save_excel_path is None:
    #     save_excel_path = f"{tag1}_vs_{tag2}_statistical_analysis_{timestamp}.xlsx"
    save_img_path = f"{save_path}/{tag1}_vs_{tag2}_comparison_{timestamp}.jpg"
    save_excel_path = f"{save_path}/{tag1}_vs_{tag2}_statistical_analysis_{timestamp}.xlsx"
    
    # 步骤1: 收集所有图像的指标数据
    print("步骤1/4: 收集所有图像的指标数据...")
    df, top_k_results = collect_all_metrics(tag1, tag2, K, threshold)
    
    # 步骤2: 执行统计检验
    print("步骤2/4: 执行统计检验...")
    t_test_results = perform_paired_t_test_on_dataframe(df, tag1, tag2)
    
    # 步骤3: 保存对比图像
    print("步骤3/4: 保存对比图像...")
    save_comparison_figure(top_k_results, K, tag1, tag2, save_img_path)
    
    # 步骤4: 保存Excel统计结果
    print("步骤4/4: 保存Excel统计结果...")
    save_results_to_excel(df, t_test_results, tag1, tag2, save_excel_path)
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print(f"✓ 对比图像已保存: {save_img_path}")
    print(f"✓ 统计结果已保存: {save_excel_path}")
    
    return df, t_test_results, top_k_results

def collect_all_metrics(tag1, tag2, K=5, threshold=0.85):
    """
    收集所有图像的指标数据，并找出F1差异最大的K张图像
    
    返回:
        df: 包含所有图像指标的DataFrame
        top_k_results: F1差异最大的K张图像的结果
    """
    # 初始化模型和转换
    paramPath = "../DeNVeR_in/models_config/freecos_Seg.pt"
    model = initModel(paramPath)
    transform = transforms.Compose([transforms.ToTensor()])
    
    # 获取所有患者和视频
    patient_names = [name for name in os.listdir(datasetPath)
                     if os.path.isdir(os.path.join(datasetPath, name))]
    
    # 存储所有图像的指标结果
    all_results = []
    # 专门存储用于图像对比的结果
    comparison_results = []
    
    # 计算总图像数量用于进度条
    total_images = 0
    for patientID in patient_names:
        patient_path = os.path.join(datasetPath, patientID, "ground_truth")
        video_names = [name for name in os.listdir(patient_path)
                       if os.path.isdir(os.path.join(patient_path, name))]
        for videoId in video_names:
            if len(videoId.split("CATH")) == 1:
                video_path = os.path.join(datasetPath, patientID, "ground_truth", videoId)
                total_images += len(os.listdir(video_path))
    
    print(f"总图像数量: {total_images}")
    
    # 创建进度条
    with tqdm(total=total_images, desc="处理图像") as pbar:
        for patientID in patient_names:
            patient_path = os.path.join(datasetPath, patientID, "ground_truth")
            video_names = [name for name in os.listdir(patient_path)
                           if os.path.isdir(os.path.join(patient_path, name))]
            
            for videoId in video_names:
                if len(videoId.split("CATH")) == 1:
                    video_path = os.path.join(datasetPath, patientID, "ground_truth", videoId)
                    
                    # 更新进度条描述
                    pbar.set_postfix(patient=patientID, video=videoId)
                    
                    # 计算该视频的均值和标准差
                    mean_tag1, std_tag1 = calculate_mean_variance(tag1, transform, patientID, videoId)
                    mean_tag2, std_tag2 = calculate_mean_variance(tag2, transform, patientID, videoId)
                    
                    for frameId in os.listdir(video_path):
                        try:
                            # 计算tag1的指标
                            img_tag1 = getImg(tag1, transform, patientID, videoId, frameId)
                            img_tag1_norm = (img_tag1 - mean_tag1) / std_tag1
                            pred_tag1 = model(img_tag1_norm)["pred"]
                            pred_tag1[pred_tag1 > threshold] = 1
                            pred_tag1[pred_tag1 <= threshold] = 0
                            
                            # 计算tag2的指标
                            img_tag2 = getImg(tag2, transform, patientID, videoId, frameId)
                            img_tag2_norm = (img_tag2 - mean_tag2) / std_tag2
                            pred_tag2 = model(img_tag2_norm)["pred"]
                            pred_tag2[pred_tag2 > threshold] = 1
                            pred_tag2[pred_tag2 <= threshold] = 0
                            
                            # 获取ground truth
                            gt_path = os.path.join(datasetPath, patientID, "ground_truth", videoId, frameId)
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
                            
                            # 存储用于统计的数据
                            result = {
                                'patientID': patientID,
                                'videoId': videoId,
                                'frameId': frameId,
                                'image_id': f"{patientID}_{videoId}_{frameId}"
                            }
                            
                            # 添加tag1的指标
                            for metric_name, metric_value in ind_tag1.items():
                                result[f'{tag1}_{metric_name}'] = float(metric_value[0] if not metric_value.dim() == 0 else metric_value)
                            
                            # 添加tag2的指标
                            for metric_name, metric_value in ind_tag2.items():
                                result[f'{tag2}_{metric_name}'] = float(metric_value[0] if not metric_value.dim() == 0 else metric_value)
                            
                            all_results.append(result)
                            
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
                                'gt': gt_tensor.detach().cpu().numpy()[0, 0]
                            }
                            comparison_results.append(comparison_result)
                            
                        except Exception as e:
                            print(f"处理错误 {patientID}/{videoId}/{frameId}: {str(e)}")
                        
                        # 更新进度条
                        pbar.update(1)
    
    # 转换为DataFrame
    df = pd.DataFrame(all_results)
    
    # 找出F1差异最大的K张图像
    top_k_results = sorted(comparison_results, key=lambda x: x['f1_diff'], reverse=True)[:K]
    
    return df, top_k_results

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

# # 使用示例
# if __name__ == "__main__":
#     # 示例1: 比较fluid和orig，同时保存图像和Excel
#     df, results, top_k = comprehensive_comparison_analysis(
#         tag1="fluid", 
#         tag2="orig", 
#         K=5, 
#         threshold=0.85,
#         save_img_path="fluid_vs_orig_comparison.jpg",
#         save_excel_path="fluid_vs_orig_statistical_analysis.xlsx"
#     )
    
#     # 示例2: 比较其他方法，使用默认保存路径
#     df2, results2, top_k2 = comprehensive_comparison_analysis(
#         tag1="noRigid1", 
#         tag2="noRigid2", 
#         K=5, 
#         threshold=0.85
#     )
##########################################################################################################

# 可用的TAG列表
AVAILABLE_TAGS = [
    "orig", "fluid", "fluid2", "noRigid1", "noRigid2", 
    "mix", "mix2", "mix4", "mix5", "mix6", "pred"
]

def list_available_tags():
    """列出所有可用的TAG"""
    print("Available tags for comparison:")
    for i, tag in enumerate(AVAILABLE_TAGS):
        print(f"  {i+1}. {tag}")
    return AVAILABLE_TAGS

# 使用示例
import yaml
if __name__ == "__main__":
    # 列出所有可用的TAG
    # available_tags = list_available_tags()

    ROOT1 = os.path.dirname(script_path)
    file_path = os.path.join(ROOT1, "../",'confs/newConfig.yaml')
    topK = 10
    threshold = 0.5 # 0.85
    # 打开并读取 YAML 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        print("notes:",config["my"]["notes"])
        rootPath = config["my"]["filePathRoot"]
    
        # 示例1: 比较fluid和orig
        print("\nExample 1: Comparing fluid vs orig")
        comprehensive_comparison_analysis(tag1="fluid", tag2="orig", K=topK, threshold=threshold, 
                                    save_path=rootPath+"/")
        comprehensive_comparison_analysis(tag1="orig", tag2="fluid", K=topK, threshold=threshold, 
                                    save_path=rootPath+"/")
        
        # 示例2: 比较noRigid1和noRigid2
        print("\nExample 2: Comparing noRigid1 vs fluid")
        comprehensive_comparison_analysis(tag1="noRigid1", tag2="fluid", K=topK, threshold=threshold,
                                    save_path=rootPath+"/")
        comprehensive_comparison_analysis(tag1="fluid", tag2="noRigid1", K=topK, threshold=threshold,
                                    save_path=rootPath+"/")
        
        # # 示例3: 比较noRigid1和noRigid2
        # print("\nExample 3: Comparing noRigid1 vs orig")
        # results2 = compare_two_methods("noRigid1", "orig", K=topK, threshold=threshold,
        #                             save_path=rootPath+"/"+"noRigid1_vs_orig_comparison.jpg")
        # results2 = compare_two_methods("orig", "noRigid1", K=topK, threshold=threshold,
        #                             save_path=rootPath+"/"+"orig_vs_noRigid1_comparison.jpg")


