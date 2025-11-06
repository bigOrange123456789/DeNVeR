import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from free_cos.ModelSegment import ModelSegment
from free_cos50.Datasetloader.XCAD_liot import DatasetXCAD_aug
###############################################################################################################
###############################################################################################################

def getModel(pathParam):
    os.environ['MASTER_PORT'] = '169711' #“master_port”的意思是主端口
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    cudnn.benchmark = True #benchmark的意思是基准

    n_channels = 1
    num_classes =  1
    Segment_model = ModelSegment(n_channels, num_classes)

    if torch.cuda.is_available():
        Segment_model = Segment_model.cuda() # 分割模型

    checkpoint = torch.load(pathParam)  # 如果模型是在GPU上训练的，这里指定为'cpu'以确保兼容性
    Segment_model.load_state_dict(checkpoint['state_dict'])
    return Segment_model

def getModel50(pathParam):
    os.environ['MASTER_PORT'] = '169711' #“master_port”的意思是主端口
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    cudnn.benchmark = True #benchmark的意思是基准

    n_channels = 4
    num_classes =  1
    Segment_model = ModelSegment(n_channels, num_classes)

    if torch.cuda.is_available():
        Segment_model = Segment_model.cuda() # 分割模型

    checkpoint = torch.load(pathParam)  # 如果模型是在GPU上训练的，这里指定为'cpu'以确保兼容性
    Segment_model.load_state_dict(checkpoint['state_dict'])
    return Segment_model


from free_cos.newTrain import initCSV, save2CVS, getIndicators

def evaluate2(pathIn, pathGt, pathOut, model,deCatheter=False):#计算AP
    dset = DatasetXCAD_aug(pathIn, pathGt)
    from torch.utils.data import DataLoader
    loader=DataLoader(
        dset, 
        batch_size=1,  # bsz, 
        shuffle=False, # shuffle, 
        num_workers=8  # nworker
    )
    for val_idx, minibatch in enumerate(loader):
        if torch.isnan(torch.sum(minibatch["img"])):
            print("isnan:",val_idx)
            continue
        val_imgs = minibatch['img']  # 图片的梯度数据
        val_imgs = val_imgs.cuda(non_blocking=True)  # NCHW
        # result = self.Segment_model(val_imgs, mask=None, trained=False, fake=False)
        print("val_imgs",type(val_imgs))
        print(val_imgs.shape)
        print(val_idx)
        exit(0)
    # DataLoader(
    #     dset,
    #     batch_size=batch_size, #指定每个批次的样本数量
    #     num_workers=num_workers, #指定加载数据时使用的子进程数量
    #     pin_memory=persistent_workers, #是否将数据加载到 GPU 的 Pin Memory 中
    #     shuffle=False, #是否在每个 epoch 开始时随机打乱数据
    #     persistent_workers=persistent_workers, #是否在 DataLoader 的生命周期内保持子进程的活动状态
    # )
    ################################################################################################
    os.makedirs(os.path.join(pathOut), exist_ok=True)
    head = ["fileName", "accuracy", "recall", "precision", "f1", "iou", "specificity"]
    path = os.path.join(pathOut, "experiment_results.csv")
    initCSV(path, head)
    model.eval()
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
    ])
    
    datasetPath = pathIn
    
    ############################################ 计算指标 ############################################
    threshold = 0.5
    0.85
    print("阈值为:",threshold)
    sum_recall = 0
    sum_precision = 0
    sum_f1 = 0
    list_recall = []
    list_precision = []
    list_f1 = []
    img_list = []
    
    # 收集所有图像用于标准化
    from PIL import Image
    for name in os.listdir(os.path.join(datasetPath)):
        path_img = os.path.join(datasetPath, name)
        img = Image.open(path_img).convert('L')
        img = transform(img).unsqueeze(0).cuda()
        img_list.append(img)
    img_list = torch.cat(img_list, dim=0)
    
    # 用于存储所有预测概率和真实标签
    all_pred_probs = []
    all_true_labels = []
    
    for name in os.listdir(os.path.join(datasetPath)):
        path_img = os.path.join(datasetPath, name)
        img = Image.open(path_img).convert('L')
        img = transform(img).unsqueeze(0).cuda()
        img = (img - img_list.mean()) / img_list.std()
        
        # 获取原始预测概率（不进行二值化）
        pred_prob = model(img)["pred"]
        
        # 保存预测概率
        all_pred_probs.append(pred_prob.detach().cpu().numpy().flatten())
        
        # 二值化预测用于指标计算
        pred = pred_prob.clone()
        pred[pred >= threshold] = 1
        pred[pred < threshold] = 0
        if deCatheter:
            from preprocess.mySkeleton import getCatheter
            catheter = getCatheter(pred[0,0].detach().cpu().numpy())
            catheter = torch.from_numpy(catheter).unsqueeze(0).unsqueeze(0)
            # pred = (1-catheter) * pred
            pred[catheter]=0#导管处为背景
        
        path_gt = os.path.join(pathGt, name)
        gt = Image.open(path_gt).convert('L')
        gt = transform(gt).unsqueeze(0).cuda()
        gt[gt >= threshold] = 1
        gt[gt < threshold] = 0
        
        # 保存真实标签
        all_true_labels.append(gt.detach().cpu().numpy().flatten())
        
        ind = getIndicators(
            pred[0, 0].detach().cpu() * 255,
            gt[0, 0].detach().cpu() * 255
        )
        ind["fileName"] = name.split(".png")[0]
        save2CVS(path, head, ind)
        sum_recall += ind["recall"]
        sum_precision += ind["precision"]
        sum_f1 += ind["f1"]

        list_recall.append(ind["recall"])
        list_precision.append(ind["precision"])
        list_f1.append(ind["f1"])
    
    # 计算平均指标
    avg_f1 = sum_f1.item() / len(os.listdir(datasetPath))
    avg_precision = sum_precision.item() / len(os.listdir(datasetPath))
    avg_recall = sum_recall.item() / len(os.listdir(datasetPath))
    
    print("f1:", avg_f1,
          "pr:", avg_precision,
          "sn:", avg_recall)
    # 计算平均值
    average_recall = np.mean(list_recall)
    average_precision = np.mean(list_precision)
    average_f1_score = np.mean(list_f1)
    # 计算标准差
    std_recall = np.std(list_recall)
    std_precision = np.std(list_precision)
    std_f1_score = np.std(list_f1)
    print(f"Recall: {average_recall:.4f} +- {std_recall:.4f}")         #Sn
    print(f"Precision: {average_precision:.4f} +- {std_precision:.4f}")#Pr
    print(f"F1 Score: {average_f1_score:.4f} +- {std_f1_score:.4f}")   #Dice
    

    # 计算PR曲线和AP值
    all_pred_probs = np.concatenate(all_pred_probs)
    all_true_labels = np.concatenate(all_true_labels)
    
    # 确保标签是二值的
    all_true_labels = (all_true_labels > 0.5).astype(np.int32)
    
    ##################################################################################################
    # 计算精确率、召回率和阈值
    from PIL import Image
    from sklearn.metrics import precision_recall_curve, average_precision_score
    import pandas as pd
    from torchvision import transforms
    precision, recall, thresholds = precision_recall_curve(all_true_labels, all_pred_probs)
    
    # 计算平均精确度 (AP) - PR曲线下面积
    ap_score = average_precision_score(all_true_labels, all_pred_probs)
    print(f"Average Precision (AP) Score: {ap_score:.4f}")
    
    # # 绘制Precision-Recall曲线
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, marker='.', label=f'AP = {ap_score:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(pathOut, 'precision_recall_curve.png'))
    plt.close()
    
    # 找到最佳阈值（基于F1分数）
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"最佳阈值: {best_threshold:.4f}")
    print(f"在此阈值下的精确率: {best_precision:.4f}")
    print(f"在此阈值下的召回率: {best_recall:.4f}")
    print(f"在此阈值下的F1分数: {best_f1:.4f}")
    
    # 保存PR曲线数据
    pr_data = pd.DataFrame({
        'Recall': recall,
        'Precision': precision,
        'Thresholds': np.append(thresholds, np.nan)  # 添加一个NaN以匹配长度
    })
    pr_data.to_csv(os.path.join(pathOut, 'pr_curve_data.analysisOri.csv'), index=False)
    
    # 保存AP值和最佳阈值信息
    with open(os.path.join(pathOut, 'ap_results.txt'), 'w') as f:
        f.write(f"Recall: {average_recall:.4f} +- {std_recall:.4f}")         #Sn
        f.write(f"Precision: {average_precision:.4f} +- {std_precision:.4f}")#Pr
        f.write(f"F1 Score: {average_f1_score:.4f} +- {std_f1_score:.4f}")   #Dice
        f.write(f"Average Precision (AP) Score: {ap_score:.4f}\n")
        f.write(f"最佳阈值: {best_threshold:.4f}\n")
        f.write(f"在此阈值下的精确率: {best_precision:.4f}\n")
        f.write(f"在此阈值下的召回率: {best_recall:.4f}\n")
        f.write(f"在此阈值下的F1分数: {best_f1:.4f}\n")
        f.write(f"平均精确率: {avg_precision:.4f}\n")
        f.write(f"平均召回率: {avg_recall:.4f}\n")
        f.write(f"平均F1分数: {avg_f1:.4f}\n")
    
    return {
        'ap_score': ap_score,
        'best_threshold': best_threshold,
        'best_precision': best_precision,
        'best_recall': best_recall,
        'best_f1': best_f1,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1
    }

if __name__ == "__main__":
    print("version:2025.09.15.1517")
    paramPath = "../DeNVeR_in/models_config/freecos_Seg.pt"
    paramPath = "../../FreeCOS-GuangYuan.02_57/FreeCOS-GuangYuan46/logs/FreeCOS46.log_real/best_Segment.pt"
    # paramPath = "./nir/best_Segment.pt"
    model_my46 = getModel(paramPath)
    # print("\n一、...分形合成、XACV数据集...")
    # gtpath = "../DeNVeR_in/xca_dataset_video/gt"
    # inpath_a = "../DeNVeR_in/xca_dataset_video/img"
    # outpath = "../DeNVeR_in/xca_dataset_video/pred.freecos1"
    # evaluate2(inpath_a, gtpath, outpath, model_my46)
    
    # print("\n二、...分形合成、XCAD数据集...")
    # gtpath = "../../DataSet-images/XCAD_FreeCOS/test/gt"
    # inpath_a = "../../DataSet-images/XCAD_FreeCOS/test/img"
    # outpath = "../DeNVeR_in/xca_dataset_video/pred.my46"
    # evaluate2(inpath_a, gtpath, outpath, model_my46)
    # print("单张图片计算准确率、效果会更好吗？")

    print("\n三、...FreeCOS(baseline)、XCAD数据集...")
    paramPath = "../../FreeCOS-GuangYuan.02_57/FreeCOS-GuangYuan50/logs/FreeCOS50.log/best_Segment.pt"
    model_my46 = getModel(paramPath)
    gtpath = "../../DataSet-images/XCAD_FreeCOS/test/gt"
    inpath_a = "../../DataSet-images/XCAD_FreeCOS/test/img"
    outpath = "../DeNVeR_in/xca_dataset_video/pred.my46"
    evaluate2(inpath_a, gtpath, outpath, model_my46)
    
    # 造影图、123456 print("abc",123)

