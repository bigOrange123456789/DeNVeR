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
if __name__ == "__main__":
    import yaml
    ROOT1 = os.path.dirname(script_path)
    file_path = os.path.join(ROOT1, "../",'confs/newConfig.yaml')
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        print("notes:",config["my"]["notes"])
        rootPath = config["my"]["filePathRoot"]
    print("rootPath:",rootPath)
    
    datasetPath="../DeNVeR_in/xca_dataset"
    paramPath = "../DeNVeR_in/models_config/freecos_Seg.pt"
    patient_names = [name for name in os.listdir(datasetPath)
                 if os.path.isdir(os.path.join(datasetPath, name))]
    CountSum = 0
    for patientID in patient_names:
        patient_path = os.path.join(datasetPath, patientID, "images")
        video_names = [name for name in os.listdir(patient_path)
                     if os.path.isdir(os.path.join(patient_path, name))]
        CountSum = CountSum + len(video_names)
    CountI = 0
    for patientID in patient_names:
        patient_path = os.path.join(datasetPath, patientID, "images")
        video_names = [name for name in os.listdir(patient_path)
                     if os.path.isdir(os.path.join(patient_path, name))]
        for videoId in video_names:
            CountI = CountI + 1
            print(str(CountI)+"/"+str(CountSum),videoId)
            inpath = os.path.join(datasetPath, patientID, "images", videoId)
            # outpath = os.path.join(datasetPath, patientID, "decouple", videoId)
            outpath = os.path.join(rootPath,  "dataset_decouple", patientID,"decouple", videoId)
            os.makedirs(outpath, exist_ok=True)
            startDecouple1(videoId, paramPath, inpath, outpath)  # 去除刚体层
            # startDecouple3(videoId, paramPath, inpath, outpath)  # 获取流体层
    # print("01：逐个图片进行归一化","vessel的效果最好")
    # print("02：放大方差","没有明显变化")
    print("03:基于整段视频的均值和方差")
    print("04:去除最大和最小的数据后再进行归一化","指标无显著变化0.7776")
    # evaluate(TAG="mix") #f1: 0.7630
    # evaluate(TAG="soft_fluid_1")#soft_fluid_1 0.7707),)
    evaluate(TAG="mix2", threshold=0.5) # evaluate(TAG="mix2")
    if False: evaluateNew(TAG="mix5")
    # evaluate(TAG="soft_fluid_2") #0.7772     # evaluate(TAG="pred")
    '''

    01：逐个图片进行归一化(vessel的效果最好)
    mix f1: 0.6523),) pr: (tensor(0.7386),) sn: tensor(0.5968)
    noRigid f1: 0.6549),) pr: (tensor(0.7398),) sn: tensor(0.6018)
    vessel f1: 0.6617),) pr: (tensor(0.7551),) sn: tensor(0.5994)
    orig f1: 0.6549),) pr: (tensor(0.7398),) sn: tensor(0.6018)
    
    02：方差乘2:
    vessel f1: (tensor(0.6617),) pr: (tensor(0.7551),) sn: tensor(0.5994)

    03:基于整段视频的均值和方差
    mix f1: (tensor(0.6523),) pr: (tensor(0.7386),) sn: tensor(0.5968)

    f1: (tensor(0.6523),) pr: (tensor(0.7386),) sn: tensor(0.5968)
    noRigid f1: (tensor(0.6549),) pr: (tensor(0.7398),) sn: tensor(0.6018)
    vessel f1: (tensor(0.6617),) pr: (tensor(0.7551),) sn: tensor(0.5994)
    orig f1: (tensor(0.6549),) pr: (tensor(0.7398),) sn: tensor(0.6018)
    这个去刚体信息的结果与之前的最佳结果相差很多、莫非是去极值的效果？

    noRigid2 f1: (tensor(0.6586),) pr: (tensor(0.7414),) sn: tensor(0.6062)
    pred f1: (tensor(0.7809),) pr: (tensor(0.7480),) sn: tensor(0.8264) 进行连通性分析
    pred f1: (tensor(0.7782),) pr: (tensor(0.7416),) sn: tensor(0.8275) 不进行连通性分析
    orig f1: (tensor(0.7558),) pr: (tensor(0.7176),) sn: tensor(0.8118)

    '''