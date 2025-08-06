import os
from nir.new import startDecouple1,startDecouple2

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
    if TAG=="vessel":
        img_path = os.path.join(datasetPath, patientID, "decouple", videoId, "recon_non2", frameId)
        img = Image.open(img_path).convert('L') 
        img = transform(img).unsqueeze(0).cuda()
    elif TAG=="noRigid1":
        img_path = os.path.join(datasetPath, patientID, "decouple", videoId, "A.rigid.main_non1", frameId)
        img = Image.open(img_path).convert('L') 
        img = transform(img).unsqueeze(0).cuda()
    elif TAG=="noRigid2":
        img_path = os.path.join(datasetPath, patientID, "decouple", videoId, "A.rigid.main_non2", frameId) # A.mask.main_nr2
        img = Image.open(img_path).convert('L') 
        img = transform(img).unsqueeze(0).cuda()
    elif TAG=="mix":
        img_path1 = os.path.join(datasetPath, patientID, "decouple", videoId, "recon_non2", frameId)
        img1 = Image.open(img_path1).convert('L') 
        img1 = transform(img1).unsqueeze(0).cuda()
        img_path2 = os.path.join(datasetPath, patientID, "decouple", videoId, "orig", frameId)
        img2 = Image.open(img_path2).convert('L') 
        img2 = transform(img2).unsqueeze(0).cuda()
        img=(img1+img2)/2
    elif TAG=="orig":
        img_path = os.path.join(datasetPath, patientID, "decouple", videoId, "A.rigid.main_non1", frameId)
        img = Image.open(img_path).convert('L') 
        img = transform(img).unsqueeze(0).cuda()
    return img
def getImgAll(TAG,transform, patientID, videoId):
    img_list = []
    path0 = os.path.join(datasetPath, patientID, "decouple", videoId, "orig")
    # print(path0,len(os.path.join(path0)))
    for frameId in os.listdir(os.path.join(path0)):
        # frameId = str(i).zfill(5)+".png"
        img = getImg(TAG,transform, patientID, videoId, frameId)
        img_list.append(img)
    imgAll = torch.cat(img_list, dim=0)
    if False:
        flat_tensor = imgAll.flatten()
        q_min = torch.quantile(flat_tensor, 0.05)
        q_max = torch.quantile(flat_tensor, 0.95)
        mask = (flat_tensor >= q_min) & (flat_tensor <= q_max)
        imgAll = flat_tensor[mask]
    mean0 = imgAll.mean()
    std0 = imgAll.std()
    return mean0, std0
from free_cos.main import mainFreeCOS
def evaluate(TAG="raft", threshold=0.5):
    print("TAG:",TAG)
    paramPath = "../DeNVeR_in/models_config/freecos_Seg.pt"
    model = getModel(paramPath)
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
                    mean0, std0 = getImgAll(TAG,transform, patientID, videoId)
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

                    pred[pred > 0.5] = 1 #是0~1与0~255的原因吗？
                    pred[pred <=  0.5] = 0
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
                    gt[gt > threshold] = 1
                    gt[gt <= threshold] = 0

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

if __name__ == "__main__":
    # datasetPath="../DeNVeR_in/xca_dataset"
    # paramPath = "../DeNVeR_in/models_config/freecos_Seg.pt"
    # patient_names = [name for name in os.listdir(datasetPath)
    #              if os.path.isdir(os.path.join(datasetPath, name))]
    # CountSum = 0
    # for patientID in patient_names:
    #     patient_path = os.path.join(datasetPath, patientID, "images")
    #     video_names = [name for name in os.listdir(patient_path)
    #                  if os.path.isdir(os.path.join(patient_path, name))]
    #     CountSum = CountSum + len(video_names)
    # CountI = 0
    # for patientID in patient_names:
    #     patient_path = os.path.join(datasetPath, patientID, "images")
    #     video_names = [name for name in os.listdir(patient_path)
    #                  if os.path.isdir(os.path.join(patient_path, name))]
    #     for videoId in video_names:
    #         CountI = CountI + 1
    #         print(str(CountI)+"/"+str(CountSum),videoId)
    #         inpath = os.path.join(datasetPath, patientID, "images", videoId)
    #         outpath = os.path.join(datasetPath, patientID, "decouple", videoId)
    #         os.makedirs(outpath, exist_ok=True)
    #         startDecouple1(videoId, paramPath, inpath, outpath)  # 去除刚体层
    #         startDecouple2(videoId, paramPath, inpath, outpath)  # 获取流体层
    # print("01：逐个图片进行归一化","vessel的效果最好")
    # print("02：放大方差","没有明显变化")
    print("03:基于整段视频的均值和方差")
    print("04:去除最大和最小的数据后再进行归一化")
    # evaluate(TAG="mix")
    # evaluate(TAG="noRigid2")
    # evaluate(TAG="vessel")
    # evaluate(TAG="orig")
    evaluate(TAG="pred")
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

    '''