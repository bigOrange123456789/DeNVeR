import os
import subprocess
import torch
# from concurrent import futures
script_path = os.path.abspath(__file__)
ROOT = os.path.dirname(script_path)
from RAFT.myPixelTracking import Track
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from free_cos.newTrain import initCSV, save2CVS, getIndicators
datasetPath=os.path.join(ROOT,"../","../DeNVeR_in/xca_dataset_sim2")
def evaluate(TAG="raft", threshold=0.5):
    labelErrorVideoId=[]
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
                for frameId in os.listdir(video_path):
                    pred_path = os.path.join(datasetPath, patientID, "decouple", videoId, "A.mask_nr2", TAG,frameId)
                    gt_path = os.path.join(datasetPath, patientID, "ground_truth", videoId,frameId)

                    pred = Image.open(pred_path).convert('L')
                    pred = transform(pred).unsqueeze(0).cuda()
                    pred[pred > threshold] = 1
                    pred[pred <= threshold] = 0

                    if True:
                        from preprocess.mySkeleton import getCatheter
                        catheter = getCatheter(pred[0,0].detach().cpu().numpy())
                        catheter = torch.from_numpy(catheter).unsqueeze(0).unsqueeze(0)
                        # pred = (1-catheter) * pred
                        pred[catheter]=0#导管处为背景

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
    # print(TAG,"f1:", f1, "pr:", pr, "sn:", sn )
    return f1, pr, sn
def evaluateNew(TAG="raft"):
    if True:
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

def traverseVideo():
    patient_names = [name for name in os.listdir(datasetPath)
                 if os.path.isdir(os.path.join(datasetPath, name))]
    CountSum = 0
    for patientID in patient_names:
        patient_path = os.path.join(datasetPath, patientID, "images")
        video_names = [name for name in os.listdir(patient_path)
                     if os.path.isdir(os.path.join(patient_path, name))]
        CountSum = CountSum + len(video_names)

    print("tag=fwd")
    with tqdm(total=CountSum) as pbar:
     for patientID in patient_names:
        patient_path = os.path.join(datasetPath, patientID, "images")
        video_names = [name for name in os.listdir(patient_path)
                     if os.path.isdir(os.path.join(patient_path, name))]
        for videoId in video_names:
            pbar.set_postfix(videoId=f"{videoId}")
            tag = "recon_non2" # "A.rigid.main_non1"
            if tag == "images":
                rgb_dir = os.path.join(datasetPath, patientID, "images", videoId)
                fwd_dir = os.path.join(datasetPath, patientID, "daft", videoId+"_fwd")
                bck_dir = os.path.join(datasetPath, patientID, "daft", videoId+"_bck")
            else:
                rgb_dir = os.path.join(datasetPath, patientID, "decouple", videoId, tag)
                fwd_dir = os.path.join(datasetPath, patientID, "daft_"+tag, videoId + "_fwd")
                bck_dir = os.path.join(datasetPath, patientID, "daft_"+tag, videoId + "_bck")
            mask_dir = os.path.join(datasetPath, patientID, "decouple", videoId, "A.mask.main_nr2", "filter")
            myTrack = Track(rgb_dir, fwd_dir, bck_dir, mask_dir)
            for i in myTrack.result:
                myTrack.save(
                    myTrack.result[i],
                    os.path.join(datasetPath, patientID, "decouple", videoId, "A.mask_nr2", i)
                )
            # myTrack.save(
            #     myTrack.vessel,
            #     os.path.join(datasetPath, patientID, "decouple", videoId, "A.mask_nr2", "raft")
            # )
            # myTrack.save(
            #     myTrack.vessel2,
            #     os.path.join(datasetPath, patientID, "decouple", videoId, "A.mask_nr2", "raft2")
            # )
            # myTrack.save(
            #     myTrack.vessel_g,
            #     os.path.join(datasetPath, patientID, "decouple", videoId, "A.mask_nr2", "raft_g")
            # )
            # myTrack.save(
            #     myTrack.vessel2_g,
            #     os.path.join(datasetPath, patientID, "decouple", videoId, "A.mask_nr2", "raft2_g")
            # )
            # myTrack.save(
            #     myTrack.vessel_gd,
            #     os.path.join(datasetPath, patientID, "decouple", videoId, "A.mask_nr2", "raft_gd")
            # )
            # myTrack.save(
            #     myTrack.vessel_gn,
            #     os.path.join(datasetPath, patientID, "decouple", videoId, "A.mask_nr2", "raft_gn")
            # )
            # myTrack.save(
            #     myTrack.vessel_gn2,
            #     os.path.join(datasetPath, patientID, "decouple", videoId, "A.mask_nr2", "raft_gn2")
            # )
            # myTrack.save(
            #     myTrack.vessel_gn_max,
            #     os.path.join(datasetPath, patientID, "decouple", videoId, "A.mask_nr2", "raft_gn_max")
            # )
            # myTrack.save(
            #     myTrack.vessel3,
            #     os.path.join(datasetPath, patientID, "decouple", videoId, "A.mask_nr2", "raft3")
            # )
            pbar.update(1)  # 每次增加 1

if __name__ == "__main__":
    print("三、基于A.rigid.main_non1计算光流图")
    print("四、光流图基于流体视频 ")
    # traverseVideo()
    # evaluateNew(TAG="raft")   #raft f1: (tensor(0.6607),) pr: (tensor(0.6640),) sn: tensor(0.6659)
    # evaluateNew(TAG="raft2")  #raft2 f1: (tensor(0.6665),) pr: (tensor(0.7139),) sn: tensor(0.6374)
    # evaluateNew(TAG="raft_g") #raft_g f1: (tensor(0.7168),) pr: (tensor(0.7289),) sn: tensor(0.7121)
    # evaluateNew(TAG="raft2_g")#raft2_g f1: (tensor(0.0510),) pr: (tensor(0.2277),) sn: tensor(0.0327)
    evaluateNew(TAG="raft_gd")  #raft_gd f1: (tensor(0.7806),) pr: (tensor(0.7477),) sn: tensor(0.8260)
    # evaluateNew(TAG="raft_gn")#raft_gn f1: (tensor(0.7585),) pr: (tensor(0.7501),) sn: tensor(0.7750)
    # evaluateNew(TAG="raft_gn_max") #raft_gn_max f1: (tensor(0.6805),) pr: (tensor(0.5613),) sn: tensor(0.8789)
    # evaluateNew(TAG="raft3") #raft3 f1: (tensor(0.5381),) pr: (tensor(0.3881),) sn: tensor(0.9229)
    '''
    一、普通
    raft:  f1:0.6498 pr:0.6204 sn:0.6904
    raft2: f1:0.6634 pr:0.6661 sn:0.6717
    raft_g f1: 0.7106 pr: (tensor(0.6873),) sn: tensor(0.7413)
    raft2_g f1: 0.0731 pr: (tensor(0.2256),) sn: tensor(0.0480)
    raft_gd: f1: 0.7644 pr: (tensor(0.6918),) sn: tensor(0.8607)
             f1: 0.7781 pr: 0.8095  sn: 0.7538 threshold=0.95
    raft_gn: f1: 0.7498 pr: 0.7027 sn: 0.8091
             f1: 0.7530 pr: 0.7278  sn: 0.7852 threshold=0.6
    raft_gn_max: 0.6396 pr: 0.4966 sn: 0.9130 #查全率过高,查准率过低
    raft3 f1: 0.4728 pr: 0.3205 sn: 0.9516
    二、原始MASK基于不含肺部的刚体层
    raft f1: (tensor(0.6607),) pr: (tensor(0.6640),) sn: tensor(0.6659)
    raft2 f1: (tensor(0.6665),) pr: (tensor(0.7139),) sn: tensor(0.6374)
    raft_g f1: (tensor(0.7168),) pr: (tensor(0.7289),) sn: tensor(0.7121)
    raft2_g f1: (tensor(0.0510),) pr: (tensor(0.2277),) sn: tensor(0.0327)
    raft_gd f1: (tensor(0.7806),) pr: (tensor(0.7477),) sn: tensor(0.8260)
    raft_gn f1: (tensor(0.7585),) pr: (tensor(0.7501),) sn: tensor(0.7750)
    raft_gn_max f1: (tensor(0.6805),) pr: (tensor(0.5613),) sn: tensor(0.8789)
    raft3 f1: (tensor(0.5381),) pr: (tensor(0.3881),) sn: tensor(0.9229)
    
    # 发生遮挡的区域很有可能是背景(M2_new = torch.where(O12_new == 1, 0.8*masks2, M2_new))
    raft f1: (tensor(0.6688),) pr: (tensor(0.6892),) sn: tensor(0.6593)
    raft2 f1: (tensor(0.4311),) pr: (tensor(0.8383),) sn: tensor(0.3190)
    raft_g f1: (tensor(0.7234),) pr: (tensor(0.7453),) sn: tensor(0.7100)
    raft2_g f1: (tensor(0.0098),) pr: (tensor(0.1788),) sn: tensor(0.0053)
    raft_gn f1: (tensor(0.7585),) pr: (tensor(0.7502),) sn: tensor(0.7749)
    raft_gn_max f1: (tensor(0.6806),) pr: (tensor(0.5619),) sn: tensor(0.8781)
    raft3 f1: (tensor(0.5505),) pr: (tensor(0.4038),) sn: tensor(0.9138)
    
    三、光流图基于去刚体视频 (与之前的实验差别不大、没有出现显著提升)
    raft f1: (tensor(0.6612),) pr: (tensor(0.6613),) sn: tensor(0.6699)
    raft2 f1: (tensor(0.6680),) pr: (tensor(0.7073),) sn: tensor(0.6455)
    raft_g f1: (tensor(0.7178),) pr: (tensor(0.7254),) sn: tensor(0.7191)
    raft2_g f1: (tensor(0.0503),) pr: (tensor(0.2358),) sn: tensor(0.0324)
    raft_gd f1: (tensor(0.7806),) pr: (tensor(0.7477),) sn: tensor(0.8260)
    raft_gn f1: (tensor(0.7608),) pr: (tensor(0.7496),) sn: tensor(0.7810)
    raft_gn_max f1: (tensor(0.6842),) pr: (tensor(0.5655),) sn: tensor(0.8805)
    raft3 f1: (tensor(0.5562),) pr: (tensor(0.4053),) sn: tensor(0.9239)
    
    四、光流图基于流体视频 #提升幅度可以忽略不记
    raft f1: (tensor(0.6859),) pr: (tensor(0.6776),) sn: tensor(0.7020)
    raft2 f1: (tensor(0.6921),) pr: (tensor(0.7140),) sn: tensor(0.6812)
    raft_g f1: (tensor(0.7350),) pr: (tensor(0.7298),) sn: tensor(0.7478)
    raft2_g f1: (tensor(0.0523),) pr: (tensor(0.2330),) sn: tensor(0.0344)
    raft_gd f1: (tensor(0.7806),) pr: (tensor(0.7477),) sn: tensor(0.8260)
    raft_gn f1: (tensor(0.7682),) pr: (tensor(0.7493),) sn: tensor(0.7963)
    raft_gn_max f1: (tensor(0.6989),) pr: (tensor(0.5864),) sn: tensor(0.8791)
    raft3 f1: (tensor(0.5893),) pr: (tensor(0.4410),) sn: tensor(0.9220)
    （去除导管）
    (1)没有去除导管
    raft_gd.old f1:0.7806 pr: (tensor(0.7477),) sn: tensor(0.8260)
    (2)去除导管之后 
    raft_gd.new f1:0.7815 pr: (tensor(0.7504),) sn: tensor(0.8252)
    (3)去除更多导管：导管长度下界由150变为75
    raft_gd f1:0.7842 pr: (tensor(0.7568),) sn: tensor(0.8235)
    (4)纠正了CVAI-2170RAO27_CAU29中的三个错误标注
    raft_gd f1:0.7846 pr: (tensor(0.7572),) sn: tensor(0.8239)
    (5)去除更多导管：导管长度下界由75变为0 #有较大提升、准确率比最开始提升了2个点
    raft_gd f1: (tensor(0.7856),) pr: (tensor(0.7606),) sn: tensor(0.8221)
    (6)去除更多导管：最低半径由mean*0.75变为mean*0.5
    raft_gd f1: (tensor(0.7859),) pr: (tensor(0.7639),) sn: tensor(0.8190)
    (7)去除更多导管：取消最低半径约束
    raft_gd f1: (tensor(0.7873),) pr: (tensor(0.7686),) sn: tensor(0.8167)
    '''
    '''
    # 难崩,无法基于光流一致性提升指标
    print("01:添加了两组新的实验:如果raft_gd和原结果一样,并且raft3不比原结果好,那就需要回到baseline的框架中.","最担心的情况出现了")
    '''
    # tag =  "recon_non2" # "orig"
    # outpath = './data/myReflection4_03'
    # rgb_dir = os.path.join(ROOT, outpath, tag)
    # fwd_dir = os.path.join(ROOT, outpath, tag + "_fwd")
    # fwd_img_dir = os.path.join(ROOT, outpath, tag + "_fwdImg")
    #
    # runRAFT(rgb_dir, fwd_dir, fwd_img_dir, +1)

'''
export PATH="~/anaconda3/bin:$PATH"
source activate DNVR
python -m RAFT.raftBatch 
'''
