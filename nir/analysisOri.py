import os
import numpy as np
from itertools import chain
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import re
import torch.nn as nn
import itertools
import math
import torch.backends.cudnn as cudnn

from free_cos.main import mainFreeCOS
from eval.eval import Evaluate

from nir.model import Siren
from nir.util import get_mgrid, jacobian
from nir.util import Dataset,ToTensor

from nir.myLib.Decouple import Decouple
from nir.myLib.mySave import check,save2img,save0

from free_cos.ModelSegment import ModelSegment

###############################################################################################################
###############################################################################################################

from nir.myLib.Layer import Layer
from nir.myLib.mySave import check,save2img,save0
from nir.myLib.VideoFitting import VideoFitting

import csv
from torchvision import transforms

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
    Segment_model.load_state_dict(checkpoint['state_dict'])  # 提取模型状态字典并赋值给模型
    return Segment_model

from free_cos.newTrain import initCSV, save2CVS, getIndicators
def evaluate(pathIn,pathGt,pathOut,model): #全部图片求均值
    os.makedirs(os.path.join(pathOut), exist_ok=True)
    head = ["fileName","accuracy","recall","precision","f1","iou","specificity"]
    path = os.path.join(pathOut, "experiment_results.csv")
    initCSV(path, head)
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
    ])
    datasetPath = pathIn#"../DeNVeR_in/xca_dataset_video"
    sum_recall = 0
    sum_precision = 0
    sum_f1 = 0
    img_list = []
    for name in os.listdir(os.path.join(datasetPath)):
        path_img = os.path.join(datasetPath,name)
        img = Image.open( path_img ).convert('L')
        img = transform(img).unsqueeze(0).cuda()
        img_list.append(img)
    img_list = torch.cat(img_list, dim=0)

    for name in os.listdir(os.path.join(datasetPath)):
        path_img = os.path.join(datasetPath,name)
        img = Image.open( path_img ).convert('L')
        img = transform(img).unsqueeze(0).cuda()
        # img = (img - img.mean() ) / img.std()
        img = (img - img_list.mean()) / img_list.std()
        pred = model(img)["pred"]
        pred[pred >= 0.5] = 1
        pred[pred <  0.5] = 0
        path_gt = os.path.join(pathGt, name)
        gt = Image.open(path_gt).convert('L')
        gt = transform(gt).unsqueeze(0).cuda()
        gt[gt >= 0.5] = 1
        gt[gt < 0.5] = 0

        ind=getIndicators(
            pred[0,0].detach().cpu()*255,
            gt[0,0].detach().cpu()*255
        )
        ind["fileName"]=name.split(".png")[0]
        save2CVS(path, head, ind)
        sum_recall += ind["recall"]
        sum_precision += ind["precision"]
        sum_f1 += ind["f1"]
    print("f1:", sum_f1.item() / len(os.listdir(datasetPath)),
          "pr:", sum_precision.item() / len(os.listdir(datasetPath)),
          "sn:", sum_recall.item() / len(os.listdir(datasetPath)))
def evaluate_old(pathIn,pathGt,pathOut,model): #每张图片逐个计算均值
    os.makedirs(os.path.join(pathOut), exist_ok=True)
    head = ["fileName","accuracy","recall","precision","f1","iou","specificity"]
    path = os.path.join(pathOut, "experiment_results.csv")
    initCSV(path, head)
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
    ])
    datasetPath = pathIn#"../DeNVeR_in/xca_dataset_video"
    sum_recall = 0
    sum_precision = 0
    sum_f1 = 0
    for name in os.listdir(os.path.join(datasetPath)):
        path_img = os.path.join(datasetPath,name)
        img = Image.open( path_img ).convert('L')
        img = transform(img).unsqueeze(0).cuda()
        img = (img - img.mean() ) / img.std()
        pred = model(img)["pred"]
        pred[pred >= 0.5] = 1
        pred[pred <  0.5] = 0
        path_gt = os.path.join(pathGt, name)
        gt = Image.open(path_gt).convert('L')
        gt = transform(gt).unsqueeze(0).cuda()
        gt[gt >= 0.5] = 1
        gt[gt < 0.5] = 0

        ind=getIndicators(
            pred[0,0].detach().cpu()*255,
            gt[0,0].detach().cpu()*255
        )
        ind["fileName"]=name.split(".png")[0]
        save2CVS(path, head, ind)
        sum_recall += ind["recall"]
        sum_precision += ind["precision"]
        sum_f1 += ind["f1"]
    print("f1:", sum_f1.item() / len(os.listdir(datasetPath)),
          "pr:", sum_precision.item() / len(os.listdir(datasetPath)),
          "sn:", sum_recall.item() / len(os.listdir(datasetPath)))

if __name__ == "__main__":
    inpath = "../DeNVeR_in/xca_dataset_video/img"
    gtpath = "../DeNVeR_in/xca_dataset_video/gt"
    outpath = "../DeNVeR_in/xca_dataset_video/pred.freecos"
    paramPath = "../DeNVeR_in/models_config/freecos_Seg.pt"
    print("version:2")
    model = getModel(paramPath)
    evaluate(inpath, gtpath, outpath, model)
