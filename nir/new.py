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

from free_cos.main import mainFreeCOS
from eval.eval import Evaluate

from nir.model import Siren
from nir.util import get_mgrid, jacobian
from nir.util import Dataset,ToTensor

from nir.myLib.Decouple import Decouple
from nir.myLib.mySave import check,save2img,save0

videoId="CVAI-2855LAO26_CRA31"
paramPath = "../DeNVeR_in/models_config/freecos_Seg.pt"
pathIn = 'nir/data/in2'
maskPath="./nir/data/mask/filter"
pathLable = "nir/data/in2_gt"
pathMaskOut = "./nir/data/mask"
print("01:test", "")

EpochNum = 6000 #5000 #3000
outpath = './nir/data/new_01'

mainFreeCOS(paramPath,pathIn,os.path.join(outpath, "mask_nir0"))
check(os.path.join(outpath, "mask_nir0"),videoId,"nir.0")

# 刚体解耦
if True:
    from nir.myLib.Decouple_rigid import Decouple_rigid
    myMain=Decouple_rigid(pathIn,maskPath=maskPath,hidden_features=256)
    myMain.train(EpochNum) #EpochNum =5000

def save1(o_scene, tag):
    o_scene = o_scene.cpu().detach().numpy()
    o_scene = (o_scene * 255).astype(np.uint8)
    save2img(o_scene[:, :, :, 0], os.path.join(outpath, tag))

#基于去除刚体后的视频预测MASK
if True:# False:
 with torch.no_grad(): #
    orig = myMain.v.video.clone()
    orig = orig.permute(0, 2, 3, 1).detach().numpy()
    orig = (orig * 255).astype(np.uint8)
    save2img(orig[:, :, :, 0], os.path.join(outpath, 'orig'))

    orig = myMain.v.video.clone()
    N, C, H, W = orig.size()  # 帧数、通道数、高度、宽度
    orig = orig.permute(0, 2, 3, 1).detach()#.numpy()

    video_pre, layers, p =myMain.getVideo(1)#使用局部形变

    save1(0.5*orig.cuda()/(p["o_rigid_all"].abs()+10**-10), "A.rigid_non2")

    mainFreeCOS(paramPath,os.path.join(outpath, "A.rigid_non2"),os.path.join(outpath, "A.mask_nr2"))
    check(os.path.join(outpath, "A.mask_nr2"),videoId,"A.nir.1.rigid_non2")
###############################################################################################################
###############################################################################################################
from nir.model import Siren
from nir.util import get_mgrid, jacobian
from nir.util import Dataset,ToTensor

from nir.myLib.Layer import Layer
from nir.myLib.Decouple import Decouple
from nir.myLib.mySave import check,save2img,save0
# 解耦出血管层
if True:
    maskPath = os.path.join(outpath, "A.mask_nr2","filter")
    myMain=Decouple(pathIn,maskPath=maskPath,hidden_features=256)
    myMain.train(EpochNum) #EpochNum =5000

#输出解耦效果
if True:# False:
 with torch.no_grad(): #
    orig = myMain.v.video.clone()
    N, C, H, W = orig.size()  # 帧数、通道数、高度、宽度
    orig = orig.permute(0, 2, 3, 1).detach()#.numpy()

    video_pre, layers, p =myMain.getVideo()

    save1(video_pre, "recon")
    save1(orig.cuda()/(video_pre.abs()+10**-10), "recon_non")
    save1(0.5*orig.cuda()/(video_pre.abs()+10**-10), "recon_non2")
    save1(p["o_rigid_all"], "rigid")
    save1(p["o_soft_all"], "soft")

    for i in range(len(layers["r"])):
        save1(layers["r"][i], "rigid" + str(i))
    for i in range(len(layers["s"])):
        save1(layers["s"][i], "soft" + str(i))
    save1(layers["f"], "fluid")

    mainFreeCOS(paramPath,os.path.join(outpath, "recon_non2"),os.path.join(outpath, "mask2"))
    check(os.path.join(outpath, "mask2"),videoId,"nir.1.recon_non2")
    if False:
        mainFreeCOS(paramPath, os.path.join(outpath, "recon_non3"), os.path.join(outpath, "mask3"))
        check(os.path.join(outpath, "mask3"), videoId, "nir.1.recon_non3")

# exit(0)
# 拟合解耦出来的血管纹理
if True: #效果不好
    from nir.myLib.ModelVessel import ModelVessel
    mySim1=ModelVessel(os.path.join(outpath,"recon_non2"),maskPath=maskPath) #只拟合血管区域
    mySim1.train(EpochNum) # EpochNum =5000
    video_pre1 = mySim1.getVideo()
    save1(video_pre1, "recon_non2_smooth_onlyVessel.01")
    # mainFreeCOS(paramPath, os.path.join(outpath, "recon_non2_smooth"), os.path.join(outpath, "mask2_"))
    # check(os.path.join(outpath, "mask2_"), videoId, "nir.1.recon_non2_smooth")
'''
    python -m nir.new
'''
