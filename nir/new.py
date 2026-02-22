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

from free_cos.main import mainFreeCOS, mainFreeCOS_sim
from eval.eval import Evaluate

from nir.model import Siren
from nir.util import get_mgrid, jacobian
from nir.util import Dataset,ToTensor

# from nir.myLib.Decouple import Decouple
from nir.myLib.mySave import check,save2img,save0

# maskPath="./nir/data/mask/filter"

EpochNum = 4000 #6000#000 #5000 #3000
# outpath = './nir/data/new_08'
flagHadRigid=False #是否已经完成了刚体解耦

def startDecouple1(videoId,paramPath,pathIn,outpath,config=None): #单独的刚体解耦
    #设置参数
    myEpochNum = EpochNum
    tag = "A"
    if not config is None:
        if "epoch" in config:
            myEpochNum = config ["epoch"]
        if "tag" in config:
            tag = config["tag"]

    if False:
        mainFreeCOS(paramPath,pathIn,os.path.join(outpath, "mask_nir0"))
        check(os.path.join(outpath, "mask_nir0"),videoId,"nir.0")

    # 刚体解耦
    if not flagHadRigid:
        from nir.myLib.Decouple_rigid import Decouple_rigid
        myMain=Decouple_rigid(pathIn,hidden_features=256)
        myMain.train(myEpochNum) 

    def save1(o_scene, tag):
        if o_scene==None or len(o_scene)==0: return
        o_scene[o_scene>1]=1#添加这个操作来去除黑点，否则超过上限后颜色值会变为黑色
        o_scene = o_scene.cpu().detach().numpy()
        o_scene = (o_scene * 255).astype(np.uint8)
        save2img(o_scene[:, :, :, 0], os.path.join(outpath, tag))

    # 基于去除刚体后的视频预测MASK
    if not flagHadRigid: # False:
     with torch.no_grad(): #
        orig = myMain.v.video.clone()
        orig = orig.permute(0, 2, 3, 1).detach().numpy()
        orig = (orig * 255).astype(np.uint8)
        save2img(orig[:, :, :, 0], os.path.join(outpath, 'orig'))

        orig = myMain.v.video.clone()
        N, C, H, W = orig.size()  # 帧数、通道数、高度、宽度
        orig = orig.permute(0, 2, 3, 1).detach()#.numpy()

        video_pre, layers, p = myMain.getVideo(1)#使用局部形变

        save1(p["o_rigid_all"], tag+".rigid")#看一下刚体层的效果
        for i in range(len(layers["r"])):
            save1(layers["r"][i], tag+".rigid" + str(i))
        save1(0.5*orig.cuda()/(p["o_rigid_all"].abs()+10**-10), tag+".rigid_non2")

        mainFreeCOS(paramPath,os.path.join(outpath, tag+".rigid_non2"),os.path.join(outpath, tag+".mask_nr2_old"))
        if False:check(os.path.join(outpath, tag+".mask_nr2_old"),videoId,tag+".nir.1.rigid_non2_old")
        mainFreeCOS(paramPath, os.path.join(outpath, tag+".rigid_non2"), os.path.join(outpath, tag+".mask_nr2"),needConnect=False)

        rigidMain=layers["r"][myMain.getMainRigidIndex()]
        save1(rigidMain, tag+".rigid.main")
        main_non1 = orig.cuda() / (rigidMain.abs() + 10 ** -10)
        # main_non1
        # main_non1[main_non1 > 1] = 1
        save1(main_non1, tag+".rigid.main_non1")#旧版这个结果有黑点、现在黑点解决了(是超过数据上限造成的)
        # if True:#测试
        #     o = orig.cuda() / (rigidMain.abs() + 10 ** -10)
        #     print(type(o))
        #     print(o.shape)
        #     # 获取最大值和最小值
        #     # max_val = np.max(o)
        #     # min_val = np.min(o)
        #     # 获取全局最大值和最小值
        #     max_val = o.max().item()
        #     min_val = o.min().item()
        #     print("max_val",max_val)
        #     print("min_val",min_val)
        #     exit(0)

        save1(0.5 * orig.cuda() / (rigidMain.abs() + 10 ** -10), tag+".rigid.main_non2")
        mainFreeCOS(paramPath, os.path.join(outpath, tag+".rigid.main_non2"), os.path.join(outpath, tag+".mask.main_nr2"))
        if False:check(os.path.join(outpath, tag+".mask.main_nr2"), videoId, tag+".mask.main_nr2")


def startDecouple1_sim(videoId,paramPath,pathIn,outpath,config=None): #单独的刚体解耦 #大部分时间浪费在推理分析和数据存储上了
    #设置参数
    epochs = None
    total_steps = EpochNum #只用epochs=None的时候才有效
    batch_size_scale = 1/8
    tag = "A"
    useSmooth=False
    openLocalDeform=False
    weight_smooth=1
    weight_concise=0.0001
    weight_component =1
    stillness=False
    NUM_rigid = 2 #刚体层数量
    NUM_soft = 0
    NUM_fluid = 0 #流体层数量
    stillnessFristLayer =stillness
    useMask = False
    openReconLoss_rigid = False
    lossParam = None
    lossParam_vessel = None
    lossFunType = {
                    "rm":"MSE",
                    "ra":"MSE",
                    "rv":"MSE",
                }
    interval = 1.0
    # configRigid = None
    configRigids={}
    configSofts={}
    configFluids={}
    adaptiveFrameNumMode = 0 
    use_dynamicFeatureMask = False
    dynamicVesselMask=False
    # useMatrix = True
    if not config is None:
        if "dynamicVesselMask" in config:
            dynamicVesselMask=config["dynamicVesselMask"]
        if "epochs" in config:
            epochs=config["epochs"]
        elif "total_steps" in config:
            total_steps = config["total_steps"]
        elif "epoch" in config:#废弃版本
            total_steps = config ["epoch"]
        if "tag" in config:
            tag = config["tag"]
        if "useSmooth"in config:
            useSmooth=config["useSmooth"]
        if "openLocalDeform"in config:
            openLocalDeform=config["openLocalDeform"]
        if "weight_smooth" in config:
            weight_smooth=config["weight_smooth"]
        if "weight_concise" in config:
            weight_concise = config["weight_concise"]
        if "weight_component" in config:
            weight_component =config["weight_component"]
        if "stillness" in config:
            stillness = config["stillness"]
        if "NUM_rigid" in config:
            NUM_rigid = config["NUM_rigid"]
        if "stillnessFristLayer" in config:
            stillnessFristLayer=config["stillnessFristLayer"]
        else:
            stillnessFristLayer=stillness
        if "NUM_soft" in config:
            NUM_soft = config["NUM_soft"]
        if "useMask" in config:
            useMask = config["useMask"]
        if "openReconLoss_rigid" in config:
            openReconLoss_rigid = config["openReconLoss_rigid"]
        if "lossType" in config:
            lossType = config["lossType"]
        if "lossParam" in config:
            lossParam = config["lossParam"]
        if "lossParam_vessel" in config:
            lossParam_vessel = config["lossParam_vessel"]
        if "lossFunType" in config:
            lossFunType = config["lossFunType"]
        # if "useMatrix" in config:
        #     useMatrix = config["useMatrix"]
        if "interval" in config:
            interval = config["interval"]
        # if "configRigid" in config:
        #     configRigid = config["configRigid"]
        if "configRigids" in config:
            configRigids = config["configRigids"]
        if "configSofts" in config:
            configSofts = config["configSofts"]
        if "NUM_fluid" in config:
            NUM_fluid = config["NUM_fluid"]
        if "configFluids" in config:
            configFluids = config["configFluids"]
        if "adaptiveFrameNumMode" in config:
            adaptiveFrameNumMode = config["adaptiveFrameNumMode"]
        if "use_dynamicFeatureMask" in config:
            use_dynamicFeatureMask = config["use_dynamicFeatureMask"]
        # if 
    # print("168-configFluids",configFluids)
    # print(weight_smooth,config["weight_smooth"])
    # print("config",config)
    # exit(0)


    # if False:
    # if useMask and not os.path.exists(os.path.join(outpath, "mask_nir0")):
    #     mainFreeCOS(paramPath,pathIn,os.path.join(outpath, "mask_nir0"))
    #     # check(os.path.join(outpath, "mask_nir0"),videoId,"nir.0")
    #分割原始数据中的血管
    Segment_model = None
    if "maskPath_pathIn" in config and not config["maskPath_pathIn"] is None: #使用之前计算好的MASK
        pathIn2=os.path.join(outpath, config["maskPath_pathIn"])
        maskPath=os.path.join(outpath, tag+".orig_mask2")
        print("useMask and not os.path.exists(maskPath)",useMask and not os.path.exists(maskPath))
        if useMask :
            if True: # not os.path.exists(maskPath): #为了每次训练的结果相同，MASK每次训练都会重新生成
                Segment_model=mainFreeCOS_sim(paramPath,pathIn2,maskPath)
            # print("Test123")
        # print("pathIn2",pathIn2)
        # exit(0)
        # config["maskPath"]
    else:#重新生成MASK
        maskPath=os.path.join(outpath, tag+".orig_mask")
        if useMask:
            if True:#not os.path.exists(maskPath):
                Segment_model=mainFreeCOS_sim(paramPath,pathIn,maskPath)

    # 刚体解耦
    if not flagHadRigid:
        from nir.myLib.Decouple_rigid import Decouple_rigid
        myMain=Decouple_rigid(pathIn,hidden_features=256,
                              useSmooth=useSmooth,
                              openLocalDeform=openLocalDeform,
                              weight_smooth=weight_smooth,
                              weight_concise=weight_concise,
                              weight_component=weight_component,
                              #   stillness=stillness, #废弃
                              stillnessFristLayer=stillnessFristLayer,
                              NUM_rigid=NUM_rigid,
                              NUM_soft=NUM_soft,
                              useMask=useMask,
                              openReconLoss_rigid=openReconLoss_rigid,
                              maskPath=maskPath,
                              lossType=lossType,
                              lossFunType=lossFunType,
                              # useMatrix=useMatrix,
                              interval=interval,
                            #   configRigid=configRigid,
                              configRigids=configRigids,
                              configSofts=configSofts,
                              NUM_fluid=NUM_fluid,
                              configFluids=configFluids,
                              adaptiveFrameNumMode=adaptiveFrameNumMode,
                              use_dynamicFeatureMask=use_dynamicFeatureMask,
                              dynamicVesselMask=dynamicVesselMask,
                              updateMaskConfig={#更新所需信息 #dynamicVesselMask不为False的时候才启用
                                  "mainFreeCOS_sim":mainFreeCOS_sim,
                                  "pathIn":pathIn,#没啥用
                                  "maskPath":maskPath,#需要实时更新文件夹中的内容
                                  "pathInNew":os.path.join(outpath,tag+".rigid.non1"),#需要实时计算文件夹中的内容
                                  "outpath":outpath,
                                  "Segment_model":Segment_model,
                                  "tag": tag+".rigid.non1",
                              }
                        )
        myMain.train(
            epochs,total_steps,
            lossParam=lossParam,
            lossParam_vessel=lossParam_vessel,
            batch_size_scale=batch_size_scale)

    def save1(o_scene, tag):
        if o_scene==None or len(o_scene)==0: return
        o_scene[o_scene>1]=1#添加这个操作来去除黑点，否则超过上限后颜色值会变为黑色
        o_scene = o_scene.cpu().detach().numpy()
        o_scene = (o_scene * 255).astype(np.uint8)
        save2img(o_scene[:, :, :, 0], os.path.join(outpath, tag))

    # 基于去除刚体后的视频预测MASK
    if not flagHadRigid: # False:
     with torch.no_grad(): #
        orig = myMain.v.video.clone()
        orig = orig.permute(0, 2, 3, 1).detach().numpy()
        orig = (orig * 255).astype(np.uint8)
        # if True: #这里是输入的视频不是用于分割的视频
        if not os.path.exists(os.path.join(outpath, 'orig')):#如果原始数据还没有复制过来
            save2img(orig[:, :, :, 0], os.path.join(outpath, 'orig'))

        orig = myMain.v.video.clone()
        # N, C, H, W = orig.size()  # 帧数、通道数、高度、宽度
        orig = orig.permute(0, 2, 3, 1).detach()#.numpy()

        video_pre, layers, p = myMain.getVideo(1)#使用局部形变

        # soft结果 
        # save1(p["o_soft_all"], tag+".soft")#看一下刚体层的效果
        if NUM_soft>0:#输出全部的软体层
         if "s" in layers:
            for i in range(len(layers["s"])):
                save1(layers["s"][i], tag+".soft" + str(i))
            if myMain.useSoftMask:
                for i in range(len(layers["softMask"])):
                    save1(layers["softMask"][i], tag+".softMask" + str(i))
            # for i in range(len(layers["s"])):
            #     save1(layers["s"][i], tag+".soft" + str(i))

        # rigid结果 
        if NUM_rigid>0: 
            save1(p["o_rigid_all"], tag+".rigid")#看一下刚体层的效果
            if not p["o_rigid_all"]==None and len(p["o_rigid_all"])>0: 
                save1(
                    orig.cuda() / (p["o_rigid_all"].abs() + 10 ** -10), 
                    tag+".rigid.non1")#有黑点、黑点解决了(是超过数据上限造成的)
        if False:
            for i in range(len(layers["r"])):
                save1(layers["r"][i], tag+".rigid" + str(i))
            save1(0.5*orig.cuda()/(p["o_rigid_all"].abs()+10**-10), tag+".rigid_non2")

        # fluid结果
        if NUM_fluid>0:#"f" in layers:
            save1(p["o_fluid_all"], tag+".fluid")
            for i in range(len(layers["f"])):
                save1(layers["f"][i], tag+".fluid" + str(i))

        # rigid (推理分割图，并评估指标，推理分割图+连通后处理)
        if False:
            mainFreeCOS(paramPath,os.path.join(outpath, tag+".rigid_non2"),os.path.join(outpath, tag+".mask_nr2_old"))
            check(os.path.join(outpath, tag+".mask_nr2_old"),videoId,tag+".nir.1.rigid_non2_old")
            mainFreeCOS(paramPath, os.path.join(outpath, tag+".rigid_non2"), os.path.join(outpath, tag+".mask_nr2"),needConnect=False)

        # main
        if False:
            rigidMain=layers["r"][myMain.getMainRigidIndex()]
            save1(rigidMain, tag+".rigid.main")

            # main_non1
            main_non1 = orig.cuda() / (rigidMain.abs() + 10 ** -10)
            save1(main_non1, tag+".rigid.main_non1")#有黑点、黑点解决了(是超过数据上限造成的)

        # main_non2 (推理分割图，并评估指标，存储分割图)
        if False:save1(0.5 * orig.cuda() / (rigidMain.abs() + 10 ** -10), tag+".rigid.main_non2")
        if False:
            mainFreeCOS(paramPath, os.path.join(outpath, tag+".rigid.main_non2"), os.path.join(outpath, tag+".mask.main_nr2"))
            check(os.path.join(outpath, tag+".mask.main_nr2"), videoId, tag+".mask.main_nr2")
        
        save1(video_pre, tag+".recon")
        save1(orig.cuda()/(video_pre.abs()+10**-10), tag+".recon_non")
        save1(0.5*orig.cuda()/(video_pre.abs()+10**-10), tag+".recon_non2")



###############################################################################################################

from nir.model import Siren
from nir.util import get_mgrid, jacobian
from nir.util import Dataset,ToTensor

from nir.myLib.Layer import Layer
from nir.myLib.mySave import check,save2img,save0
from nir.myLib.VideoFitting import VideoFitting
class Layer_rigid(nn.Module):
    def __init__(self,useGlobal=True,useLocal=True,useMatrix=True,useDeformation=False,
                 deformationSize=3*0.004,hidden_features=128):
        super().__init__()
        self.useGlobal=useGlobal
        self.useLocal=useLocal
        self.useMatrix=useMatrix
        self.useDeformation=useDeformation
        self.deformationSize=deformationSize
        self.f_2D = Siren(in_features=2, out_features=1,
                          hidden_features=hidden_features, hidden_layers=4,
                          outermost_linear=True)
        self.f_2D.cuda()
        self.parameters = [self.f_2D.parameters()]
        if self.useGlobal:
            self.g_global = Siren(in_features=1, out_features=4 if self.useMatrix else 2,
                                  # hidden_features=16, hidden_layers=2,
                                  hidden_features=hidden_features, hidden_layers=4,
                                  outermost_linear=True)
            self.parameters.append(self.g_global.parameters())
            self.g_global.cuda()
        if self.useLocal:
            self.g_local = Siren(in_features=3, out_features=2,
                                 hidden_features=hidden_features, hidden_layers=4,
                                 outermost_linear=True)
            self.g_local.cuda()
            self.parameters.append(self.g_local.parameters())
    def forward(self,xyt,stage):
        t = xyt[:, [-1]]
        h_global = self.g_global(t)
        ############################################################
        # xy_ = xyt[:, :-1] + h_global #+ h_local
        if self.useMatrix:
            c =h_global
            u = xyt[:, 0]
            v = xyt[:, 1]
            if c.shape[1]==6:
                new_u = c[:,0] * u + c[:,1] * u + c[:,2]
                new_v = c[:,3] * v + c[:,4] * v + c[:,5]
            else:#4
                # 提取参数 (忽略可能的第五个参数)
                tx = c[:, 0]  # X轴位移
                ty = c[:, 1]  # Y轴位移
                rotation = torch.tensor(0)#c[:, 2]  # 旋转角度(弧度)
                scale = torch.tensor(1)#c[:, 3]  # 放缩因子
                # 计算旋转和放缩后的坐标
                cos_theta = torch.cos(rotation)
                sin_theta = torch.sin(rotation)
                # 向量化计算所有点
                u=scale*u
                v=scale*v
                new_u = (u * cos_theta - v * sin_theta) + tx
                new_v = (u * sin_theta + v * cos_theta) + ty
            # 组合成新坐标张量
            new_uv = torch.stack([new_u, new_v], dim=1)
            xy_ = new_uv
        else: #不使用变换矩阵
            xy_ = xyt[:, :-1] + h_global
        ############################################################
        if stage==0:#纹理学习，不分析局部位移
            h_local = torch.tensor(0.0)# h_local = self.g_local(xyt) if self.useLocal else torch.tensor(0.0)
        else:#只分析局部位移，不学习整体运动和纹理
            h_local = self.g_local(xyt)
            h_local = 2 * torch.sigmoid(h_local) - 1
            h_local = h_local * self.deformationSize
            xy_ = xy_ + h_local
        color = torch.sigmoid(self.f_2D(xy_))
        return color,{
            "xy_":xy_,
            "h_local":h_local,
            "h_global":h_global.clone().detach()
        }
weight_init = [10, 0.1, 10] #RSF
class Decouple(nn.Module):
    def log(self,x): #x中的全部元素均不能为负数
        import math
        e = math.e # 获取自然数 e 的值
        eps = e ** -101
        # return x
        # eps = 1e-10 #torch.finfo(torch.float64).eps
        return -1.*torch.log(x+eps) # return -1.*torch.log(x.abs()+eps)
    def __init__(self, path,maskPath="./nir/data/mask/filter",hidden_features=128):
        super().__init__()
        v = VideoFitting(path,maskPath=maskPath)
        videoloader = DataLoader(v, batch_size=1, pin_memory=True, num_workers=0)
        model_input, ground_truth, mask = next(iter(videoloader))
        model_input, ground_truth, mask = model_input[0].cuda(), ground_truth[0].cuda(), mask[0].cuda()
        ground_truth = ground_truth[:, 0:1]  # 将RGB图像转换为灰度图
        if False:
            print("gt_illu_max",torch.max(ground_truth))
            print("gt_illu_min",torch.min(ground_truth))
            ground_truth = self.log(ground_truth)
            print("gt_dis_max", torch.max(ground_truth))
            print("gt_dis_min", torch.min(ground_truth))
        # exit(0)
        self.v=v
        self.model_input=model_input
        self.ground_truth=ground_truth
        self.mask=mask

        #######################################################################

        # 一、软体
        self.NUM_soft = 0 # 软体层的数据
        self.f_soft_list = []
        for i in range(self.NUM_soft):
            self.f_soft_list.append(Layer(useGlobal=False,hidden_features=hidden_features))
        # 二、刚体
        self.NUM_figid = 0 #1# 刚体层的数据
        self.f_rigid_list=[]
        for i in range(self.NUM_figid):
            self.f_rigid_list.append(Layer(useDeformation=True,hidden_features=hidden_features))
            # self.f_rigid_list.append(Layer_rigid(
            #     useLocal=False, hidden_features=128, useMatrix=True
            # ))
        # 三、流体
        self.f2 = Siren(in_features=3, out_features=1, outermost_linear=True,
                        hidden_features=hidden_features, hidden_layers=4) #256、4
        self.f2.cuda()

        self.parameters=[
            self.f2.parameters()
        ] #+ self.f_soft.parameters #+ self.f_rigid.parameters
        for i in range(self.NUM_figid):
            self.parameters = self.parameters + self.f_rigid_list[i].parameters
        for i in range(self.NUM_soft):
            self.parameters = self.parameters + self.f_soft_list[i].parameters

    def forward(self,xyt): # soft, rigid, fluid
        # 1.刚体
        o_rigid_all = 1
        o_rigid_list = []
        h_local_list = []
        for i in range(self.NUM_figid):
            o_rigid0, p_rigid0 = self.f_rigid_list[i](xyt)#(xyt,0)
            o_rigid_all = o_rigid_all*o_rigid0
            o_rigid_list.append(o_rigid0)
            h_local_list.append(p_rigid0["h_local"])

        # 2.软体
        o_soft_all = 1
        o_soft_list = []
        for i in range(self.NUM_soft):
            o_soft0, _ = self.f_soft_list[i](xyt)
            o_soft_all = o_soft_all * o_soft0
            o_soft_list.append(o_soft0)

        # 3.流体
        o_fluid = torch.sigmoid(self.f2(xyt))
        o = o_rigid_all * o_soft_all * (1.0 - o_fluid)
        return o , {
            "r": o_rigid_list,
            "s": o_soft_list,
            "f": o_fluid
        } ,{
            "h_local":h_local_list,
            "o_rigid_all":o_rigid_all,
            "o_soft_all":o_soft_all
        }

    def loss(self, xyt, step, start, end):
        ground_truth = self.ground_truth
        mask = self.mask
        o, layers, p = self.forward(xyt)
        o_soft_list  = layers["s"]
        o_fluid = layers["f"]


        # loss_recon = ((o - self.ground_truth[start:end]) ** 2) * (10 ** 5)
        eps = 10 ** -10
        loss_recon = torch.log(
            (self.ground_truth[start:end].abs() + eps) / (o.abs() + eps)
        ).abs() * (10 ** 5)
        if self.v.useMask:
            # print("self.mask[start:end]",self.mask[start:end].shape,self.mask[start:end].max(),self.mask[start:end].min())
            loss_recon = loss_recon*self.mask[start:end]
            loss_recon = loss_recon.sum()/(self.mask[start:end].sum()+1e-8)
        else:
            loss_recon = loss_recon.mean()
        # loss_recon = ((o - ground_truth[start:end]) ** 2).mean() * (10**5)
        if False:# 不对血管区域进行重建监督
            loss_recon = (((1.0-o_fluid)*(o - ground_truth[start:end])) ** 2).mean()*10.
        # if False:
        #     loss_recon = ((self.log(o) - ground_truth[start:end]) ** 2).mean() * 10.
        # 一、刚体
        loss_rigid = 0 #刚体的局部位移尽量小
        for i in range(self.NUM_figid):
            loss_rigid = loss_rigid + p["h_local"][i].abs().mean()  # 减少刚体的信息量
        loss_rigid = loss_rigid * weight_init[0] #(10**2)
        # 二、软体
        loss_soft = 0#(1. - p["o_soft_all"]).abs().mean() * weight_init[1] #(0.1) # 减少软体的信息量
        # 三、流体 # 减少流体的信息量
        loss_fluid = 0#fluidInf(o_fluid.abs()) * weight_init[2] #1 # 减少流体的信息量

        loss = loss_recon + loss_soft + loss_fluid + loss_rigid

        self.layers=layers
        if not step % 2000:#200:
            print("Step [%04d]: loss=%0.8f, recon=%0.8f, loss_soft=%0.8f, loss_fluid=%0.8f, loss_rigid=%0.4f" % (
                step, loss, loss_recon, loss_soft , loss_fluid , loss_rigid))
            # print("i_r0:", i_r0.item(),"; i_s0:", i_s0.item(),"; i_f0:", i_f0.item(),"\ntemp",
            #       [temp[0].item(),temp[1].item(),temp[2].item()])
            # print("i_r",i_r.item(), "\twr", wr.item()) # i_f为0
            # print("i_s",i_s.item(), "\tws", ws.item())
            # print("i_f",i_f.item(), "\twf", wf.item()) #wr、ws为无穷大
            # print(self.log(torch.tensor([0.0])))
            # exit(0)
        return loss

    def train(self,total_steps):
        model_input = self.model_input

        optim = torch.optim.Adam(lr=1e-4, params = itertools.chain.from_iterable(self.parameters))
        # optim = torch.optim.Adam(lr=1e-4, params=chain(self.parameters))

        batch_size = (self.v.H * self.v.W) // 8
        for step in range(total_steps):
            start = (step * batch_size) % len(model_input)
            end = min(start + batch_size, len(model_input))

            xyt = model_input[start:end].requires_grad_()
            loss = self.loss(xyt, step,start,end)

            optim.zero_grad()
            loss.backward()
            optim.step()

    def getVideo(self):
        N, C, H, W = self.v.video.size()  # 帧数、通道数、高度、宽度
        # 创建空列表存储每帧预测结果
        pred_frames = []
        layers_frames = {
            "r": [],
            "s": [],
            "f": []
        }
        p_frames = {
            "o_rigid_all": [],
            "o_soft_all": []
        }
        if self.NUM_soft == 0 and self.NUM_figid == 0:
            layers_frames = {
                "f": []
            }
            p_frames = {}
        # 生成时间轴的归一化值（-1到1）
        t_vals = torch.linspace(-1, 1, steps=N).cuda() if N > 1 else torch.zeros(1).cuda()

        # 逐帧推理
        with torch.no_grad():
            for i in range(N):
                # 生成当前帧的空间网格 (H*W, 2)
                spatial_grid = get_mgrid([H, W]).cuda()

                # 为所有空间点添加当前时间值
                t_val = t_vals[i]
                t_column = torch.full((spatial_grid.shape[0], 1), t_val).cuda()
                coords = torch.cat([spatial_grid, t_column], dim=1)

                # 模型推理并激活
                frame_output, layers, p = self.forward(coords)

                # 调整形状为图像格式 (C, H, W)
                frame_image = frame_output.view(H, W, 1)  # .permute(2, 0, 1)
                # print("frame_image",frame_image.shape)

                pred_frames.append(frame_image)
                for id in layers_frames:
                    if id == "f":
                        layers_frames[id].append(layers[id].view(H, W, 1))
                    else:
                        layers_frames[id].append(layers[id])
                for id in p_frames:
                    p_frames[id].append(p[id].view(H, W, 1))
        # return pred_frames, layers_frames, p_frames
            video_pre = torch.stack(pred_frames, dim=0)

            def p01(original_list):
                l = list(map(list, zip(*original_list)))  # 交换列表的前两层
                for i in range(len(l)):
                    for j in range(len(l[i])):
                        l[i][j] = l[i][j].view(H, W, 1)
                        # print(type(l[i][j]),l[i][j].shape)
                        # exit(0)
                    l[i] = torch.stack(l[i], dim=0)
                return l

            if self.NUM_soft == 0 and self.NUM_figid == 0:#软体层个数和刚体层个数都为0
                layers = {
                    "r": [],
                    "s": [],
                    "f": torch.stack(layers_frames["f"], dim=0)
                }
                p = {
                    "o_rigid_all": [],
                    "o_soft_all": []
                }
            else:
                layers = {
                    "r": p01(layers_frames["r"]),
                    "s": p01(layers_frames["s"]),
                    "f": torch.stack(layers_frames["f"], dim=0)
                }
                p = {
                    "o_rigid_all": torch.stack(p_frames["o_rigid_all"], dim=0),#会增加一个新的维度，从而扩展张量的维度。所有输入的张量必须具有相同的形状。
                    "o_soft_all": torch.stack(p_frames["o_soft_all"], dim=0)
                }
            return video_pre, layers, p

def startDecouple2(videoId,paramPath,pathIn,outpath):#用视频学习网络进行拟合
    def save1(o_scene, tag):
        if o_scene==None or len(o_scene)==0: return
        o_scene = o_scene.cpu().detach().numpy()
        o_scene = (o_scene * 255).astype(np.uint8)
        save2img(o_scene[:, :, :, 0], os.path.join(outpath, tag))

    # 解耦出血管层
    if True:
        script_path = os.path.abspath(__file__)
        ROOT = os.path.dirname(script_path)
        maskPath = os.path.join(ROOT,"..",outpath, "A.mask.main_nr2","filter")
        os.makedirs(os.path.join(ROOT,"..",outpath), exist_ok=True)
        check(maskPath+"/..", videoId, "A.mask_nir2")
        myMain = Decouple(pathIn,maskPath=maskPath,hidden_features=256*4)
        myMain.train(EpochNum) #EpochNum =5000

    # 输出解耦效果
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

        if len(layers["r"])>1:
         for i in range(len(layers["r"])):
            save1(layers["r"][i], "rigid" + str(i))
        if len(layers["s"]) > 1:
         for i in range(len(layers["s"])):
            save1(layers["s"][i], "soft" + str(i))
        save1(layers["f"], "fluid")

        mainFreeCOS(paramPath,os.path.join(outpath, "recon_non2"),os.path.join(outpath, "mask2"))
        check(os.path.join(outpath, "mask2"),videoId,"nir.1.recon_non2")
        if False:
            mainFreeCOS(paramPath, os.path.join(outpath, "recon_non3"), os.path.join(outpath, "mask3"))
            check(os.path.join(outpath, "mask3"), videoId, "nir.1.recon_non3")

    # 拟合解耦出来的血管纹理
    if False: #效果不好
        from nir.myLib.ModelVessel import ModelVessel
        mySim1=ModelVessel(os.path.join(outpath,"recon_non2"),maskPath=maskPath) #只拟合血管区域
        mySim1.train(EpochNum) 
        video_pre1 = mySim1.getVideo()
        save1(video_pre1, "recon_non2_smooth_onlyVessel.01")
        # mainFreeCOS(paramPath, os.path.join(outpath, "recon_non2_smooth"), os.path.join(outpath, "mask2_"))
        # check(os.path.join(outpath, "mask2_"), videoId, "nir.1.recon_non2_smooth")

# def startDecouple2_sim(videoId,paramPath,pathIn,outpath):#用没有去除刚体的数据进行拟合
def startDecouple2_sim(videoId,paramPath,pathIn,outpath,config=None,origVideoPath=None):
    #设置参数
    myEpochNum = EpochNum
    tag = "A"
    useSmooth=False
    openLocalDeform=False
    weight_smooth=1
    if not config is None:
        if "epoch" in config:
            myEpochNum = config ["epoch"]
        if "tag" in config:
            tag = config["tag"]
        if "useSmooth"in config:
            useSmooth=config["useSmooth"]
        if "openLocalDeform"in config:
            openLocalDeform=config["openLocalDeform"]
        if "weight_smooth" in config:
            weight_smooth=config["weight_smooth"]

    def save1(o_scene, tag):
        if o_scene==None or len(o_scene)==0: return
        o_scene = o_scene.cpu().detach().numpy()
        o_scene = (o_scene * 255).astype(np.uint8)
        save2img(o_scene[:, :, :, 0], os.path.join(outpath, tag))

    #分割原始数据中的血管
    maskPath=os.path.join(outpath, tag+".orig_mask")
    if True:
        # print("maskPath",maskPath)
        mainFreeCOS_sim(paramPath,origVideoPath,maskPath)
        # exit(0)

    # 解耦出血管层
    if True:
        # script_path = os.path.abspath(__file__)
        # ROOT = os.path.dirname(script_path)
        # maskPath = os.path.join(ROOT,"..",outpath, "A.mask.main_nr2","filter")
        # os.makedirs(os.path.join(ROOT,"..",outpath), exist_ok=True)
        # check(maskPath+"/..", videoId, "A.mask_nir2")
        from nir.myLib.Decouple import Decouple
        myMain = Decouple(#startDecouple2使用视频抽取模型，而startDecouple2_sim中软体、刚体、流体都有
            pathIn,
            maskPath=maskPath,
            hidden_features=256*4)
        myMain.train(myEpochNum) #EpochNum =5000

    # 输出解耦效果
    if True:# False:
     with torch.no_grad(): #
        orig = myMain.v.video.clone()
        N, C, H, W = orig.size()  # 帧数、通道数、高度、宽度
        orig = orig.permute(0, 2, 3, 1).detach()#.numpy()

        video_pre, layers, p =myMain.getVideo()

        save1(video_pre, tag+".recon")
        # save1(orig.cuda()/(video_pre.abs()+10**-10), tag+".recon_non")
        save1(0.5*orig.cuda()/(video_pre.abs()+10**-10), tag+".recon_non2")
        save1(p["o_rigid_all"], tag+"rigid")
        # print("rigid_all",type(p["o_rigid_all"]))
        save1(orig.cuda()/(p["o_rigid_all"].abs()+10**-10), tag+".rigid_non")
        save1(p["o_soft_all"], tag+".soft")

        # if len(layers["r"])>1:
        #  for i in range(len(layers["r"])):
        #     save1(layers["r"][i], "rigid" + str(i))
        # if len(layers["s"]) > 1:
        #  for i in range(len(layers["s"])):
        #     save1(layers["s"][i], "soft" + str(i))
        # save1(layers["f"], "fluid")

        # mainFreeCOS(paramPath,os.path.join(outpath, "recon_non2"),os.path.join(outpath, "mask2"))
        # check(os.path.join(outpath, "mask2"),videoId,"nir.1.recon_non2")
    if False: #旧的输出版本
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

        if len(layers["r"])>1:
         for i in range(len(layers["r"])):
            save1(layers["r"][i], "rigid" + str(i))
        if len(layers["s"]) > 1:
         for i in range(len(layers["s"])):
            save1(layers["s"][i], "soft" + str(i))
        save1(layers["f"], "fluid")

        mainFreeCOS(paramPath,os.path.join(outpath, "recon_non2"),os.path.join(outpath, "mask2"))
        check(os.path.join(outpath, "mask2"),videoId,"nir.1.recon_non2")
        if False:
            mainFreeCOS(paramPath, os.path.join(outpath, "recon_non3"), os.path.join(outpath, "mask3"))
            check(os.path.join(outpath, "mask3"), videoId, "nir.1.recon_non3")

    # 拟合解耦出来的血管纹理
    if False: #效果不好
        from nir.myLib.ModelVessel import ModelVessel
        mySim1=ModelVessel(os.path.join(outpath,"recon_non2"),maskPath=maskPath) #只拟合血管区域
        mySim1.train(EpochNum) 
        video_pre1 = mySim1.getVideo()
        save1(video_pre1, "recon_non2_smooth_onlyVessel.01")
        # mainFreeCOS(paramPath, os.path.join(outpath, "recon_non2_smooth"), os.path.join(outpath, "mask2_"))
        # check(os.path.join(outpath, "mask2_"), videoId, "nir.1.recon_non2_smooth")
'''
    python -m nir.new
'''
def startDecouple3(videoId,paramPath,pathIn,outpath,mytag="C"):#2号解耦方法与3号解耦方法的区别是：拟合原视频 vs 拟合去刚视频
    def save1(o_scene, tag):
        if o_scene==None or len(o_scene)==0: return
        o_scene = o_scene.cpu().detach().numpy()
        o_scene = (o_scene * 255).astype(np.uint8)
        save2img(o_scene[:, :, :, 0], os.path.join(outpath, tag))

    # 解耦出血管层
    if True:
        script_path = os.path.abspath(__file__)
        ROOT = os.path.dirname(script_path)
        maskPath = os.path.join(ROOT,"..",outpath, "A.mask.main_nr2","filter")
        os.makedirs(os.path.join(ROOT,"..",outpath), exist_ok=True)
        check(maskPath+"/..", videoId, "A.mask_nir2")
        myMain = Decouple(
            os.path.join(ROOT,"..",outpath, "A.rigid.main_non1"),
            maskPath=maskPath,hidden_features=256*4)
        myMain.train(EpochNum) #EpochNum =5000

    # 输出解耦效果
    if True:# False:
     with torch.no_grad(): #
        orig = myMain.v.video.clone()
        N, C, H, W = orig.size()  # 帧数、通道数、高度、宽度
        orig = orig.permute(0, 2, 3, 1).detach()#.numpy()

        video_pre, layers, p =myMain.getVideo()

        save1(video_pre, mytag+".recon")
        save1(orig.cuda()/(video_pre.abs()+10**-10), mytag+".recon_non")
        save1(0.5*orig.cuda()/(video_pre.abs()+10**-10), mytag+".recon_non2")
        save1(p["o_rigid_all"], mytag+".rigid")
        save1(p["o_soft_all"], mytag+".soft")

        if len(layers["r"])>1:
         for i in range(len(layers["r"])):
            save1(layers["r"][i], mytag+".rigid" + str(i))
        if len(layers["s"]) > 1:
         for i in range(len(layers["s"])):
            save1(layers["s"][i], mytag+".soft" + str(i))
        save1(layers["f"], mytag+".fluid")

        mainFreeCOS(paramPath,os.path.join(outpath, mytag+".recon_non2"),os.path.join(outpath, mytag+".mask2"))
        check(os.path.join(outpath, mytag+".mask2"),videoId,mytag+".nir.1.recon_non2")

def startDecouple4(videoId,paramPath,outpath="",mytag="D",maskPath=""):#2号解耦方法与3号解耦方法的区别是什么？
    # mytag="D"
    def save1(o_scene, tag):
        if o_scene==None or len(o_scene)==0: return
        o_scene = o_scene.cpu().detach().numpy()
        o_scene = (o_scene * 255).astype(np.uint8)
        save2img(o_scene[:, :, :, 0], os.path.join(outpath, tag))

    # 解耦出血管层
    if True:
        script_path = os.path.abspath(__file__)
        ROOT = os.path.dirname(script_path)
        os.makedirs(os.path.join(ROOT,"..",outpath), exist_ok=True)
        myMain = Decouple(
            os.path.join(ROOT,"..",outpath, "A.rigid.main_non1"),
            maskPath=maskPath,hidden_features=256*4)
        myMain.train(EpochNum) #EpochNum =5000

    # 输出解耦效果
    if True:# False:
     with torch.no_grad(): #
        orig = myMain.v.video.clone()
        N, C, H, W = orig.size()  # 帧数、通道数、高度、宽度
        orig = orig.permute(0, 2, 3, 1).detach()#.numpy()

        video_pre, layers, p =myMain.getVideo()

        save1(video_pre, mytag+".recon")
        save1(orig.cuda()/(video_pre.abs()+10**-10), mytag+".recon_non")
        save1(0.5*orig.cuda()/(video_pre.abs()+10**-10), mytag+".recon_non2")
        save1(p["o_rigid_all"], mytag+".rigid")
        save1(p["o_soft_all"], mytag+".soft")

        if len(layers["r"])>1:
         for i in range(len(layers["r"])):
            save1(layers["r"][i], mytag+".rigid" + str(i))
        if len(layers["s"]) > 1:
         for i in range(len(layers["s"])):
            save1(layers["s"][i], mytag+".soft" + str(i))
        save1(layers["f"], mytag+".fluid")

        mainFreeCOS(paramPath,os.path.join(outpath, mytag+".recon_non2"),os.path.join(outpath, mytag+".mask2"))
        check(os.path.join(outpath, mytag+".mask2"),videoId,mytag+".nir.1.recon_non2")
    


if False: #if __name__ == "__main__":
    '''
    print("01:test", "可行性测试、结果正确(正确？很久之前的测试，当时的main.py部分的代码还没有出BUG)")
    #'血管层去除'中MASK遮挡的范围可以大一些  
    print("02:拟合背景的时候血管遮挡范围不考虑连通约束", "(main.py中的标准差写成了方差、现已改正)，最终效果非常好，竟然都不需要微调FreeCOS")
    print("03:钢体更硬、表达能力更强、无损失约束", "F1比02下降了一个点(查全率升高、查准率下降)")#84-83
    print("04:(1)使用旧版刚体(2)均方重构损失改为对数重构损失[更关注黯的区域]", "无明显变化") #84-84
    print("05:不对软体和流体进行损失监督", "刚体无用、软体用处不大")
    print("06:刚体和软体的数量为0", "效果没有太大变化；重构损失3442")
    print("07:增加隐含特征数量256=>256*4", "重构损失3839")#失败
    print("08:隐含特征数量256，隐含层数4=>4*4", "失败:没有正常重构出视频") #失败 
    '''
    print("09:", "")
    inpath  = "../DeNVeR_in/xca_dataset/CVAI-1207/images/CVAI-1207LAO44_CRA29"
    outpath = "../DeNVeR_in/xca_dataset/CVAI-1207/decouple/CVAI-1207LAO44_CRA29"
    videoId = "CVAI-2855LAO26_CRA31"
    # inpath  = 'nir/data/in2'
    # outpath = './nir/data/new_09_04'
    # videoId = "CVAI-LAO_CRA"
    paramPath = "../DeNVeR_in/models_config/freecos_Seg.pt"
    startDecouple1(videoId,paramPath, inpath, outpath)#去除刚体层
    startDecouple2(videoId,paramPath, inpath, outpath)#获取流体层
if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    ROOT1 = os.path.dirname(script_path)
    file_path = os.path.join(ROOT1, "../",'confs/newConfig.yaml')
    # 打开并读取 YAML 文件
    import yaml
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        print("notes:",config["my"]["notes"])
        rootPath = config["my"]["filePathRoot"]

        userId = "CVAI-2186"
        videoId = userId+"RAO28_CAU28"
        # frameId = "00055"
        inpath  = "../DeNVeR_in/xca_dataset/"+userId+"/images/"+videoId
        outpath = rootPath+"/decouple2" #"../DeNVeR_in/xca_dataset/"+userId+"/decouple/"+videoId

        paramPath = "../DeNVeR_in/models_config/freecos_Seg.pt"
        startDecouple1(videoId,paramPath, inpath, outpath)#去除刚体层
        # startDecouple2(videoId,paramPath, inpath, outpath)#获取流体层

    