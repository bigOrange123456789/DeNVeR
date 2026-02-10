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
from nir.util import get_mgrid, jacobian, hessian, hessian_vectorized
from nir.util import Dataset,ToTensor

from nir.myLib.Decouple import Decouple
from nir.myLib.mySave import check,save2img,save0
#####################################################################
from nir.myLib.Layer import Layer,Layer2,Tex2D,Layer_video, Layer_rigid

#####################################################################

# outpath = './nir/data/removeRigid_27'
# EpochNum = 6000 #5000 #3000

import torch
import gc
def memoryOpt():
    # 清空未使用的缓存
    torch.cuda.empty_cache()
    gc.collect()

    # 如果还在报错，强制同步并查看实际占用
    torch.cuda.synchronize()
    print(f"当前已分配: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"缓存占用: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
#########################################################################

from nir.myLib.VideoFitting import VideoFitting

weight_target = [0.25,100,400] #RSF #信息量比例的目标
weight_init = [10, 0.1, 10] #RSF
import torch
import numpy as np

from nir.myLib.GradientMonitor import GradientMonitor

class Decouple_rigid(nn.Module):
    def log(self,x): #x中的全部元素均不能为负数
        import math
        e = math.e # 获取自然数 e 的值
        eps = e ** -101
        # return x
        # eps = 1e-10 #torch.finfo(torch.float64).eps
        return -1.*torch.log(x+eps) # return -1.*torch.log(x.abs()+eps)
    
    def _updateMask(self, step):
        if not self.dynamicVesselMask:
            return
        def save1(o_scene, tag):
            if o_scene==None or len(o_scene)==0: return
            o_scene[o_scene>1]=1#添加这个操作来去除黑点，否则超过上限后颜色值会变为黑色
            o_scene = o_scene.cpu().detach().numpy()
            o_scene = (o_scene * 255).astype(np.uint8)
            save2img(o_scene[:, :, :, 0], os.path.join(self.updateMaskConfig["outpath"], tag))
        if self.NUM_rigid<=0:
            print("ERR:没有刚体")
            exit(0)
            return False
        step0 = step-self.dynamicVesselMask["startStep"]#["startEpoch"]
        if step0>=0 and ((step0%self.dynamicVesselMask['intervalStep'])==0):#['intervalEpoch'])==0):
            orig = self.v.video.clone()
            orig = orig.permute(0, 2, 3, 1).detach()
            video_pre, layers, p = self.getVideo(1)#使用局部形变
            rigid_non1 = orig.cuda() / (p["o_rigid_all"].abs() + 10 ** -10)
            # 将去噪结果存入 tag+rigid.non1
            save1(rigid_non1, self.updateMaskConfig["tag"]) 
            # 更新分割图
            mainFreeCOS_sim=self.updateMaskConfig['mainFreeCOS_sim']
            mainFreeCOS_sim(
                None,#self.updateMaskConfig['paramPath'],
                self.updateMaskConfig["pathInNew"],#tag+rigid.non1
                self.updateMaskConfig['maskPath'],
                Segment_model=self.updateMaskConfig["Segment_model"]
            )
            print("完成了分割测试")
            # exit(0)
            
            # 使用原来的mask路径重新加载
            self.v.reload_mask()  
            
            # 重新获取DataLoader中的数据
            videoloader = DataLoader(self.v, batch_size=1, pin_memory=True, num_workers=0)
            _,_, mask = next(iter(videoloader))
            mask = mask[0].cuda() #model_input, ground_truth, mask = model_input[0].cuda(), ground_truth[0].cuda(), mask[0].cuda()
            # ground_truth = ground_truth[:, 0:1]  # 将RGB图像转换为灰度图

            self.mask=mask

        return True


    def __init__(self, path,hidden_features=128,useSmooth=False,openLocalDeform=False,
                 weight_smooth=1,weight_concise=0.0001,weight_component=1,
                #  stillness=False, #废弃
                 stillnessFristLayer=True,
                 NUM_soft=0,NUM_rigid=2,NUM_fluid=0,
                 useMask=False,openReconLoss_rigid=False,
                 lossType=1,
                 lossFunType=None,
                 maskPath=None,
                 interval=1.0,#计算平滑损失时的偏导数计算中的步长
                #  useMatrix=True,#整体运动最好不直接用矩阵，矩阵优化的分析较为复杂
                configRigid=None,
                configRigids={},
                configSofts={},
                configFluids={},
                adaptiveFrameNumMode=0,
                use_dynamicFeatureMask=False,
                dynamicVesselMask=False,
                updateMaskConfig=None, #dynamicVesselMask不为False的时候才启用
                 ):
        super().__init__()
        self.dynamicVesselMask=dynamicVesselMask
        self.updateMaskConfig=updateMaskConfig
        self.use_dynamicFeatureMask = use_dynamicFeatureMask
        self.maskPath=maskPath
        v = VideoFitting(path,useMask=useMask,maskPath=maskPath)
        print("num_frames:",v.num_frames)
        if adaptiveFrameNumMode==1:
            # configRigids["configRigid"]
            # configRigids["layer"]['hidden_features_map']
            configSofts["layer"]["hidden_features_map"]
            configSofts["layer"]["hidden_features_global"]
            configSofts["layer"]["hidden_features_local"]

        videoloader = DataLoader(v, batch_size=1, pin_memory=True, num_workers=0)
        model_input, ground_truth, mask = next(iter(videoloader)) # 坐标、灰度、分割结果
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
        self.useSmooth=useSmooth#一个布尔参数，用来决定本次测试是否使用基于雅可比矩阵的平滑损失函数
        self.openLocalDeform=openLocalDeform
        self.weight_smooth=weight_smooth
        self.weight_concise=weight_concise
        self.weight_component=weight_component
        # self.stillness=stillness#是否静止不动 #废弃
        self.openReconLoss_rigid=openReconLoss_rigid#是否使用刚体自身的重构损失(只优化刚体层、不影响软体层,无MASK)
        self.lossType=lossType
        self.configFluids=configFluids
        if "loss_recon_all_type" in configRigids:
            print("废弃的冗余参数:loss_recon_all_type(现已替换为lossFunType[ra])")
        self.lossFunType=lossFunType
        # self.loss_recon_all_type=configRigids["loss_recon_all_type"]
        self.gradualImageLayers = configFluids["gradualImageLayers"] if "gradualImageLayers" in configFluids else None
        
        #######################################################################

        # 一、软体
        self.NUM_soft = NUM_soft#0#2 # 软体层的数据
        self.f_soft_list = []
        self.f_soft_mask_list = [] # 软体层的遮挡层
        self.useSoftMask = configSofts["useSoftMask"]
        for i in range(self.NUM_soft):
            # self.f_soft_list.append(Layer(useGlobal=False,hidden_features=hidden_features))
            self.f_soft_list.append(Layer2(#第二版软体层代码添加了流体层的PE和渐进式featureMask功能
                # useGlobal=False, 
                # useLocal=configSofts["useLocal"],
                # hidden_features=512,
                config=configSofts["layer"],
                use_dynamicFeatureMask=use_dynamicFeatureMask,
                deformationSize=3*(2/(self.v.video.size()[2] - 1)),
                ))
            if self.useSoftMask:
                self.f_soft_mask_list.append(
                    Layer_video(configSofts["layerMask"])
                )
            # self.f_soft_list.append(Layer_rigid(
            #     useGlobal=False, 
            #     hidden_features=512,
            #     config=configSofts["layer"]
            #     ))
        # 二、刚体
        self.NUM_rigid = NUM_rigid#2#20 # 刚体层的数据
        self.f_rigid_list=[]
        for i in range(self.NUM_rigid):
            # self.f_rigid_list.append(Layer(useDeformation=True,hidden_features=512,deformationSize=1.5))
            if False: self.f_rigid_list.append(Layer(useLocal=False, hidden_features=512))
            if False: self.f_rigid_list.append(Layer_rigid(useLocal=False, hidden_features=128,useMatrix=False))
            # useLocal = True
            # if "useLocal" in configRigids:
            #     useLocal = configRigids["useLocal"]
            # self.f_rigid_list.append(Layer_rigid(
            #     useLocal=useLocal, 
            #     deformationSize=3*(2/(self.v.video.size()[2] - 1)),
            #     # hidden_features=128, 
            #     useMatrix=True,
            #     useSmooth=self.useSmooth,
            #     stillness=stillness,
            #     v=self.v,
            #     interval=interval,
            #     config=configRigid,
            #     use_dynamicFeatureMask=use_dynamicFeatureMask,
            # ))
            self.f_rigid_list.append(Layer2(
                config=configRigids["layer"],
                use_dynamicFeatureMask=use_dynamicFeatureMask,
                deformationSize=3*(2/(self.v.video.size()[2] - 1)),
                # useMatrix=True,
                useSmooth=self.useSmooth,
                # stillness=stillness,#废弃
                v=self.v,
                interval=interval,
                
            ))
        if self.NUM_rigid>0:
            self.f_rigid_list[0].stillness = stillnessFristLayer
            # useGlobal=True,useLocal=True,useMatrix=True,useDeformation=False
        # 三、流体
        self.NUM_fluid = NUM_fluid
        # if NUM_fluid>0:
        #     self.f2 = Layer_video(
        #         configFluids
        #     )
        self.f_fluid_list=[]
        for i in range(self.NUM_fluid):
            # self.f_fluid_list.append(Layer_video(configFluids))
            # configFluids["dynamicTex"]=True
            self.f_fluid_list.append(
                # Layer_video(configFluids)
                Layer2(#第二版软体层代码添加了流体层的PE和渐进式featureMask功能
                    # useGlobal=False, 
                    # useLocal=False,
                    config=configFluids["layer"],#configFluids,
                    use_dynamicFeatureMask=False,
                    deformationSize=3*(2/(self.v.video.size()[2] - 1)),
                )
            )
        if NUM_fluid>0:
            self.f2 = self.f_fluid_list[0]


        self.parameters=[
            # self.f2.parameters()
        ] #+ self.f_soft.parameters #+ self.f_rigid.parameters
        for i in range(self.NUM_rigid): #self.parameters = [self.f_2D.parameters()]
            self.parameters = self.parameters + self.f_rigid_list[i].parameters
        for i in range(self.NUM_soft):
            self.parameters = self.parameters + self.f_soft_list[i].parameters
            if self.useSoftMask:
                self.parameters = self.parameters + self.f_soft_mask_list[i].parameters
        # if NUM_fluid>0:
        #     # self.parameters = self.parameters + [self.f2.parameters()]
        #     self.parameters = self.parameters + self.f2.parameters
        for i in range(self.NUM_fluid):
            self.parameters = self.parameters + self.f_fluid_list[i].parameters
        
    def getMoveDis(self, i):#获取第i层刚体的整体位移的积分
        N, C, H, W = self.v.video.size()  # 帧数、通道数、高度、宽度
        n = N//2
        t = torch.arange(n, dtype=torch.float32)/(2*N)-0.5
        xyt = torch.cat([torch.zeros(n, 2, dtype=torch.float32),
                         t.unsqueeze(-1) ], dim=-1)
        o_rigid0, p_rigid0 = self.f_rigid_list[i](xyt.cuda(),0)#stage=0 纹理学习，不分析局部位移
        h_global = p_rigid0["h_global"] #这几个点的全局位移
        h_global_x = h_global[:, 0]
        h_global_y = h_global[:, 1]
        return torch.std(h_global_x)+torch.std(h_global_y)
    
    def getMainRigidIndex(self):#获取全局形变最少的刚体层
        i=0
        s=10**10
        for i0 in range(self.NUM_rigid):
            s0 = self.getMoveDis(i0)
            if s0<s:
                s=s0
                i=i0
        return i 
    
    def forward(self,xyt,
                stage, #原本用于刚体的openLocalDeform参数，现已经废弃 
                step ,epochs0, vesselMask=None): # soft, rigid, fluid
        def get_mixing_alpha_old(current_step=None, warmup_steps=None, i=None, total_layers=None):
            """
            基于线性渐进策略计算第i层的混合权重alpha_i。
            
            策略说明：
            - 每个层按照其深度顺序依次启动
            - 浅层(i小)先启动，深层(i大)后启动
            - 每个层在启动后，alpha从0线性增加到1
            
            Args:
                current_step (int): 当前训练步数
                warmup_steps (int): 总热身步数
                i (int): 当前神经网络层编号（从0开始）
                total_layers (int): 模型总层数（i的最大值加1）
                
            Returns:
                float: 混合权重alpha，介于0和1之间
            """
            if current_step is None or warmup_steps is None:
                return 1.0
            # 确保步数在合理范围内
            current_step = min(current_step, warmup_steps)
            
            # 如果总层数为0或1，所有层使用相同的启动时间
            if total_layers <= 1:
                # 单层或零层情况，直接线性增长
                if current_step >= warmup_steps:
                    return 1.0
                else:
                    return current_step / warmup_steps
            
            # 计算每层的启动间隔
            layer_interval = warmup_steps / total_layers
            
            # 计算当前层的启动步数
            layer_start_step = i * layer_interval
            
            # 如果当前步数还未达到该层的启动步数，完全禁用
            if current_step < layer_start_step:
                return 0.0
            
            # 如果当前步数超过热身阶段，完全启用
            if current_step >= warmup_steps:
                return 1.0
            
            # 计算该层已经启动的步数
            active_steps = current_step - layer_start_step
            
            # 计算该层的有效热身步数
            layer_effective_warmup = warmup_steps - layer_start_step
            
            # 线性增长：从0到1
            alpha = active_steps / layer_effective_warmup
            
            # 确保alpha在[0,1]范围内
            return max(0.0, min(1.0, alpha))
        
        def get_mixing_alpha(current_step=None, warmup_steps=None, i=None, total_layers=None):
            """
            基于离散三阶段策略计算第i层的混合权重alpha1和alpha2。
            
            策略说明：
            - 情况1：没训练之前该模块直接输出1，所以 alpha1=0、alpha2=0
            - 情况2：训练的时候全部计算梯度，所以alpha1=1、alpha2=0
            - 情况3：训练完成该层固定输出不计算梯度，所以alpha1=0、alpha2=1
            
            Args:
                current_step (int): 当前训练步数
                warmup_steps (int): 总热身步数（用于确定训练阶段）
                i (int): 当前神经网络层编号（从0开始）
                total_layers (int): 模型总层数（i的最大值加1）
                
            Returns:
                tuple: (alpha1, alpha2) - 需要计算梯度的比例和不用计算梯度的比例
            """
            if current_step is None or warmup_steps is None:
                return 0, 1
            # 确保步数在合理范围内
            current_step = min(current_step, warmup_steps)
            # 计算每个层的训练阶段长度
            stage_length = warmup_steps // total_layers if total_layers > 0 else warmup_steps
            
            # 计算当前层所处的阶段
            if stage_length == 0:
                # 如果阶段长度为0，则所有层同时处理
                if current_step == 0:
                    # 情况1：没训练之前
                    return 0, 0
                elif current_step < warmup_steps:
                    # 情况2：训练中
                    return 1, 0
                else:
                    # 情况3：训练完成
                    return 0, 1
            else:
                # 分层处理：每个层有自己的训练时间窗口
                layer_start = i * stage_length
                layer_end = (i + 1) * stage_length
                
                if current_step < layer_start:# 情况1：该层尚未开始训练
                    # return 0, 0
                    return 0.005, 0 # o_fluid0 , detach()
                elif layer_start <= current_step < layer_end:# 情况2：该层正在训练中
                    return 1, 0
                else: # 情况3：该层训练已完成
                    # return 0, 1 # o_fluid0 , detach()
                    return 0.005, 0.995 # o_fluid0 , detach()
                # if current_step < layer_start:# 情况1：该层尚未开始训练
                #     return 0, 0 # o_fluid0 , detach()
                # elif layer_start <= current_step < layer_end:# 情况2：该层正在训练中
                #     return 1, 0
                # else: # 情况3：该层训练已完成
                #     return 0.005, 0.995 # o_fluid0 , detach()
        
        # 1.刚体
        o_rigid_all = 1
        o_rigid_list = []
        h_local_list = []
        h_global_list=[]
        h_global = torch.tensor(0.0)
        loss_smooth = torch.tensor(0.0)
        loss_conciseR = torch.tensor(0.0)
        for i in range(self.NUM_rigid):
            # step = 2/(self.v.video.size()[0] - 1)
            # o_rigid0, p_rigid0 = self.f_rigid_list[i](xyt,openLocalDeform=stage)#因为这的stage为0，所以并没有真正使用软体形变
            o_rigid0, p_rigid0 = self.f_rigid_list[i](xyt,epochs0)#(xyt,step)
            o_rigid_all = o_rigid_all*o_rigid0
            o_rigid_list.append(o_rigid0)
            h_local_list.append(p_rigid0["h_local"])
            h_global_list.append(p_rigid0["h_global"])
            h_global = p_rigid0["h_global"]
            loss_smooth = loss_smooth + p_rigid0["loss_smooth"]
            if self.use_dynamicFeatureMask:
                k = self.f_rigid_list[i].kFeatureMask()
                loss_conciseR = loss_conciseR + (k**2)
        if self.NUM_rigid>0: #将UV扩充为UVT，用于后面的软体运动叠加操作
            h_global = torch.cat([h_global, torch.zeros(h_global.shape[0], 1).cuda()], dim=1)

        # 2.软体
        o_soft_all = 1
        o_soft_list = []
        o_soft_mask_list = []
        loss_conciseS = torch.tensor(0.0)
        for i in range(self.NUM_soft):
            o_soft0, _ = self.f_soft_list[i](xyt+h_global, step) #软体运动基于刚体运动
            if self.useSoftMask:#如果使用了软体遮挡
                o_soft_mask = self.f_soft_mask_list[i](xyt,epochs0)#(xyt, step)
                o_soft0 = o_soft0 * o_soft_mask + ( 1 - o_soft_mask )
                o_soft_mask_list.append( o_soft_mask )
            o_soft_all = o_soft_all * o_soft0
            o_soft_list.append(o_soft0)
            if self.use_dynamicFeatureMask:
                k = self.f_soft_list[i].kFeatureMask()
                loss_conciseS = loss_conciseS + (k**2)

        # 3.流体
        # o_fluid = 1 #
        o_fluid_all = 1
        o_fluid_list = []
        for i in range(self.NUM_fluid):
            o_fluid0,_ = self.f_fluid_list[i](xyt,epochs0)#(xyt, step)
            # print("o_fluid0",o_fluid0.shape)
            # print("self.gradualImageLayers",self.gradualImageLayers)
            # exit(0)
            if self.gradualImageLayers:
                # alpha = self.get_mixing_alpha(i, step)
                # alpha = get_mixing_alpha_old(
                #     current_step=step, 
                #     warmup_steps=self.gradualImageLayers["warmup_steps"], 
                #     i=i, 
                #     total_layers=self.NUM_fluid)
                # o_fluid0 = alpha*o_fluid0+(1-alpha)*1
                alpha1,alpha2 = get_mixing_alpha(
                    current_step=step, 
                    warmup_steps=self.gradualImageLayers["warmup_steps"], 
                    i=i, 
                    total_layers=self.NUM_fluid)
                o_fluid0 = alpha1*o_fluid0 + alpha2*o_fluid0.detach().clone() +(1-alpha1-alpha2)*1
            o_fluid_all = o_fluid_all * o_fluid0
            # if True:#使用流体层遮挡
            #     self.mask[start:end] + self.lossFunType["rv_eps"]
            o_fluid_list.append(o_fluid0)
        # if "vesselMaskInference" in self.configFluids and self.configFluids["vesselMaskInference"] and not vesselMask is None:
        if "vesselMaskInference" in self.configFluids and self.configFluids["vesselMaskInference"]:
            if vesselMask is None:
                print("ERROR: vesselMask is None")
                exit(0)
            # vesselMask_clamp = torch.clamp(vesselMask,min=0.1,max=1)
            vesselMask_clamp = torch.clamp(vesselMask,min=self.lossFunType["vesselMask_eps"],max=1)
            # print("vesselMaskClamp",vesselMask_clamp.shape)
            # print("o_fluid_all", o_fluid_all.shape)
            # print("o_fluid_all",o_fluid_all)
            o_fluid_all = o_fluid_all*vesselMask_clamp + 1*(1-vesselMask_clamp)
            # print("o_fluid_all", o_fluid_all.shape)

        if self.NUM_fluid>0 and len(o_fluid_list)==1:
            o_fluid = o_fluid_list[0]#self.f2(xyt) 
            # print("o_fluid:",o_fluid)
            if torch.isnan(o_fluid.mean()):
                print("o_fluid is nan",o_fluid.mean())
                total_norm = torch.nn.utils.clip_grad_norm_(self.f2.parameters(), float('inf'))
                if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 1e4:
                    print('>>> 出现梯度爆炸！', total_norm)
                else:
                    print('>>> 没有梯度爆炸！', total_norm)

        o = o_rigid_all * o_soft_all * o_fluid_all
        # print("o_rigid_all",o_rigid_all.shape)
        # print("o_soft_all", o_soft_all.shape)
        # print("o_fluid_all", o_fluid_all.shape)
        # print("o",o.shape)
        # o = o_rigid_all * o_soft_all # o = o_rigid_all * o_soft_all * (1.0 - o_fluid)
        return o , {
            "r": o_rigid_list,
            "s": o_soft_list,
            "f": o_fluid_list,
            # "f_old": o_fluid,
            "o_soft_mask_list":o_soft_mask_list, #为了省事，这个对象在第二参数和第三参数里面都有输出
        } ,{
            "h_local_list":h_local_list,#这里的原始版本是"h_local_list":h_local_list,现在修改后要注意兼容性问题
            "h_global_list":h_global_list,
            "o_rigid_all":o_rigid_all,
            "o_soft_all":o_soft_all,
            "o_fluid_all":o_fluid_all,
            "loss_smooth":loss_smooth,
            "loss_concise" :loss_conciseR + loss_conciseS,
            "loss_conciseR":loss_conciseR,
            "loss_conciseS":loss_conciseS,
            # "o_soft_mask_list":o_soft_mask_list, #在第二参数中用于训练的loss2, 在第三参数中用于推理的getVideo
            
        }#输出三样东西：重构结果、分层结果、相关参数

    def loss(self, xyt, step,epochs0, start, end,openLocalDeform,lossParam):
        if self.lossType==1:
            return self.loss1(xyt, step,epochs0, start, end,openLocalDeform)
        elif self.lossType==2:
            return self.loss2(xyt, step,epochs0, start, end,openLocalDeform,lossParam)

    def loss1(self, xyt, step,epochs0, start, end,openLocalDeform):
        o, layers, p = self.forward(xyt,openLocalDeform)#纹理学习
        #局部

        # 一、整体重构损失 loss=M*(S*R-O)
        eps=10**-10
        loss_recon = torch.log(
            (self.ground_truth[start:end].abs()+eps)/(o.abs()+eps)
        ).abs()
        # print("useMask",self.v.useMask)
        # exit(0)
        if self.v.useMask:#if False: #self.v.useMask:
            loss_recon = loss_recon*self.mask[start:end]
            loss_recon = loss_recon.sum()/(self.mask[start:end].sum()+1e-8)
        else:
            loss_recon = loss_recon.mean()
        
        # 二、刚体重构损失 loss=(R-O)
        loss_recon_rigid=0
        if self.openReconLoss_rigid:
            loss_recon_rigid = torch.log(
                (self.ground_truth[start:end].abs()+eps)/(p["o_rigid_all"].abs()+eps)
            ).abs()
            loss_recon_rigid = loss_recon_rigid.mean()
        # loss_recon = ((o - ground_truth[start:end]) ** 2).mean() * (10**5)
        if False:# 不对血管区域进行重建监督
            loss_recon = (((1.0-o_fluid)*(o - ground_truth[start:end])) ** 2).mean()*10.
        # if False:
        #     loss_recon = ((self.log(o) - ground_truth[start:end]) ** 2).mean() * 10.
        # 一、刚体
        loss_rigid = 0 #刚体的局部位移尽量小
        if False:
            for i in range(self.NUM_rigid):
                loss_rigid = loss_rigid + p["h_local_list"][i].abs().mean()  # 刚体的局部位移尽量少
            loss_rigid = loss_rigid + (p["o_rigid_all"]-0).abs().mean()#刚体层级尽量暗一些
        loss_rigid = 0 
        # 二、软体
        loss_soft = 0 # (1. - p["o_soft_all"]).abs().mean() * weight_init[1] #(0.1) # 软体的暗度
        # 三、流体 # 减少流体的信息量
        loss_fluid = 0 # fluidInf(o_fluid.abs()) * weight_init[2] #1 # 减少流体的信息量

        
        loss_smooth=0 # 刚体整体运动的平滑损失
        if self.useSmooth:
            # for i in range(self.NUM_rigid):
            #     h = p["h_global_list"][i]#全局位移(包括旋转、放缩)参数
            #     t = xyt[:, [-1]]#全局位移只受到时间的影响
            #     loss_smooth = loss_smooth+jacobian(h, t).abs().mean()#全局位移的变化尽量平缓
            #     #离谱，离了大谱，代码说这里的h和t没有构建任何计算图依赖关系
            loss_smooth = p["loss_smooth"] # 权重为 1



        loss = loss_recon + loss_recon_rigid + loss_soft + loss_fluid + loss_rigid + loss_smooth*self.weight_smooth

        self.layers = layers
        # if not loss_smooth==0:#不对劲，很不对劲，这里的平滑损失貌似一直为0 #如果梯度一直为0，说明刚体没有没有任何运动
        #     print(step,"Step [%04d]: loss=%0.8f, recon=%0.8f, loss_rigid=%0.4f, loss_smooth=%0.4f" % (
        #             step, loss, loss_recon, loss_rigid, loss_smooth))
        #     print("loss_smooth此时不为0") #平滑损失 #速度为0
        #     exit(0)
        if not step % 100:#500:#200:
            # print("Step [%04d]: loss=%0.8f, recon=%0.8f, loss_soft=%0.8f, loss_fluid=%0.8f, loss_rigid=%0.4f" % (
            #     step, loss, loss_recon, loss_soft , loss_fluid , loss_rigid))
            if self.useSmooth:
                print("Step [%04d]: loss=%0.8f, recon=%0.8f, recon2=%0.8f, loss_rigid=%0.4f, loss_smooth=%0.4f" % (
                    step, loss, loss_recon, loss_recon_rigid, loss_rigid, loss_smooth))
            else:
                print("Step [%04d]: loss=%0.8f, recon=%0.8f, recon2=%0.8f, loss_rigid=%0.4f" % (
                    step, loss, loss_recon, loss_recon_rigid, loss_rigid))
            # print("i_r0:", i_r0.item(),"; i_s0:", i_s0.item(),"; i_f0:", i_f0.item(),"\ntemp",
            #       [temp[0].item(),temp[1].item(),temp[2].item()])
            # print("i_r",i_r.item(), "\twr", wr.item()) # i_f为0
            # print("i_s",i_s.item(), "\tws", ws.item())
            # print("i_f",i_f.item(), "\twf", wf.item()) #wr、ws为无穷大
            # print(self.log(torch.tensor([0.0])))
            # exit(0)
        return loss

    def loss2_old(self, xyt, step,epochs0, start, end,openLocalDeform):
        o, layers, p = self.forward(xyt,openLocalDeform)#纹理学习
        #局部

        eps=10**-10
        R=p["o_rigid_all"]
        S=p["o_soft_all"]
        R_clone = R.detach().clone() # 避免梯度回传
        S_clone = S.detach().clone()
        # 一、软体有遮挡重构损失 loss=M*(S*R-O)
        loss_recon_soft = torch.log(
            (self.ground_truth[start:end].abs()+eps)/((S*R_clone).abs()+eps)
        ).abs()
        loss_recon_soft = loss_recon_soft*self.mask[start:end]
        loss_recon_soft = loss_recon_soft.sum()/(self.mask[start:end].sum()+1e-8)

        
        # 二、刚体无遮挡重构损失 loss=(R-O)
        loss_recon_rigid = torch.log(
                (self.ground_truth[start:end].abs()+eps)/((S_clone*R).abs()+eps)
            ).abs()
        loss_recon_rigid = loss_recon_rigid.mean()
        
        # 三、平滑损失
        loss_smooth=0 # 刚体整体运动的平滑损失
        if self.useSmooth:
            loss_smooth = p["loss_smooth"] # 权重为 1

        loss = loss_recon_soft + loss_recon_rigid + loss_smooth*self.weight_smooth

        self.layers = layers
        if not step % 100:
            if self.useSmooth:
                print("Step [%04d]: loss=%0.8f, recon_soft=%0.8f, recon_rigid=%0.8f, loss_smooth=%0.4f" % (
                    step, loss, loss_recon_soft, loss_recon_rigid, loss_smooth))
            else:
                print("Step [%04d]: loss=%0.8f, recon_soft=%0.8f, recon_rigid=%0.8f" % (
                    step, loss, loss_recon_soft, loss_recon_rigid))

        return loss

    def loss2(self, xyt, step,epochs0, start, end,openLocalDeform,lossParam={"rm":"S","ra":"R"}, batch_size_scale=1.0):
        vesselMask = self.mask[start:end]
        o, layers, p = self.forward(xyt,openLocalDeform, step ,epochs0,vesselMask = vesselMask)#纹理学习

        eps=10**-10
        R=p["o_rigid_all"]
        S=p["o_soft_all"]
        F=p["o_fluid_all"]#layers["f"]
        R_clone = R.detach().clone() if self.NUM_rigid>0 else 1
        S_clone = S.detach().clone() if self.NUM_soft>0 else 1
        F_clone = F.detach().clone() if self.NUM_fluid>0 else 1
        
        def getData_old(s0): #优化选项后面细化为: 整体运动、局部运动、纹理
            if s0=="S,R" or s0=="R,S":  return R*S  #刚体软件都优化
            elif s0=="S":  return R_clone*S #只优化软体层
            elif s0=="R":  return R*S_clone #只优化刚体层
            # elif s0=="SC*RC":  return S_clone*R_clone
            else: print("Decouple_rigid中loss函数参数错误")
        def getData(s0): #优化选项后面细化为: 整体运动、局部运动、纹理
            R2=R if len(s0.split("R")) else R_clone
            S2=S if len(s0.split("S")) else S_clone
            F2=F if len(s0.split("F")) else F_clone
            return R2*S2*F2
            
        # 一、有遮挡重构损失 loss=M*(S*R-O)
        # 1.1 背景重构损失
        loss_recon_mask = torch.tensor(0.0) #背景重构损失
        if not lossParam["rm"] is None:
            # rm_in=getData(lossParam["rm"])
            # loss_recon_mask = torch.log(
            #     (self.ground_truth[start:end].abs()+eps)/((rm_in).abs()+eps)
            # ).abs() # ground_truth是目标图像，mask是分割图
            # loss_recon_mask = loss_recon_mask*self.mask[start:end]#只重构血管？
            # loss_recon_mask = loss_recon_mask.sum()/(self.mask[start:end].sum()+1e-8)
            rm_in=getData(lossParam["rm"])
            if self.lossFunType["rm"]=="MSE":
                loss_recon_mask = ( self.ground_truth[start:end] - rm_in ) ** 2 # ground_truth是目标图像，mask是背景分割图
            else:#myLog
                loss_recon_mask = torch.log(
                    (self.ground_truth[start:end].abs()+eps)/((rm_in).abs()+eps)
                ).abs() # ground_truth是目标图像，mask是分割图
            m0 = 1-vesselMask#self.mask[start:end]
            loss_recon_mask = (loss_recon_mask*m0).sum()/(m0.sum()+1e-8)
        # 1.2 前景重构损失
        # loss_recon_vessel = torch.tensor(0.0) if lossParam["rf"] is None else dist0( #前景重构损失
        #     "myLog", 
        #     getData(lossParam["rf"]), 
        #     self.ground_truth[start:end] )
        loss_recon_vessel = torch.tensor(0.0) #血管重构损失
        if not lossParam["rv"] is None:
            rv_in=getData(lossParam["rv"])
            if self.lossFunType["rv"]=="MSE":
                loss_recon_vessel = ( self.ground_truth[start:end] - rv_in ) ** 2
            else:#myLog
                loss_recon_vessel = torch.log(
                    (self.ground_truth[start:end].abs()+eps)/((rv_in).abs()+eps)
                ).abs()
            m0 = torch.clamp(vesselMask, min=self.lossFunType["rv_eps"]) # m0 = vesselMask + self.lossFunType["rv_eps"]
            loss_recon_vessel = (loss_recon_vessel*m0).sum()/(m0.sum()+1e-8)
        
        # 二、无遮挡重构损失 loss=(R-O)
        loss_recon_all = torch.tensor(0.0) # 整体重构损失
        if not lossParam["ra"] is None:
            ra_in=getData(lossParam["ra"])
            if True:#用于输出拟合程度
             loss_recon_all_type = self.lossFunType["ra"]#self.loss_recon_all_type
             if not loss_recon_all_type=="MSE" and not step % 100:
                loss_recon_all0 = ( self.ground_truth[start:end] - ra_in )**2
                loss_recon_all0 = loss_recon_all0.mean()#这个对象在训练后期变为了None
                # print("loss_recon_all0",loss_recon_all0)
            if torch.isnan(ra_in.mean()):
                print("ra_in is nan.",ra_in.mean()) #ra_in is None. tensor(nan, device='cuda:0', grad_fn=<MeanBackward0>)
                exit(0)
            if loss_recon_all_type=="MSE":
                loss_recon_all = ( self.ground_truth[start:end] - ra_in )**2
                loss_recon_all = loss_recon_all.mean()#这个对象在训练后期变为了None
            elif loss_recon_all_type=="atten_d":#类似最大值的思想
                temperature = 0.1**2 #1.0 # 温度参数调节注意力集中程度
                # temperature越小，越关注最大误差（类似max）
                # temperature越大，越接近平均（类似mean）
                errors = ( self.ground_truth[start:end] - ra_in )**2
                attention_weights = torch.softmax(errors.clone().detach()/ temperature, dim=0) 
                loss_recon_all = (attention_weights * errors).sum()
            elif loss_recon_all_type=="atten":#类似最大值的思想(估计会不稳定)
                temperature =1.0
                # 温度参数调节注意力集中程度
                # temperature越小，越关注最大误差（类似max）
                # temperature越大，越接近平均（类似mean）
                errors = ( self.ground_truth[start:end] - ra_in )**2
                attention_weights = torch.softmax(errors/ temperature, dim=0) # 使用softmax让大误差获得更多关注
                loss_recon_all = (attention_weights * errors).sum()
            else: #myLog
                loss_recon_all = torch.log(
                    (self.ground_truth[start:end].abs()+eps)/((ra_in).abs()+eps)
                ).abs()
                loss_recon_all = loss_recon_all.mean() 
            # loss_recon_all = loss_recon_all.mean()
        
        # 三、平滑损失
        loss_smooth=0 # 刚体整体运动的平滑损失
        if self.useSmooth:
            loss_smooth = p["loss_smooth"] # 权重为 1

        loss_concise=0
        if self.use_dynamicFeatureMask:
            loss_concise = p["loss_concise"]

        # # 四、非空白损失 #blank
        # loss_blank = torch.tensor(0.0)
        # if "blank" in lossParam:
        #     if lossParam["blank"]=="F":
        #         (1-F).abs()

        # 四、分量小于总量 Any component is less than the total
        loss_component = torch.tensor(0.0) # 分量约束 #应该比重构损失要小、为啥更大？
        if self.weight_component>0:
            raw = self.ground_truth[start:end]
            loss_component = (torch.clamp(raw-R,min=0)**2 + 
                              torch.clamp(raw-S,min=0)**2 + 
                              torch.clamp(raw-F,min=0)**2)/3 #让分量变亮一些
            loss_component = loss_component.mean()

        # 五、MASK二值化损失
        loss_binaryMask_all = torch.tensor(0.0)
        for masklayer in layers["o_soft_mask_list"]:
            loss_binaryMask_all = loss_binaryMask_all + ( (masklayer-0.5)**2 ).mean()
        if len( layers["o_soft_mask_list"] ) > 0 :
            loss_binaryMask_all = loss_binaryMask_all / len( layers["o_soft_mask_list"] )
        if loss_binaryMask_all>0:
            loss_binaryMask_all = 1/loss_binaryMask_all

        loss = (loss_recon_mask + loss_recon_all + loss_recon_vessel + 
                loss_smooth * self.weight_smooth + 
                loss_concise * self.weight_concise + 
                loss_component * self.weight_component + 
                loss_binaryMask_all)



        self.layers = layers
        if (step % 200 == 0) or (step ==1999) :
            # print("loss_concise",loss_concise)
            # if self.NUM_fluid>0:
            #     total_norm = torch.nn.utils.clip_grad_norm_(self.f2.parameters(), float('inf'))# 计算全局梯度范数（不裁剪）
            #     eps = 1e-7          # 比默认的 1e-8 稍宽松，也可设 1e-6/1e-5
            #     if torch.isnan(total_norm) or torch.isinf(total_norm):
            #         print('>>> 梯度爆炸(NaN/Inf)!', total_norm)
            #     elif total_norm < eps:
            #         print('>>> 梯度消失！', total_norm)
            #     else:
            #         print('>>> 梯度正常！', total_norm)
            if self.useSmooth:
                print("Step [%04d]: loss=%0.8f, recon_mask=%0.8f, recon_all=%0.8f, loss_smooth=%0.4f" % (
                    step, loss, loss_recon_mask, loss_recon_all, loss_smooth))
            else: # 不用平滑损失
                # if self.loss_recon_all_type=="MSE": #整体重构损失
                    # print("Step [%04d]: loss=%0.8f, recon_mask=%0.8f, recon_all=%0.8f" % (
                    #     step, loss, loss_recon_mask, loss_recon_all))
                    # print("Step [%04d]: loss=%0.8f, recon_mask=%0.8f, recon_vess=%0.8f, recon_al=%0.8f, bMaskA=%0.8f, conciseR=%0.8f, conciseS=%0.8f" % (
                    #                     step, loss, loss_recon_mask, loss_recon_vessel, loss_recon_all, loss_binaryMask_all, p["loss_conciseR"], p["loss_conciseS"]))
                    print("Step [%04d]: loss=%0.8f, recon_b=%0.8f, recon_vess=%0.8f, recon_all=%0.8f, componet=%0.8f, conciseR=%0.8f, conciseS=%0.8f" % (
                           step, loss, loss_recon_mask, loss_recon_vessel, loss_recon_all, loss_component, p["loss_conciseR"], p["loss_conciseS"]))
                # else:
                #     print("Step [%04d]: loss=%0.8f, recon_mask=%0.8f, recon_all=%0.8f, recon_all0=%0.8f" % (
                #         step, loss, loss_recon_mask, loss_recon_all,loss_recon_all0))

        return loss

    def train(self,
              epochs,total_steps,
              paramLoss, batch_size_scale = 1.0 ):
        if self.NUM_fluid==1:
            gradientMonitor = GradientMonitor(self.f2)
        model_input = self.model_input

        # print("1;self.parameters",self.parameters)
        optim = torch.optim.Adam(lr=1e-4, params = itertools.chain.from_iterable(self.parameters))
        # print("2;self.parameters",self.parameters)
        # exit(0)
        # optim = torch.optim.Adam(lr=1e-4, params=chain(self.parameters))

        # batch_size =  (self.v.H * self.v.W) // 8  # 32768 #每张图片分为8个batch
        '''
            之前帧数更小为啥最后的拟合效果更好: 之前batch大小固定、帧数小的话训练训练次数更多
        '''
        # batch大小与帧数无关
        # batch_size = (self.v.H * self.v.W) // 8
        batch_size = int( batch_size_scale * self.v.H * self.v.W) 
        print("batch_size_scale:",batch_size_scale,"\tbatch_size:",batch_size)
        # batch大小与帧数有关
        # batch_size = ( self.v.H * self.v.W * self.v.num_frames) // (8*200)
        # batch_size = int( batch_size_scale * self.v.H * self.v.W ) // (8*200) #batch变小了，并且帧约小的视频训练的越少
        if not (epochs is None):    
            total_steps = int((epochs * len(model_input)) / batch_size)
        print("训练一遍所有数据需要的次数:", len(model_input) // batch_size)
        print("总共训练了多少遍数据:", total_steps*batch_size / len(model_input) )
        # exit(0)
        for step in range(total_steps): #生成纹理、整体运动
            start = (step * batch_size) % len(model_input) # len(model_input) 是像素点的总数
            end = min(start + batch_size, len(model_input))

            xyt = model_input[start:end].requires_grad_()
            loss = self.loss(xyt, 
                             step,step*batch_size/len(model_input),
                             start,end,self.openLocalDeform,paramLoss)#离谱，stage竟然为0

            optim.zero_grad()
            loss.backward()
            optim.step()
            if not step % 100 and step>0 and self.NUM_fluid==1:
                gradientMonitor.analyze(step)

            if False:
                if loss<0.1**5: #重构损失足够小的时候退出,达到这个标注后仍然有问题
                    print("step",step,";loss",loss)
                    break
            if True:
                self._updateMask(step*batch_size/len(model_input))
            else: # 旧版
                if not step==total_steps-1: #最后一次迭代就不用更新MASK了，训练都要结束了、更新也用不上了
                    self._updateMask(step) #在训练过程中更新MASK

        if self.NUM_fluid==1:
            gradientMonitor.close()

        # for i in range(self.NUM_rigid): # 遍历全部的刚体层
        #     f_2D = self.f_rigid_list[i].f_2D # 一个刚体层的纹理
        #     for param in f_2D.parameters():
        #         param.requires_grad = False  # 关闭梯度计算
        #     g_global = self.f_rigid_list[i].g_global
        #     for param in g_global.parameters():
        #         param.requires_grad = False  # 关闭梯度计算
        # for step in range(total_steps): #生成局部运动
        #     start = (step * batch_size) % len(model_input)
        #     end = min(start + batch_size, len(model_input))
        #     xyt = model_input[start:end].requires_grad_()
        #     loss = self.loss(xyt, step,start,end,stage=1)
        #     optim.zero_grad()
        #     loss.backward()
        #     optim.step()

    def getVideo(self,stage):
        print("逐帧推理")
        ####################################开始使用MASK####################################
        def get_video_tensor(path):
            """
            从指定路径加载视频帧，转换为张量
            Args:
                path: 视频帧图片所在的文件夹路径
            Returns:
                video_tensor: 形状为 (N, C, H, W) 的张量
            """
            # 获取文件夹中的所有图片文件
            frames = sorted([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            
            if not frames:
                raise ValueError(f"No image files found in directory: {path}")
            
            video = []
            
            # 如果没有设置transform，使用默认的ToTensor
            transform = self.transform if hasattr(self, 'transform') else ToTensor()
            
            for frame_name in frames:
                img_path = os.path.join(path, frame_name)
                img = Image.open(img_path).convert('L')  # 转换为灰度图，如果已经是灰度图则保持
                img = transform(img)
                
                # 确保张量有正确的通道数
                if img.dim() == 2:  # 如果只有高度和宽度，添加通道维度
                    img = img.unsqueeze(0)  # 变为 (1, H, W)
                
                video.append(img)
            
            # 将所有帧堆叠成一个张量
            return torch.stack(video, 0)
        if "vesselMaskInference" in self.configFluids and self.configFluids["vesselMaskInference"]:
            mask_video = get_video_tensor(self.maskPath)
            mask_video = mask_video.cuda()  # 移到GPU上
        else:
            mask_video=None
        ####################################完成使用MASK####################################
        memoryOpt() # 1.32GB , 1.44GB
        if True:#with torch.inference_mode(): #使用 PyTorch 1.9+ 更快的 inference_mode #1.31=>1.33
        # torch.inference_mode应该与torch.no_grad是等效的，但是inference_mode在windows环境下容易报错
            N, C, H, W = self.v.video.size()  # 帧数、通道数、高度、宽度
            # 创建空列表存储每帧预测结果
            pred_frames = []
            layers_frames = {
                # "r": [],
                # "s": [],
                # "f": []
            }
            p_frames = {
                # "o_rigid_all": [],
                # "o_soft_all": []
            }
            if self.NUM_fluid>0:
                layers_frames["f"]=[]
                p_frames["o_fluid_all"]=[]
            if self.NUM_rigid>0:
                layers_frames["r"]=[]
                p_frames["o_rigid_all"]=[]
            if self.NUM_soft>0:
                layers_frames["s"]=[]
                p_frames["o_soft_all"]=[]
                if self.useSoftMask:
                    layers_frames["o_soft_mask_list"]=[]
            # 生成时间轴的归一化值（-1到1）
            t_vals = torch.linspace(-1, 1, steps=N).cuda() if N > 1 else torch.zeros(1).cuda()

            # 逐帧推理
            with torch.no_grad():
                for i in range(N):
                    # 生成当前帧的空间网格 (H*W, 2)
                    spatial_grid = get_mgrid([H, W]).cuda() #xy

                    # 为所有空间点添加当前时间值
                    t_val = t_vals[i]
                    t_column = torch.full((spatial_grid.shape[0], 1), t_val).cuda()#t
                    coords = torch.cat([spatial_grid, t_column], dim=1)#xyt

                    ####################################开始使用MASK####################################
                    vesselMask = None if mask_video is None else mask_video[i].view(H * W, -1).cuda()  # 展平为(H*W, 1)
                    ####################################完成使用MASK####################################

                    # 模型推理并激活
                    frame_output, layers, p = self.forward(coords, stage, None, vesselMask=vesselMask) #frame_output, layers, p = self.forward(coords,stage,None)
                    # print("layers:",layers)
                    # print("p",p)
                    # exit(0)

                    # 调整形状为图像格式 (C, H, W)
                    frame_image = frame_output.view(H, W, 1)  # .permute(2, 0, 1)
                    # print("frame_image",frame_image.shape)

                    pred_frames.append(frame_image)
                    for id in layers_frames:
                        # if id == "f":
                        #     layers_frames[id].append(layers[id].view(H, W, 1))
                        # else:
                        #     layers_frames[id].append(layers[id])
                        layers_frames[id].append(layers[id])
                    for id in p_frames:
                        p_frames[id].append(p[id].view(H, W, 1))
            # return pred_frames, layers_frames, p_frames
                video_pre = torch.stack(pred_frames, dim=0)

                def p01(original_list):
                    l = list(map(list, zip(*original_list)))  # 交换列表的前两层
                    for i in range(len(l)):
                        for j in range(len(l[i])):
                            l[i][j] = l[i][j].view(H, W, 1) #torch.Size([16384, 1])=>torch.Size([128, 128, 1])
                            # print(type(l[i][j]),l[i][j].shape)
                            # exit(0)
                        l[i] = torch.stack(l[i], dim=0)
                    return l

                layers = {
                    # "r": p01(layers_frames["r"]),
                    # "s": p01(layers_frames["s"]),
                    # "f": torch.stack(layers_frames["f"], dim=0)
                }
                p = {
                    # "o_rigid_all": torch.stack(p_frames["o_rigid_all"], dim=0),
                    # "o_soft_all": torch.stack(p_frames["o_soft_all"], dim=0)
                }
                if self.NUM_fluid>0:
                    # layers["f"]=torch.stack(layers_frames["f"], dim=0)
                    layers["f"]=p01(layers_frames["f"])
                    p["o_fluid_all"]=torch.stack(p_frames["o_fluid_all"], dim=0)
                else:
                    layers["f"]=None
                if self.NUM_rigid>0:
                    layers["r"]=p01(layers_frames["r"])
                    p["o_rigid_all"]=torch.stack(p_frames["o_rigid_all"], dim=0)
                else:
                    p["o_rigid_all"]=None
                if self.NUM_soft>0:
                    layers["s"]=p01(layers_frames["s"])
                    p["o_soft_all"]=torch.stack(p_frames["o_soft_all"], dim=0)
                    if self.useSoftMask:
                        layers["softMask"]=p01(layers_frames["o_soft_mask_list"])
                        # p["o_soft_mask_list"]=torch.stack(p_frames["o_soft_mask_list"], dim=0)
                else:
                    p["o_soft_all"]=None
                '''
                    print("video_pre",video_pre.shape)
                    print("layers[s]",layers["s"][0].shape)
                    print('p["o_soft_all"][0].shape:',p["o_soft_all"][0].shape)
                    --------------------------------------------------------------
                    video_pre torch.Size([10, 128, 128, 1])
                    layers[s] torch.Size([10, 128, 128, 1])
                    p["o_soft_all"][0].shape: torch.Size([128, 128, 1])
                '''
                return video_pre, layers, p

    def getVideo_row(self, stage):
        print("逐行推理")
        ####################################开始使用MASK####################################
        def get_video_tensor(path):
            frames = sorted(
                [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

            if not frames:
                raise ValueError(f"No image files found in directory: {path}")

            video = []
            transform = self.transform if hasattr(self, 'transform') else ToTensor()

            for frame_name in frames:
                img_path = os.path.join(path, frame_name)
                img = Image.open(img_path).convert('L')
                img = transform(img)

                if img.dim() == 2:
                    img = img.unsqueeze(0)

                video.append(img)

            return torch.stack(video, 0)

        if "vesselMaskInference" in self.configFluids and self.configFluids["vesselMaskInference"]:
            mask_video = get_video_tensor(self.maskPath)
            mask_video = mask_video.cuda()
        else:
            mask_video = None
        ####################################完成使用MASK####################################

        N, C, H, W = self.v.video.size()
        # 创建空列表存储每帧预测结果
        pred_frames = []
        layers_frames = {}
        p_frames = {}

        if self.NUM_fluid > 0:
            layers_frames["f"] = []
            p_frames["o_fluid_all"] = []
        if self.NUM_rigid > 0:
            layers_frames["r"] = []
            p_frames["o_rigid_all"] = []
        if self.NUM_soft > 0:
            layers_frames["s"] = []
            p_frames["o_soft_all"] = []
            if self.useSoftMask:
                layers_frames["o_soft_mask_list"] = []

        # 生成时间轴的归一化值（-1到1）
        t_vals = torch.linspace(-1, 1, steps=N).cuda() if N > 1 else torch.zeros(1).cuda()

        # 逐帧、逐行推理
        with torch.no_grad():
            for i in range(N):
                # 为当前帧创建空容器，按行收集结果
                frame_rows = []
                frame_layers = {id: [] for id in layers_frames}
                frame_p = {id: [] for id in p_frames}

                ####################################开始使用MASK####################################
                vesselMask_frame = None if mask_video is None else mask_video[i].cuda()
                if len(vesselMask_frame.shape) == 3:  # 3通道
                    vesselMask_frame = vesselMask_frame[0]
                ####################################完成使用MASK####################################

                # 逐行处理
                for row in range(H):
                    # 生成当前行的空间网格 (W, 2)
                    y_norm = -1 + 2 * row / (H - 1) if H > 1 else 0
                    x_norm = torch.linspace(-1, 1, steps=W).cuda()
                    y_coord = torch.full((W, 1), y_norm).cuda()
                    x_coord = x_norm.view(-1, 1)
                    spatial_grid = torch.cat([x_coord, y_coord], dim=1)  # (W, 2)

                    # 为当前行添加时间值
                    t_val = t_vals[i]
                    t_column = torch.full((W, 1), t_val).cuda()
                    coords = torch.cat([spatial_grid, t_column], dim=1)  # (W, 3)

                    ####################################开始使用MASK####################################
                    vesselMask = None if vesselMask_frame is None else vesselMask_frame[row].view(W,
                                                                                                  -1).cuda()  # (W, 1)
                    ####################################完成使用MASK####################################

                    # 模型推理当前行
                    row_output, layers, p = self.forward(coords, stage, None, vesselMask=vesselMask)

                    # 收集当前行的结果
                    frame_rows.append(row_output.view(1, W, 1))  # 形状改为 (1, W, 1)

                    # 收集层的输出
                    for id in layers_frames:
                        # 根据不同的层类型处理输出
                        # print("layers[id]",len(layers[id]))
                        # print("layers[id][0]", layers[id][0].shape)
                        # exit(0)
                        for imgLayerId in range(len(layers[id])):
                            # print(layers[id][imgLayerId].shape)
                            layers[id][imgLayerId]=layers[id][imgLayerId].view(1,1,1, W, 1) # [1,1,1,128,1]<=[128, 1]
                            # print(layers[id][imgLayerId].shape)
                            # exit(0)
                        layers[id] = torch.cat(layers[id], dim=0) # 层数、帧数、高、宽、通道

                        # print(layers[id],layers[id].shape)
                        # print("layers[id]:",layers[id].shape)
                        frame_layers[id].append(layers[id])

                    # 收集物理量输出
                    for id in p_frames:
                        frame_p[id].append(p[id].view(1, W, 1))

                # 将当前帧的所有行合并成一帧
                frame_image = torch.cat(frame_rows, dim=0)  # (H, W, 1)
                pred_frames.append(frame_image)

                # 处理当前帧的各层输出
                for id in layers_frames:
                    layer_frame = torch.cat(frame_layers[id], dim=2)  #所有行合并
                    layers_frames[id].append(layer_frame)

                # 处理当前帧的物理量输出
                for id in p_frames:
                    p_frame = torch.cat(frame_p[id], dim=0)  # (H, W, 1)
                    p_frames[id].append(p_frame)
            for id in layers_frames:
                layers_frames[id] = torch.cat(layers_frames[id], dim=1)  # 所有帧合并 #[1, 10, 128, 128, 1]
        # 将所有帧堆叠起来
        video_pre = torch.stack(pred_frames, dim=0)

        def p01(original_list):
            """处理多物体输出的辅助函数"""
            return original_list
            # l = list(map(list, zip(*original_list)))  # 前两个维度是“图层、帧”是列表，后面的维度是张量
            # for i in range(len(l)):
            #     # for j in range(len(l[i])):
            #     #     l[i][j] = l[i][j].view(H, W, 1) #torch.Size([16384, 1])=>torch.Size([128, 128, 1])
            #     l[i] = torch.stack(l[i], dim=0)#所有帧叠加
            # return l

        # 构建最终输出字典
        layers = {}
        p = {}

        if self.NUM_fluid > 0:
            # print('layers_frames["f"]',type(layers_frames["f"]),len(layers_frames["f"]))
            # print(type(layers_frames["f"][0]))
            # print(layers_frames["f"][0].shape)
            # exit(0)
            layers["f"] = p01(layers_frames["f"])
            '''
                layers["f"] 1
                layers["f"][0] torch.Size([10, 128, 128, 1])
            '''
            p["o_fluid_all"] = torch.stack(p_frames["o_fluid_all"], dim=0)

        if self.NUM_rigid > 0:
            layers["r"] = p01(layers_frames["r"])
            p["o_rigid_all"] = torch.stack(p_frames["o_rigid_all"], dim=0)

        if self.NUM_soft > 0:
            layers["s"] = p01(layers_frames["s"])
            p["o_soft_all"] = torch.stack(p_frames["o_soft_all"], dim=0)
            if self.useSoftMask:
                layers["softMask"] = p01(layers_frames["o_soft_mask_list"])

        return video_pre, layers, p

    # def getVideo(self,stage):
    #     N, C, H, W = self.v.video.size()  # 帧数、通道数、高度、宽度
    #     # 创建空列表存储每帧预测结果
    #     pred_frames = []
    #     layers_frames = {
    #         "r": [],
    #         # "s": [],
    #         # "f": []
    #     }
    #     p_frames = {
    #         "o_rigid_all": [],
    #         # "o_soft_all": []
    #     }
    #     # 生成时间轴的归一化值（-1到1）
    #     t_vals = torch.linspace(-1, 1, steps=N).cuda() if N > 1 else torch.zeros(1).cuda()
    #     # 逐帧推理
    #     with torch.no_grad():
    #         for i in range(N):
    #             # 生成当前帧的空间网格 (H*W, 2)
    #             spatial_grid = get_mgrid([H, W]).cuda()
    #             # 为所有空间点添加当前时间值
    #             t_val = t_vals[i]
    #             t_column = torch.full((spatial_grid.shape[0], 1), t_val).cuda()
    #             coords = torch.cat([spatial_grid, t_column], dim=1)
    #             # 模型推理并激活
    #             frame_output, layers, p = self.forward(coords,stage)
    #             # 调整形状为图像格式 (C, H, W)
    #             frame_image = frame_output.view(H, W, 1)  # .permute(2, 0, 1)
    #             # print("frame_image",frame_image.shape)
    #             pred_frames.append(frame_image)
    #             for id in layers_frames:
    #                 if id == "f":
    #                     layers_frames[id].append(layers[id].view(H, W, 1))
    #                 else:
    #                     layers_frames[id].append(layers[id])
    #             for id in p_frames:
    #                 p_frames[id].append(p[id].view(H, W, 1))
    #     # return pred_frames, layers_frames, p_frames
    #         video_pre = torch.stack(pred_frames, dim=0)
    #         def p01(original_list):
    #             l = list(map(list, zip(*original_list)))  # 交换列表的前两层
    #             for i in range(len(l)):
    #                 for j in range(len(l[i])):
    #                     l[i][j] = l[i][j].view(H, W, 1)
    #                     # print(type(l[i][j]),l[i][j].shape)
    #                     # exit(0)
    #                 l[i] = torch.stack(l[i], dim=0)
    #             return l
    #         layers = {
    #             "r": p01(layers_frames["r"]),
    #             # "s": p01(layers_frames["s"]),
    #             # "f": torch.stack(layers_frames["f"], dim=0)
    #         }
    #         p = {
    #             "o_rigid_all": torch.stack(p_frames["o_rigid_all"], dim=0),
    #             # "o_soft_all": torch.stack(p_frames["o_soft_all"], dim=0)
    #         }
    #         return video_pre, layers, p


