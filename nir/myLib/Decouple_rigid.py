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
from nir.myLib.Layer import Layer,Tex2D,Layer_video
class Layer_rigid(nn.Module):
    def __init__(self,useGlobal=True,useLocal=True,useMatrix=True,useDeformation=False,
                 deformationSize=3*0.004,
                #  hidden_features=128,
                #  hidden_layers_map=4,
                #  hidden_layers_global=4,
                #  hidden_layers_local=4,
                #  hidden_features_map=128,
                #  hidden_features_global=128,
                #  hidden_features_local=128,
                 config=None,
                 useSmooth=False, # 是否计算平滑损失函数
                 stillness=False, # 是否静止不动
                 v=None,#原始视频数据的解析对象, 这里读取该对象是为了获取视频的尺寸长度参数
                 interval=1.0,
                 ):
        super().__init__()
        self.useGlobal=useGlobal
        self.useLocal=useLocal
        self.useMatrix=useMatrix
        self.useDeformation=useDeformation
        self.deformationSize=deformationSize
        self.useSmooth=useSmooth
        self.interval=interval
        self.stillness=stillness
        hidden_layers_map=4
        hidden_layers_global=4
        hidden_layers_local=4
        hidden_features_map=128
        hidden_features_global=128
        hidden_features_local=128
        if not config is None:
            hidden_layers_map = config["hidden_layers_map"]   
            hidden_layers_global = config["hidden_layers_global"]
            hidden_layers_local = config["hidden_layers_local"] 
            hidden_features_map = config["hidden_features_map"] 
            hidden_features_global = config["hidden_features_global"] 
            hidden_features_local = config["hidden_features_local"] 
            
        self.v=v #原始视频数据的解析对象, 这里读取该对象是为了获取视频的尺寸长度参数
        # print(2*4*512, 2)
        # print("hidden_features=",hidden_features_map, "hidden_layers=",hidden_layers_map)
        # exit(0)
        self.f_2D = Siren(in_features=2, out_features=1,
                          hidden_features=hidden_features_map, hidden_layers=hidden_layers_map,
                          outermost_linear=True)
        self.f_2D.cuda()
        self.parameters = [self.f_2D.parameters()]
        self.useTex2D=False#True
        if self.useTex2D:
            # self.tex2D = nn.Parameter(torch.zeros(512, 512))
            # self.parameters.append(self.tex2D.parameters())
            # self.tex2D = torch.zeros(1, 512, 512, requires_grad=True)
            # self.parameters.append(self.tex2D)
            self.tex2D = Tex2D(512)
            self.parameters.append(self.tex2D.parameters())
        if self.useGlobal:
            self.g_global = Siren(in_features=1, out_features=4 if self.useMatrix else 2,
                                  # hidden_features=16, hidden_layers=2,
                                  hidden_features=hidden_features_global, hidden_layers=hidden_layers_global,
                                  outermost_linear=True)
            self.parameters.append(self.g_global.parameters())
            self.g_global.cuda()
        if self.useLocal:
            self.g_local = Siren(in_features=3, out_features=2,
                                 hidden_features=hidden_features_local, hidden_layers=hidden_layers_local,
                                 outermost_linear=True)
            self.g_local.cuda()
            self.parameters.append(self.g_local.parameters())    
    
    def forward(self,xyt,openLocalDeform):#openLocalDeform原来是stage,当stage=0的时候对应open=False
        if self.stillness: #静止不动
            xy_ = xyt[:, :-1]
            h_local = torch.zeros_like(xy_)   #torch.tensor(0.0)
            h_global =torch.zeros_like(xy_)   #torch.tensor(0.0)
            loss_smooth =torch.tensor(0.0)
        else:
            t = xyt[:, [-1]]
            h_global = self.g_global(t)
            loss_smooth = torch.tensor(0.0)
            
            '''
                print("h_global:",type(h_global),h_global.shape)
                print("t:",type(t),t.shape)
                print("self.useSmooth:",self.useSmooth)
                j0=jacobian(h_global, t)
                print("j0:",type(j0),j0.shape)
                h0=hessian_vectorized(h_global, t)
                print("h0",type(h0),h0.shape)
                exit(0)
            
                h_global: <class 'torch.Tensor'> torch.Size([32768, 4])
                t: <class 'torch.Tensor'> torch.Size([32768, 1])
                self.useSmooth: 2
                j0: <class 'torch.Tensor'> torch.Size([32768, 4, 1])
                h0 <class 'torch.Tensor'> torch.Size([32768, 4, 1, 1])
            '''
            if self.useSmooth:#>0
                if self.useSmooth==1: #运动速度趋于0 #失败:无法迫使刚体正确运动
                    loss_smooth = loss_smooth + jacobian(h_global, t).abs().mean()
                elif self.useSmooth==2: #加速度趋于0
                    loss_smooth = loss_smooth + hessian_vectorized(h_global, t).abs().mean()
                elif self.useSmooth==3:#全局运动趋向于固定
                    N, C, H, W = self.v.video.size()  # 帧数、通道数、高度、宽度
                    interval0 = torch.tensor(self.interval, device=xyt.device)*2/(N-1)
                    data0 = h_global#[:,:2]#只有前两列被用到了，后两列没有用到
                    data_behind = self.g_global(t+interval0)#[:,:2] #后面
                    speed = (data_behind-data0)/interval0
                    loss_smooth = ( speed**2 ).mean() 
                elif self.useSmooth==4: # 二阶导数
                    N, C, H, W = self.v.video.size()  # 帧数、通道数、高度、宽度
                    interval0 = torch.tensor(self.interval, device=xyt.device)*2/(N-1)
                    data0 = h_global#[:,:2]#只有前两列被用到了，后两列没有用到
                    data_behind = self.g_global(t+interval0)#[:,:2] #后面
                    # data_front = self.g_global(t+interval0)#[:,:2] #前面
                    data_front = self.g_global(t-interval0)#[:,:2] #前面
                    acceleration = (data_behind-2*data0+data_front)/(interval0**2)
                    loss_smooth = (acceleration**2).mean() 
                elif self.useSmooth==5: # 加速度÷速度 #没有正确运动、反而出现了晃动问题
                    N, C, H, W = self.v.video.size()  # 帧数、通道数、高度、宽度
                    interval0 = torch.tensor(self.interval, device=xyt.device)*2/(N-1)
                    data0 = h_global#[:,:2]#只有前两列被用到了，后两列没有用到
                    data_behind = self.g_global(t+interval0)#[:,:2] #后面
                    # data_front = self.g_global(t+interval0)#[:,:2] #前面
                    data_front = self.g_global(t-interval0)#[:,:2] #前面
                    acceleration = (data_behind-2*data0+data_front)/(interval0**2)
                    speed = (data_behind-data0)/interval0 #速度的上限应该是2，速度超过2是不可想象的
                    eps = 10**-10
                    loss_smooth = ((#静止的时候就不能受力？哪有这样的道理
                        acceleration/(torch.sigmoid(speed).clone().detach()+eps)
                        )**2).mean()
                    # loss_smooth = ((
                    #     acceleration/(torch.sigmoid(speed+2).clone().detach()+eps)
                    #     )**2).mean()
                # elif self.useSmooth==6: # 加速度÷速度 #没有正确运动、反而出现了晃动问题
                #     N, C, H, W = self.v.video.size()  # 帧数、通道数、高度、宽度
                #     interval0 = torch.tensor(self.interval, device=xyt.device)*2/(N-1)
                #     # torch.rand((), device=xyt.device)
                #     data0 = self.g_global(t)# h_global#[:,:2] #只有前两列被用到了，后两列没有用到
                #     data_behind = self.g_global(t+interval0)#[:,:2] #后面
                #     data_front = self.g_global(t-interval0)#[:,:2] #前面
                #     acceleration = (data_behind-2*data0+data_front)/(interval0**2)
                #     speed = (data_behind-data0)/interval0 #速度的上限应该是2，速度超过2是不可想象的
                #     eps = 10**-10
                #     loss_smooth = ((#静止的时候就不能受力？哪有这样的道理
                #         acceleration/(torch.sigmoid(speed+2).clone().detach()+eps)
                #         )**2).mean()
                elif self.useSmooth==6: # 加速度÷速度 #没有正确运动、反而出现了晃动问题
                    N, C, H, W = self.v.video.size()  # 帧数、通道数、高度、宽度
                    interval0 = torch.tensor(self.interval, device=xyt.device)*2/(N-1)
                    t_new = t + torch.rand((), device=xyt.device)*2/(N-1) #时间序列添加随机扰动
                    data0 = self.g_global(t_new)# h_global#[:,:2] #只有前两列被用到了，后两列没有用到
                    data_behind = self.g_global(t_new+interval0)#[:,:2] #后面
                    data_front = self.g_global(t_new-interval0)#[:,:2] #前面
                    acceleration = (data_behind-2*data0+data_front)/(interval0**2)
                    speed = (data_behind-data0)/interval0 #速度的上限应该是2，速度超过2是不可想象的
                    eps = 10**-10
                    loss_smooth = ((#静止的时候就不能受力？哪有这样的道理
                        acceleration/(torch.sigmoid(speed+2).clone().detach()+eps)
                        )**2).mean()
                # elif self.useSmooth==7: # 多尺度斜率恒定的约束
                #     N, C, H, W = self.v.video.size()  # 帧数、通道数、高度、宽度
                #     interval0 = torch.tensor(self.interval, device=xyt.device)*2/(N-1)
                #     data0 = h_global
                #     data_behind = self.g_global(t-interval0) #后面
                #     speed0 = (data0-data_behind)/interval0 #速度的上限应该是2，速度超过2是不可想象的
                #     num=3
                #     # speedList=[]
                #     loss_smooth = 0
                #     for i0 in range(num):
                #         i = i0+1
                #         data_front = self.g_global(t+(i+1)*interval0)#[:,:2] #后面
                #         speed1 = (data_front-data0)/interval0
                #         loss_smooth = loss_smooth + ( speed1 - speed0 )**2

                #     eps = 10**-10
                #     loss_smooth = ((
                #         acceleration/(torch.sigmoid(speed+2).clone().detach()+eps)
                #         )**2).mean()

            ############################################################
            # xy_ = xyt[:, :-1] + h_global #+ h_local
            if self.useMatrix: #True
                c =h_global
                u = xyt[:, 0]
                v = xyt[:, 1]
                if c.shape[1]==6: # 6个自由度的全局变换
                    new_u = c[:,0] * u + c[:,1] * u + c[:,2]
                    new_v = c[:,3] * v + c[:,4] * v + c[:,5]
                else: # 4个自由度的全局变换 (后面可能添加5个自由度的模式)
                    # 提取参数 (忽略可能的第五个参数)
                    tx = c[:, 0]  # X轴位移 #范围大致是[-1,+1],不严格
                    ty = c[:, 1]  # Y轴位移
                    #实际上没有使用旋转自由度和缩放自由度
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

                    h_global = torch.cat([
                        c[:, [0]],
                        c[:, [1]]
                    ], dim=-1)
                # 组合成新坐标张量
                new_uv = torch.stack([new_u, new_v], dim=1)
                xy_ = new_uv
            else: #不使用变换矩阵 #两个自由度的模式
                xy_ = xyt[:, :-1] + h_global
            ############################################################
            if not openLocalDeform:#stage==0:#纹理学习，不分析局部位移  #不使用局部位移
                h_local = torch.tensor(0.0)# h_local = self.g_local(xyt) if self.useLocal else torch.tensor(0.0)
            else:#只分析局部位移，不学习整体运动和纹理
                h_local = self.g_local(xyt)
                if False:
                 if self.useSmooth:#局部形变也要平滑
                    loss_smooth = loss_smooth + jacobian(h_local, xyt).abs().mean()# if self.useSmooth else 0
                h_local = 2 * torch.sigmoid(h_local) - 1
                h_local = h_local * self.deformationSize
                xy_ = xy_ + h_local
        # color = torch.sigmoid(self.f_2D(xy_))
        # if False:
        color = torch.sigmoid(self.f_2D(xy_))
        if self.useTex2D:
            color = color + torch.sigmoid(self.f_2D(xy_))
        # color = torch.sigmoid(self.f_2D(xy_))
        return color,{
            "xy_":xy_,
            "h_local":h_local, #训练的时候stage为0，所以输出的是一个常数0
            "h_global":h_global, # 不明白为啥之前的全局位移要让它不可学习 #之前的版本是下面一行
            #"h_global":h_global.clone().detach()
            "loss_smooth":loss_smooth
        }

#####################################################################

# outpath = './nir/data/removeRigid_27'
# EpochNum = 6000 #5000 #3000

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
    def __init__(self, path,hidden_features=128,useSmooth=False,openLocalDeform=False,weight_smooth=1,stillness=False,stillnessFristLayer=True,
                 NUM_soft=0,NUM_rigid=2,NUM_fluid=0,
                 useMask=False,openReconLoss_rigid=False,lossType=1,
                 maskPath=None,
                 interval=1.0,#计算平滑损失时的偏导数计算中的步长
                #  useMatrix=True,#整体运动最好不直接用矩阵，矩阵优化的分析较为复杂
                configRigid=None,
                configRigids={},
                configSofts={},
                configFluids={}
                 ):
        super().__init__()
        v = VideoFitting(path,useMask=useMask,maskPath=maskPath)
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
        self.useSmooth=useSmooth#一个布尔参数，用来决定本次测试是否使用基于雅可比矩阵的平滑损失函数
        self.openLocalDeform=openLocalDeform
        self.weight_smooth=weight_smooth
        self.stillness=stillness#是否静止不动
        self.openReconLoss_rigid=openReconLoss_rigid#是否使用刚体自身的重构损失(只优化刚体层、不影响软体层,无MASK)
        self.lossType=lossType
        self.loss_recon_all_type=configRigids["loss_recon_all_type"]
        
            

        #######################################################################

        # 一、软体
        self.NUM_soft = NUM_soft#0#2 # 软体层的数据
        self.f_soft_list = []
        for i in range(self.NUM_soft):
            # self.f_soft_list.append(Layer(useGlobal=False,hidden_features=hidden_features))
            self.f_soft_list.append(Layer(
                useGlobal=False, 
                hidden_features=512,
                config=configSofts["layer"]
                ))
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
            self.f_rigid_list.append(Layer_rigid(
                useLocal=True, 
                deformationSize=3*(2/(self.v.video.size()[2] - 1)),
                # hidden_features=128, 
                useMatrix=True,
                useSmooth=self.useSmooth,
                stillness=stillness,
                v=self.v,
                interval=interval,
                config=configRigid,
            ))
        if self.NUM_rigid>0:
            self.f_rigid_list[0].stillness=stillnessFristLayer
            # useGlobal=True,useLocal=True,useMatrix=True,useDeformation=False
        # 三、流体(流体层至多1层)
        self.NUM_fluid = NUM_fluid
        # if NUM_fluid>0:
        #     self.f2 = Layer_video(
        #         configFluids
        #     )
        self.f_fluid_list=[]
        for i in range(self.NUM_fluid):
            self.f_fluid_list.append(Layer_video(
                configFluids
            ))
        if NUM_fluid>0:
            self.f2 = self.f_fluid_list[0]


        self.parameters=[
            # self.f2.parameters()
        ] #+ self.f_soft.parameters #+ self.f_rigid.parameters
        for i in range(self.NUM_rigid): #self.parameters = [self.f_2D.parameters()]
            self.parameters = self.parameters + self.f_rigid_list[i].parameters
        for i in range(self.NUM_soft):
            self.parameters = self.parameters + self.f_soft_list[i].parameters
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

    def forward(self,xyt,stage): # soft, rigid, fluid
        # 1.刚体
        o_rigid_all = 1
        o_rigid_list = []
        h_local_list = []
        h_global_list=[]
        h_global = torch.tensor(0.0)
        loss_smooth = torch.tensor(0.0)
        for i in range(self.NUM_rigid):
            # step = 2/(self.v.video.size()[0] - 1)
            o_rigid0, p_rigid0 = self.f_rigid_list[i](xyt,stage)#因为这的stage为0，所以并没有真正使用软体形变
            o_rigid_all = o_rigid_all*o_rigid0
            o_rigid_list.append(o_rigid0)
            h_local_list.append(p_rigid0["h_local"])
            h_global_list.append(p_rigid0["h_global"])
            h_global = p_rigid0["h_global"]
            loss_smooth = loss_smooth + p_rigid0["loss_smooth"]
        if self.NUM_rigid>0:
            h_global = torch.cat([h_global, torch.zeros(h_global.shape[0], 1).cuda()], dim=1)

        # 2.软体
        o_soft_all = 1
        o_soft_list = []
        for i in range(self.NUM_soft):
            o_soft0, _ = self.f_soft_list[i](xyt+h_global) #软体运动基于刚体运动
            o_soft_all = o_soft_all * o_soft0
            o_soft_list.append(o_soft0)

        # 3.流体
        o_fluid = 1
        o_fluid_all = 1
        o_fluid_list = []
        for i in range(self.NUM_fluid):
            o_fluid0 = self.f_fluid_list[i](xyt)
            o_fluid_all = o_fluid_all * o_fluid0
            o_fluid_list.append(o_fluid0)
        if self.NUM_fluid>0 and len(o_fluid_list)==1:
            o_fluid = o_fluid_list[0]#self.f2(xyt) 
            if torch.isnan(o_fluid.mean()):
                print("o_fluid is nan",o_fluid.mean())
                total_norm = torch.nn.utils.clip_grad_norm_(self.f2.parameters(), float('inf'))
                if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 1e4:
                    print('>>> 出现梯度爆炸！', total_norm)
                else:
                    print('>>> 没有梯度爆炸！', total_norm)

        o = o_rigid_all * o_soft_all * o_fluid_all 
        # o = o_rigid_all * o_soft_all # o = o_rigid_all * o_soft_all * (1.0 - o_fluid)
        return o , {
            "r": o_rigid_list,
            "s": o_soft_list,
            "f": o_fluid_list,
            "f_old": o_fluid
        } ,{
            "h_local_list":h_local_list,#这里的原始版本是"h_local_list":h_local_list,现在修改后要注意兼容性问题
            "h_global_list":h_global_list,
            "o_rigid_all":o_rigid_all,
            "o_soft_all":o_soft_all,
            "o_fluid_all":o_fluid_all,
            "loss_smooth":loss_smooth,
        }#输出三样东西：重构结果、分层结果、相关参数

    def loss(self, xyt, step, start, end,openLocalDeform,lossParam):
        if self.lossType==1:
            return self.loss1(xyt, step, start, end,openLocalDeform)
        elif self.lossType==2:
            return self.loss2(xyt, step, start, end,openLocalDeform,lossParam)

    def loss1(self, xyt, step, start, end,openLocalDeform):
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

    def loss2_old(self, xyt, step, start, end,openLocalDeform):
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

    def loss2(self, xyt, step, start, end,openLocalDeform,lossParam={"rm":"S","ra":"R"}):
        o, layers, p = self.forward(xyt,openLocalDeform)#纹理学习

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
        # 一、软体有遮挡重构损失 loss=M*(S*R-O)
        loss_recon_mask = torch.tensor(0.0)
        if not lossParam["rm"] is None:
            rm_in=getData(lossParam["rm"])
            loss_recon_mask = torch.log(
                (self.ground_truth[start:end].abs()+eps)/((rm_in).abs()+eps)
            ).abs()
            loss_recon_mask = loss_recon_mask*self.mask[start:end]
            loss_recon_mask = loss_recon_mask.sum()/(self.mask[start:end].sum()+1e-8)
        
        # 二、刚体无遮挡重构损失 loss=(R-O)
        loss_recon_all = torch.tensor(0.0)
        
        if not lossParam["ra"] is None:
            ra_in=getData(lossParam["ra"])
            if True:#用于输出拟合程度
             if not self.loss_recon_all_type=="MSE" and not step % 100:
                loss_recon_all0 = ( self.ground_truth[start:end] - ra_in )**2
                loss_recon_all0 = loss_recon_all0.mean()#这个对象在训练后期变为了None
                # print("loss_recon_all0",loss_recon_all0)
            if torch.isnan(ra_in.mean()):
                print("ra_in is nan.",ra_in.mean()) #ra_in is None. tensor(nan, device='cuda:0', grad_fn=<MeanBackward0>)
                exit(0)
            if self.loss_recon_all_type=="MSE":
                loss_recon_all = ( self.ground_truth[start:end] - ra_in )**2
                loss_recon_all = loss_recon_all.mean()#这个对象在训练后期变为了None
            elif self.loss_recon_all_type=="atten_d":#类似最大值的思想
                temperature = 0.1**2 #1.0 # 温度参数调节注意力集中程度
                # temperature越小，越关注最大误差（类似max）
                # temperature越大，越接近平均（类似mean）
                errors = ( self.ground_truth[start:end] - ra_in )**2
                attention_weights = torch.softmax(errors.clone().detach()/ temperature, dim=0) 
                loss_recon_all = (attention_weights * errors).sum()
            elif self.loss_recon_all_type=="atten":#类似最大值的思想(估计会不稳定)
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
        loss = loss_recon_mask + loss_recon_all + loss_smooth*self.weight_smooth

        self.layers = layers
        if not step % 100:
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
            else:
                if self.loss_recon_all_type=="MSE":
                    print("Step [%04d]: loss=%0.8f, recon_mask=%0.8f, recon_all=%0.8f" % (
                        step, loss, loss_recon_mask, loss_recon_all))
                else:
                    print("Step [%04d]: loss=%0.8f, recon_mask=%0.8f, recon_all=%0.8f, recon_all0=%0.8f" % (
                        step, loss, loss_recon_mask, loss_recon_all,loss_recon_all0))

        return loss

    def train(self,total_steps,paramLoss):
        if self.NUM_fluid==1:
            gradientMonitor = GradientMonitor(self.f2)
        model_input = self.model_input

        # print("1;self.parameters",self.parameters)
        optim = torch.optim.Adam(lr=1e-4, params = itertools.chain.from_iterable(self.parameters))
        # print("2;self.parameters",self.parameters)
        # exit(0)
        # optim = torch.optim.Adam(lr=1e-4, params=chain(self.parameters))

        batch_size = (self.v.H * self.v.W) // 8
        for step in range(total_steps): #生成纹理、整体运动
            start = (step * batch_size) % len(model_input)
            end = min(start + batch_size, len(model_input))

            xyt = model_input[start:end].requires_grad_()
            loss = self.loss(xyt, step,start,end,self.openLocalDeform,paramLoss)#离谱，stage竟然为0

            optim.zero_grad()
            loss.backward()
            optim.step()
            if not step % 100 and step>0 and self.NUM_fluid==1:
                gradientMonitor.analyze(step)

            if False:
                if loss<0.1**5: #重构损失足够小的时候退出,达到这个标注后仍然有问题
                    print("step",step,";loss",loss)
                    break
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
                frame_output, layers, p = self.forward(coords,stage)

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
                        l[i][j] = l[i][j].view(H, W, 1)
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
            else:
                p["o_soft_all"]=None
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


