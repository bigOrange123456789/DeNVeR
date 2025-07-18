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

from nir.myLib.Layer import Layer
from nir.myLib.Decouple import Decouple
from nir.myLib.mySave import check,save2img,save0

videoId="CVAI-2855LAO26_CRA31"
paramPath = "../DeNVeR_in/models_config/freecos_Seg.pt"
pathIn = 'nir/data/in2'
maskPath="./nir/data/mask/filter"
pathLable = "nir/data/in2_gt"
pathMaskOut = "./nir/data/mask"
print("01:解耦刚体、并尽量高质量重构出视频", "(1)刚体的信息量不足、(2)在计算MASK的时候与recon-non不同、rigid-non不用除2。")
print("02:不进行血管监督去除", "可视化结果由进步、但指标没有上涨。")
# 如果使用MASK监督去除，某个位置刚体被误认为血管，该位置一直不移动、刚体解构就找不到该区域。
# 因此刚体解耦绝不能去除血管监督。
print("03:提高刚体的2D纹理表达能力, 刚体的隐含层特征维度由128变为512", "刚体层没有解耦出来任何信息。")
print("04:降低刚体的局部变形能力deformationSize由8修改为1.5、所有刚体层计算损失的均值而不是求和", "刚体层依旧没有解耦出来任何信息")
print("05:将刚体层的损失缩小100->10", "刚体层依旧没有解耦出任何信息")
print("06:取消刚体层的局部形变", "软体层没有解耦出任何信息")
# 对于刚体层来说、如果没有整体变化、那也不应该有局部变化 #刚体的抖动范围应该减半
# 也可能是初始加权导致的刚体信息较少
print("07:增加流体层的约束", "")

outpath = './nir/data/removeRigid_07'
EpochNum =6000 #5000 #3000

#########################################################################

from nir.myLib.VideoFitting import VideoFitting

weight_target = [0.25,100,400]#RSF #信息量比例的目标
weight_init = [10, 1, 10] #RSF

class Decouple2(nn.Module):
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
        self.NUM_soft = 1 # 软体层的数据
        self.f_soft_list = []
        for i in range(self.NUM_soft):
            self.f_soft_list.append(Layer(useGlobal=False,hidden_features=hidden_features))
        # 二、刚体
        self.NUM_figid = 3 # 刚体层的数据
        self.f_rigid_list=[]
        for i in range(self.NUM_figid):
            # self.f_rigid_list.append(Layer(useDeformation=True,hidden_features=512,deformationSize=1.5))
            self.f_rigid_list.append(Layer(useLocal=False, hidden_features=512))
            # useGlobal=True,useLocal=True,useMatrix=True,useDeformation=False
        # 三、流体
        self.f2 = Siren(in_features=3, out_features=1, hidden_features=hidden_features,
                   hidden_layers=4, outermost_linear=True)
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
            o_rigid0, p_rigid0 = self.f_rigid_list[i](xyt)
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
        o, layers, p = self.forward(xyt)
        o_fluid = layers["f"]

        def fluidInf(x):
            k = 10 ** 10  # 一个无限大的超参数
            x = torch.log(1 + k * x) / math.log(1 + k) #将非零数据都变为1
            mask_binary = torch.log(1 + k * x) / math.log(1 + k) #用同一个函数处理两次、结果更像二值图
            return mask_binary.abs().mean()
        with torch.no_grad():
            def fun0(x):
                eps = math.e ** -101
                x = x.detach().clone()
                x = x.clamp(min=eps)
                return (-torch.log(x))**4


            # 下述衡量信息量的三个指标都是0-1
            # i_r = p["o_rigid_all"].detach().clone().abs().mean()
            i_r0 = torch.var(p["o_rigid_all"].detach().clone().abs()) # 刚体层方差信息量
            i_s0 = 1-p["o_soft_all"].detach().clone().abs().mean()    # 软体层暗度信息量
            # i_f = o_fluid.detach().clone().abs().mean()              # 流体层亮度信息量
            i_f0 = fluidInf(o_fluid.detach().clone())  # 流体层亮度信息量
            # wr = (1-i_r)*fun0((1-i_s)*(1-i_f)) #本层的信息量越少则约束越小
            # ws = (1-i_s)*fun0((1-i_r)*(1-i_f))
            # wf = (1-i_f)*fun0((1-i_r)*(1-i_s))
            temp=[
                i_r0 / weight_target[0],
                i_s0 / weight_target[1],
                i_f0 / weight_target[2]
            ]
            i_r = temp[0] / sum(temp)
            i_s = temp[1] / sum(temp)
            i_f = temp[2] / sum(temp)

            wr = i_r * fun0(i_s * i_f)  # 1.本层的信息量越少则本层约束越小
            ws = i_s * fun0(i_r * i_f)  # 2.它层的信息量越少则本层约束越大
            wf = i_f * fun0(i_r * i_s)

        loss_recon = ((o - self.ground_truth[start:end]) ** 2) * (10 ** 5)
        if False: #self.v.useMask:
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
            loss_rigid = loss_rigid + p["h_local"][i].abs().mean()  # 刚体的局部位移
        loss_rigid = loss_rigid * weight_init[0]/self.NUM_figid #(10**2)
        # 二、软体
        loss_soft = (1. - p["o_soft_all"]).abs().mean() * weight_init[1] #(0.1) # 软体的暗度
        # 三、流体 # 减少流体的信息量
        loss_fluid = fluidInf(o_fluid.abs()) * weight_init[2] #1 # 减少流体的信息量


        loss = loss_recon + loss_soft + loss_fluid + loss_rigid

        self.layers=layers
        if not step % 200:
            print("Step [%04d]: loss=%0.8f, recon=%0.8f, loss_soft=%0.8f, loss_fluid=%0.8f, loss_rigid=%0.4f" % (
                step, loss, loss_recon, loss_soft , loss_fluid , loss_rigid))
            print("i_r0:", i_r0.item(),"; i_s0:", i_s0.item(),"; i_f0:", i_f0.item(),"\ntemp",
                  [temp[0].item(),temp[1].item(),temp[2].item()])
            print("i_r",i_r.item(), "\twr", wr.item()) # i_f为0
            print("i_s",i_s.item(), "\tws", ws.item())
            print("i_f",i_f.item(), "\twf", wf.item()) #wr、ws为无穷大
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

            layers = {
                "r": p01(layers_frames["r"]),
                "s": p01(layers_frames["s"]),
                "f": torch.stack(layers_frames["f"], dim=0)
            }
            p = {
                "o_rigid_all": torch.stack(p_frames["o_rigid_all"], dim=0),
                "o_soft_all": torch.stack(p_frames["o_soft_all"], dim=0)
            }
            return video_pre, layers, p


#########################################################################

# 解耦出血管层
if True:
    myMain=Decouple2(pathIn,maskPath=maskPath,hidden_features=256)
    myMain.train(EpochNum) #EpochNum =5000

def save1(o_scene, tag):
    o_scene = o_scene.cpu().detach().numpy()
    o_scene = (o_scene * 255).astype(np.uint8)
    save2img(o_scene[:, :, :, 0], os.path.join(outpath, tag))


#输出解耦效果
if True:# False:
 with torch.no_grad(): #
    orig = myMain.v.video.clone()
    orig = orig.permute(0, 2, 3, 1).detach().numpy()
    orig = (orig * 255).astype(np.uint8)
    save2img(orig[:, :, :, 0], os.path.join(outpath, 'orig'))

    ###########################################################################################
    orig = myMain.v.video.clone()
    N, C, H, W = orig.size()  # 帧数、通道数、高度、宽度
    orig = orig.permute(0, 2, 3, 1).detach()#.numpy()

    video_pre, layers, p =myMain.getVideo()

    save1(video_pre, "recon")
    save1(p["o_rigid_all"], "rigid")
    save1(p["o_soft_all"], "soft")
    save1(orig.cuda()/(p["o_rigid_all"].abs()+10**-10), "rigid_non")
    save1(0.5*orig.cuda()/(p["o_rigid_all"].abs()+10**-10), "rigid_non2")
    save1(layers["f"], "fluid")

    mainFreeCOS(paramPath, os.path.join(outpath, "rigid_non"), os.path.join(outpath, "mask_nr1"))
    check(os.path.join(outpath, "mask_nr1"), videoId, "nir.1.rigid_non1")
    mainFreeCOS(paramPath,os.path.join(outpath, "rigid_non2"),os.path.join(outpath, "mask_nr2"))
    check(os.path.join(outpath, "mask_nr2"),videoId,"nir.1.rigid_non2")


'''
    python -m nir.removeRigid
'''
