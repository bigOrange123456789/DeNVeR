import os
import numpy as np
from itertools import chain

import torch
from torch.utils.data import DataLoader

from model import Siren
from util import get_mgrid, jacobian, VideoFitting

from PIL import Image
outpath = './data/myReflection2_global34'
print("01:多个刚体(1个)-效果应该和之前一致","成功：效果一致")
print("02:多个软体(1个)-效果应该和01一致","成功：效果一致")
print("03:测试多个刚体(2个)的效果","失败：忘记将参数修改为2")
print("04:测试基于sel软体太多、流体f.log的损失函数","失败：局部位移不需要self.log")
print("05:测试动态损失函数权重","效果没有显著变化")
print("06:不对血管区域进行重建监督","效果极差")
print("07:加大刚体约束【0.2->0.6】","软体信息量增大了一点")
print("08:加大刚体约束【0.6->1.2】","软体信息量增大了一点")
print("09:缩小软体约束【0.05->0.03】","软体层信息显著增加，但流体层意义不明")
print("10:使用2个刚体层","对结果没啥影响")
print("11:使用10个刚体层","失败:设置成了软体层","新增软体层会增加软体层信息")
print("12:对流体层使用二值化约束","约束后流体层消失了")
print("13:使用10个刚体层【2->10】","捕获了一些血管信息")
print("14:本层的信息量越少则约束越小","没有效果")
print("15:尝试增加流体层的信息量","没有效果")
print("16:信息量越少其他层的约束越大","发现一个异常:刚体的损失函数为0")
print("17:用方差表示刚体的信息量","效果良好")
print("18:(1)用非0像素数估计流体信息量(2)去除了流体二值约束","流体效果非常好,但是软体消失了")
print("19:改正相对信息权重的计算错误","没啥效果","流体信息很多、损失确为0")
print("20:增大对流体的损失约束【0.01->1.0】","没有软体信息、流体损失过小")
print("21:增大对流体的损失约束【1.0->500.0】","流体的信息没有转移到软体上、而是转移到了刚体上")
print("22:流体约束迭代两次二值化函数","流体效果更好一些，仍然没有软体")
print("23:增大对流体的损失约束【500->10**5】","没啥变化")
print("24:减小软体层约束【0.03->10**-5】 ","软体增多、没有流体")
print("25:信息比例RSF【0.1,1,400】 ")
print("26:去除独立的权重【刚体:1.2->1;软体:0.1**5->0.1;流体:10**5->10】","重构损失相对过小")
print("27:重构损失的权重【10->10**5】","刚体过多、没有软体")
print("28:RSF【0.1,1,400】->【0.1**2,10,400】","已经具有良好的效果")
print("29:刚体100 流体0.01","刚体的约束已经足够了")
print("30:放松对刚体的约束,集中精力区分刚体和软体【0.1**2,10,400】->【0.5,10,400】","流体层信息太少")
print("31:测试3个软体层的效果","软体太多、流体太少、无法自动解决")
print("32:软体0.1、流体1","流体太多、软体太少、出现误判")
print("33:修改评判标准RSF[0.5,10->100,400->200]","感觉基本上这已经到极限了")
print("34:测试整段视频的效果","")
print("未实现内容: 03_01:刚体层能够描述放缩和旋转")
RSF_Weight_List=[0.5,100,200]#RSF
EpochNum =5000
def save2img(imgs,path):
    if not os.path.exists(path):os.makedirs(path)
    for i in range(imgs.shape[0]):
        image_array = imgs[i]
        image = Image.fromarray(image_array, mode='L')
        image.save(os.path.join(path, str(i).zfill(5) + '.png'))

class Layer():
    def __init__(self):
        self.g_global = Siren(in_features=1, out_features=2, hidden_features=16,
                              hidden_layers=2, outermost_linear=True)
        self.g_global.cuda()
        self.g_local = Siren(in_features=3, out_features=2, hidden_features=128,
                       hidden_layers=4, outermost_linear=True)
        self.g_local.cuda()
        self.f_2D = Siren(in_features=2, out_features=1, hidden_features=128,
                        hidden_layers=4, outermost_linear=True)
        self.f_2D.cuda()
        self.parameters=[
            self.g_global.parameters(),
            self.g_local.parameters(),
            self.f_2D.parameters()
        ]
    def forward0(self,xyt):
        h_global=self.g_global(xyt[:, [-1]])
        h_local =self.g_local(xyt)
        xy_ = xyt[:, :-1]+h_global+h_local
        return torch.sigmoid(self.f_2D(xy_)),{
            "xy_":xy_,
            "h_global":h_global,
            "h_local":h_local
        }
import itertools
import math
class Main():
    def log(self,x): #x中的全部元素均不能为负数
        import math
        e = math.e # 获取自然数 e 的值
        eps = e ** -101
        # return x
        # eps = 1e-10 #torch.finfo(torch.float64).eps
        return -1.*torch.log(x+eps) # return -1.*torch.log(x.abs()+eps)
    def __init__(self, path):
        v = VideoFitting(path)
        videoloader = DataLoader(v, batch_size=1, pin_memory=True, num_workers=0)
        model_input, ground_truth = next(iter(videoloader))
        model_input, ground_truth = model_input[0].cuda(), ground_truth[0].cuda()
        ground_truth = ground_truth[:, 0:1]  # 将RGB图像转换为灰度图
        # print(ground_truth.shape)
        # print("ground_truth",type(ground_truth))
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

        #######################################################################

        # 一、软体
        # self.f_soft=Layer()
        self.NUM_soft = 3  # 软体层的数据
        self.f_soft_list = []
        for i in range(self.NUM_soft):
            self.f_soft_list.append(Layer())
        # 二、刚体
        self.NUM_figid=5 #刚体层的数据
        self.f_rigid_list=[]
        for i in range(self.NUM_figid):
            self.f_rigid_list.append(Layer())
        # 三、流体
        self.f2 = Siren(in_features=3, out_features=1, hidden_features=128,
                   hidden_layers=4, outermost_linear=True)
        self.f2.cuda()

        self.parameters=[
            self.f2.parameters()
        ] #+ self.f_soft.parameters #+ self.f_rigid.parameters
        for i in range(self.NUM_figid):
            self.parameters = self.parameters + self.f_rigid_list[i].parameters
        for i in range(self.NUM_soft):
            self.parameters = self.parameters + self.f_soft_list[i].parameters

    def synthesis(self,xyt): # soft, rigid, fluid
        # 1.刚体
        o_rigid_all = 1
        o_rigid_list = []
        h_local_list = []
        for i in range(self.NUM_figid):
            o_rigid0, p_rigid0 = self.f_rigid_list[i].forward0(xyt)
            o_rigid_all = o_rigid_all*o_rigid0
            o_rigid_list.append(o_rigid0)
            h_local_list.append(p_rigid0["h_local"])

        # 2.软体
        o_soft_all = 1
        o_soft_list = []
        for i in range(self.NUM_soft):
            o_soft0, _ = self.f_soft_list[i].forward0(xyt)
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

    def loss(self, xyt, ground_truth, step, start, end):
        o, layers, p = self.synthesis(xyt)
        o_soft_list  = layers["s"]
        o_fluid = layers["f"]

        def fluidInf1(x): #获取MASK中白色区域的像素数量占比
            EXPONENT = 7
            mask_binary = torch.pow(x - 0.5, EXPONENT) / (0.5 ** EXPONENT) + 0.5
            return mask_binary.abs().mean()
        def fluidInf2(x): #获取MASK中非0区域的像素数量占比
            EXPONENT = 7
            mask_binary = torch.pow(x - 0.5, EXPONENT) / (0.5 ** EXPONENT) + 0.5#将大于0.5的数据变为1
            return mask_binary.abs().mean()
        def fluidInf(x):
            k = 10 ** 10  # 一个无限大的超参数
            # k = 2 ** 64 - 2  # 一个超参数
            x = torch.log(1 + k * x) / math.log(1 + k) #将非零数据都变为1
            mask_binary = torch.log(1 + k * x) / math.log(1 + k) #用同一个函数处理两次、结果更像二值图
            return mask_binary.abs().mean()
        with torch.no_grad():
            def fun0(x):
                eps = math.e ** -101
                x = x.detach().clone()
                x = x.clamp(min=eps)
                # x = x.clamp(max=1 - eps)
                return (-torch.log(x))**4

            def fun1(x):
                eps = 1e-10#torch.tensor([1e-10])
                x = x.detach().clone()
                x=x.clamp(min=eps)
                x=x.clamp(max=1-eps)
                inner_expr = (math.pi/2) * (x+1)
                return -1.0 * torch.tan(inner_expr)

                # return -np.tan( (math.pi/2) * (x +1) )
            # print("0",fun0(torch.tensor([0.0])))
            # print("1", fun0(torch.tensor([1.0])))
            # exit(0)

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
                i_r0 / RSF_Weight_List[0],
                i_s0 / RSF_Weight_List[1],
                i_f0 / RSF_Weight_List[2]
            ]
            i_r = temp[0] / sum(temp)
            i_s = temp[1] / sum(temp)
            i_f = temp[2] / sum(temp)

            wr = i_r * fun0(i_s * i_f)  # 1.本层的信息量越少则本层约束越小
            ws = i_s * fun0(i_r * i_f)  # 2.它层的信息量越少则本层约束越大
            wf = i_f * fun0(i_r * i_s)
            k001 = 1#20 #为了强制三个图层的信息量相近，这里要放大他们的差距
            wr = wr ** k001
            ws = ws ** k001
            wf = wf ** k001

        loss_recon = ((o - ground_truth[start:end]) ** 2).mean() * (10**5)
        if False:# 不对血管区域进行重建监督
            loss_recon = (((1.0-o_fluid)*(o - ground_truth[start:end])) ** 2).mean()*10.
        # if False:
        #     loss_recon = ((self.log(o) - ground_truth[start:end]) ** 2).mean() * 10.
        # 一、刚体
        loss_rigid = 0 #刚体的局部位移尽量小
        for i in range(self.NUM_figid):
            # loss_rigid = loss_rigid+1.20 * p["h_local"][i].abs().mean() # 减少刚体的信息量
            loss_rigid = loss_rigid + p["h_local"][i].abs().mean()  # 减少刚体的信息量
        loss_rigid = loss_rigid * (10**2)
        # 二、软体
        # loss_soft = (10**-5) * (1. - p["o_soft_all"]).abs().mean() # 减少软体的信息量
        loss_soft = (0.1) * (1. - p["o_soft_all"]).abs().mean()  # 减少软体的信息量
        # 三、流体 # 减少流体的信息量
        # loss_fluid = 0.02 * o_fluid.abs().mean() # 减少流体的信息量
        # loss_fluid = loss_fluid + 0.01 * (o_fluid*(1-o_fluid)).abs().mean() # 二值化约束
        # loss_fluid = (10**5) * fluidInf(o_fluid.abs()) # 减少流体的信息量
        loss_fluid = 1 * fluidInf(o_fluid.abs())  # 减少流体的信息量
        # 刚体:1.2->1;软体:0.1**5->0.1;流体:10**5->10

        loss_rigid = loss_rigid * wr
        loss_soft  = loss_soft  * ws
        loss_fluid = loss_fluid * wf

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
        ground_truth = self.ground_truth

        optim = torch.optim.Adam(lr=1e-4, params = itertools.chain.from_iterable(self.parameters))
        # optim = torch.optim.Adam(lr=1e-4, params=chain(self.parameters))

        batch_size = (self.v.H * self.v.W) // 8
        for step in range(total_steps):
            start = (step * batch_size) % len(model_input)
            end = min(start + batch_size, len(model_input))

            xyt = model_input[start:end].requires_grad_()
            loss = self.loss(xyt,ground_truth, step,start,end)

            optim.zero_grad()
            loss.backward()
            optim.step()

myMain=Main('data/in2')
myMain.train(EpochNum) #EpochNum =5000
orig = myMain.v.video
# g, f1, f2, orig = train_reflection('./data/in', 3000)

def save0(o_scene,tag):
    N, _, H, W = orig.size()
    o_scene = o_scene.view(H, W, N, 1).permute(2, 0, 1, 3).cpu().detach().numpy()
    o_scene = (o_scene * 255).astype(np.uint8)
    save2img(o_scene[:, :, :, 0], os.path.join(outpath, tag))
with torch.no_grad():
    N, _, H, W = orig.size()
    xyt = get_mgrid([H, W, N]).cuda()
    o, layers, p=myMain.synthesis(xyt) # h = g(xyt)
    save0(o, "recon")
    save0(p["o_rigid_all"], "rigid")
    save0(p["o_soft_all"], "soft")

    for i in range(len(layers["r"])):
        save0(layers["r"][i], "rigid"+str(i))
    for i in range(len(layers["s"])):
        save0(layers["s"][i], "soft"+str(i))
    # save0(layers[1], "soft")
    save0(layers["f"], "fluid")

    orig = orig.permute(0, 2, 3, 1).detach().numpy()
    orig = (orig * 255).astype(np.uint8)
    save2img(orig[:, :, :, 0], os.path.join(outpath, 'orig'))

