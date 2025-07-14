import os
import numpy as np
from itertools import chain
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import re

from nir.model import Siren
from nir.util import get_mgrid, jacobian#, VideoFitting
from nir.util import Dataset,ToTensor

class VideoFitting(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.numChannel=1 #不为3
        self.useMask=True

        self.path = path
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform

        self.video = self.get_video_tensor(path)
        if self.useMask:
            self.mask_ = 1-self.get_video_tensor("./nir/data/mask/filter")
        self.num_frames, _, self.H, self.W = self.video.size()

        self.pixels = self.video.permute(2, 3, 0, 1).contiguous().view(-1, self.numChannel)
        if self.useMask:
            self.mask = self.mask_.permute(2, 3, 0, 1).contiguous().view(-1, self.numChannel)
        self.coords = get_mgrid([self.H, self.W, self.num_frames])

        shuffle = torch.randperm(len(self.pixels))
        self.pixels = self.pixels[shuffle]
        self.coords = self.coords[shuffle]
        if self.useMask:
            self.mask = self.mask[shuffle]
        else: self.mask =torch.tensor([[1.0]])

    def get_video_tensor(self,path):
        frames = sorted(os.listdir(path))
        video = []
        for i in range(len(frames)):
            img = Image.open(os.path.join(path, frames[i]))
            img = self.transform(img)
            video.append(img)
        return torch.stack(video, 0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError
        return self.coords, self.pixels, self.mask
videoId="CVAI-2855LAO26_CRA31"
paramPath = "../DeNVeR_in/models_config/freecos_Seg.pt"
pathIn = 'nir/data/in2'
pathLable = "nir/data/in2_gt"
pathMaskOut = "./nir/data/mask"
from free_cos.main import mainFreeCOS
if False:
    mainFreeCOS(paramPath,pathIn,pathMaskOut)


from eval.eval import Evaluate
evaluate = Evaluate()
def check(predictPath,video_id,tag):
    def natural_sort_key(s):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r'(\d+)', s)]

    # 统一转成单通道，再转 Tensor
    transform = T.Compose([
        T.Grayscale(num_output_channels=1),  # 确保是单通道
        T.ToTensor()  # 转 Tensor，范围 [0,1]
    ])
    y_list=[]
    for name in sorted(os.listdir(os.path.join(predictPath,"filter")),key=natural_sort_key):
        print("name",name)
        img0 = Image.open(os.path.join(predictPath,"filter",name))  # PIL Image
        t0 = transform(img0)  # (1, H, W)
        # t0[t0>=0.5]=1
        # t0[t0<0.5]=0
        y_list.append(t0)
    evaluate.analysis(tag, video_id, 255*torch.cat(y_list, dim=0), -1)

check(pathMaskOut,videoId,"nir.0.origin")
# exit(0)
if True:
    outpath = './nir/data/myReflection3_10'
    mainFreeCOS(paramPath,os.path.join(outpath, "recon_non2"),os.path.join(outpath, "mask2"))
    check(os.path.join(outpath, "mask2"),videoId,"nir.1.recon_non2")
    exit(0)

print("03_01:刚体层能够描述放缩和旋转","刚体的解耦效果模糊并且包含血管(单显示刚体信息较多)")
print("03_02:刚体层无局部、软体层无全局","刚体层没有移动","有一个错误、刚体关闭了全局")
print("03_03:放弃动态调整这三种数据的信息量、刚体仅全局","刚体层空无一物，软体层信息过多")
print("03_04:增大软体层的约束、0.1->1","刚体只有放缩、没有移动。软体层信息过多")
print("03_05:(1)钢体层只能进行长宽同比例的放缩(2)缩小重构损失10**6->10**5","刚体信息少、但是正确度高")
print("03_06:不对血管区域进行重构约束","软体区域只有噪声、刚体信息不正确") # 不计算血管MASK区域内的重构损失 #重构了血管而不是背景
print("03_07:刚体使用了局部形变","") #重构了血管而不是背景
print("03_08:(1)用mask进行相对加权、而不是直接抹去一部分重构损失；(2)改正了重构监督的错误[改为只监督背景]","重建的效果不错、但是刚体的效果不好")
print("03_09:减少层数软3刚5->软1刚3","逆推效果不好、刚体效果不好")
print("03_10:关闭刚体解耦中的旋转变换和放缩变换","")
print("03_11:测试解耦后的血管分割准确性","")
# 局部解耦
outpath = './nir/data/myReflection3_11'
weight_target = [0.25,100,400]#RSF #信息量比例的目标
weight_init = [10**2, 1, 1] #RSF

# 初始约束权重
EpochNum =6 #5000 #3000
import torch.nn as nn
class Layer(nn.Module):
    def __init__(self,useGlobal=True,useLocal=True,useMatrix=True,useDeformation=False,deformationSize=8):
        super().__init__()
        self.useGlobal=useGlobal
        self.useLocal=useLocal
        self.useMatrix=useMatrix
        self.useDeformation=useDeformation
        self.deformationSize=deformationSize
        self.f_2D = Siren(in_features=2, out_features=1,
                          hidden_features=128, hidden_layers=4,
                          outermost_linear=True)
        self.f_2D.cuda()
        self.parameters = [self.f_2D.parameters()]
        if self.useGlobal:
            self.g_global = Siren(in_features=1, out_features=4 if self.useMatrix else 2,
                                  hidden_features=16, hidden_layers=2,
                                  outermost_linear=True)
            self.parameters.append(self.g_global.parameters())
            self.g_global.cuda()
        if self.useLocal:
            self.g_local = Siren(in_features=3, out_features=2,
                                 hidden_features=128, hidden_layers=4,
                                 outermost_linear=True)
            self.g_local.cuda()
            self.parameters.append(self.g_local.parameters())
    def forward(self,xyt):
        h_local = self.g_local(xyt) if self.useLocal else torch.tensor(0.0)
        if self.useDeformation:
            h_local=2*torch.sigmoid(h_local)-1
            h_local=h_local*self.deformationSize
        if self.useMatrix and self.useGlobal:
            c =self.g_global(xyt[:, [-1]])
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
            xy_ = new_uv + h_local
        else:
            h_global = self.g_global(xyt[:, [-1]]) if self.useGlobal else 0
            xy_ = xyt[:, :-1] + h_global + h_local
        return torch.sigmoid(self.f_2D(xy_)),{
            "xy_":xy_,
            "h_local":h_local
        }
import itertools
import math
class Decouple(nn.Module):
    def log(self,x): #x中的全部元素均不能为负数
        import math
        e = math.e # 获取自然数 e 的值
        eps = e ** -101
        # return x
        # eps = 1e-10 #torch.finfo(torch.float64).eps
        return -1.*torch.log(x+eps) # return -1.*torch.log(x.abs()+eps)
    def __init__(self, path):
        super().__init__()
        v = VideoFitting(path)
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
            self.f_soft_list.append(Layer(useGlobal=False))
        # 二、刚体
        self.NUM_figid = 3 # 刚体层的数据
        self.f_rigid_list=[]
        for i in range(self.NUM_figid):
            self.f_rigid_list.append(Layer(useDeformation=True))
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
        ground_truth = self.ground_truth
        mask = self.mask
        o, layers, p = self.forward(xyt)
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
            # loss_rigid = loss_rigid+1.20 * p["h_local"][i].abs().mean() # 减少刚体的信息量
            loss_rigid = loss_rigid + p["h_local"][i].abs().mean()  # 减少刚体的信息量
        loss_rigid = loss_rigid * weight_init[0] #(10**2)
        # 二、软体
        # loss_soft = (10**-5) * (1. - p["o_soft_all"]).abs().mean() # 减少软体的信息量
        loss_soft = (1. - p["o_soft_all"]).abs().mean() * weight_init[1] #(0.1) # 减少软体的信息量
        # 三、流体 # 减少流体的信息量
        # loss_fluid = 0.02 * o_fluid.abs().mean() # 减少流体的信息量
        # loss_fluid = loss_fluid + 0.01 * (o_fluid*(1-o_fluid)).abs().mean() # 二值化约束
        # loss_fluid = (10**5) * fluidInf(o_fluid.abs()) # 减少流体的信息量
        loss_fluid = fluidInf(o_fluid.abs()) * weight_init[2] #1 # 减少流体的信息量
        # 刚体:1.2->1;软体:0.1**5->0.1;流体:10**5->10

        if False:#放弃了平衡信息量的做法
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

myMain=Decouple(pathIn)
myMain.train(EpochNum) #EpochNum =5000
orig = myMain.v.video.clone()
# g, f1, f2, orig = train_reflection('./data/in', 3000)

def save2img(imgs,path):
    if not os.path.exists(path):os.makedirs(path)
    for i in range(imgs.shape[0]):
        image_array = imgs[i]
        image = Image.fromarray(image_array, mode='L')
        image.save(os.path.join(path, str(i).zfill(5) + '.png'))
def save0(o_scene,tag):
    N, _, H, W = orig.size()
    o_scene = o_scene.view(H, W, N, 1).permute(2, 0, 1, 3).cpu().detach().numpy()
    o_scene = (o_scene * 255).astype(np.uint8)
    save2img(o_scene[:, :, :, 0], os.path.join(outpath, tag))
# if False:
with torch.no_grad():
    orig = orig.permute(0, 2, 3, 1).detach().numpy()
    orig = (orig * 255).astype(np.uint8)
    save2img(orig[:, :, :, 0], os.path.join(outpath, 'orig'))

    if False:
        N, _, H, W = orig.size()
        xyt = get_mgrid([H, W, N]).cuda()
        o, layers, p=myMain(xyt) # h = g(xyt)
        save0(o, "recon")
        save0(p["o_rigid_all"], "rigid")
        save0(p["o_soft_all"], "soft")

        for i in range(len(layers["r"])):
            save0(layers["r"][i], "rigid"+str(i))
        for i in range(len(layers["s"])):
            save0(layers["s"][i], "soft"+str(i))
        # save0(layers[1], "soft")
        save0(layers["f"], "fluid")

###########################################################################################
orig = myMain.v.video.clone()
N, C, H, W = orig.size()  # 帧数、通道数、高度、宽度
orig = orig.permute(0, 2, 3, 1).detach()#.numpy()

# 创建空列表存储每帧预测结果
pred_frames = []
layers_frames = {
    "r": [],
    "s": [],
    "f": []
}
p_frames = {
    "o_rigid_all":[],
    "o_soft_all":[]
}

# 生成时间轴的归一化值（-1到1）
if N > 1:
    t_vals = torch.linspace(-1, 1, steps=N).cuda()
else:
    t_vals = torch.zeros(1).cuda()
# N, _, H, W = orig.size()
def save1(o_scene,tag):
    o_scene = o_scene.cpu().detach().numpy()
    o_scene = (o_scene * 255).astype(np.uint8)
    save2img(o_scene[:, :, :, 0], os.path.join(outpath, tag))
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
        frame_output, layers, p = myMain(coords)

        # 调整形状为图像格式 (C, H, W)
        frame_image = frame_output.view(H, W, 1)#.permute(2, 0, 1)
        # print("frame_image",frame_image.shape)

        pred_frames.append(frame_image)
        for id in layers_frames:
            if id=="f":
                layers_frames[id].append(layers[id].view(H, W, 1))
            else:
                layers_frames[id].append(layers[id])
        for id in p_frames:
            p_frames[id].append(p[id].view(H, W, 1))
# exit(0)
# 组合所有帧 (N, C, H, W)
video_pre = torch.stack(pred_frames, dim=0)
o_rigid_list=[]
o_soft_list=[]
def p01(original_list):
    l = list(map(list, zip(*original_list))) #交换列表的前两层
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
    "o_rigid_all":torch.stack(p_frames["o_rigid_all"], dim=0),
    "o_soft_all":torch.stack(p_frames["o_soft_all"], dim=0)
}
save1(video_pre, "recon")
save1(orig.cuda()/(video_pre.abs()+10**-10), "recon_non")
save1(0.5*orig.cuda()/(video_pre.abs()+10**-10), "recon_non2")
save1(p["o_rigid_all"], "rigid")
save1(p["o_soft_all"], "soft")
# print(orig.shape,orig.max(),orig.min(),orig.mean(),orig.std())
# print(p["o_rigid_all"].shape,p["o_rigid_all"].max(),p["o_rigid_all"].min(),p["o_rigid_all"].mean())
# exit(0)
save1(orig.cuda()/(p["o_rigid_all"].abs()+10**-10), "rigid_non")
save1(0.5*orig.cuda()/(p["o_rigid_all"].abs()+10**-10), "rigid_non2")

for i in range(len(layers["r"])):
    save1(layers["r"][i], "rigid" + str(i))
for i in range(len(layers["s"])):
    save1(layers["s"][i], "soft" + str(i))
save1(layers["f"], "fluid")

mainFreeCOS(paramPath,os.path.join(outpath, "recon_non2"),os.path.join(outpath, "mask2"))
check(os.path.join(outpath, "mask2"),videoId,"nir.1.recon_non2")
'''
python -m nir.myReflection3
'''




