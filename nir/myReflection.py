import os
import numpy as np
from itertools import chain

import torch
from torch.utils.data import DataLoader

from model import Siren
from util import get_mgrid, jacobian, VideoFitting

from PIL import Image
outpath = './data/myReflection_global_6'
print("单个刚体、单个流体、单个软体")
print("5.","没有软体信息")
print("6.减少软体的约束")
# print("outpath", outpath)
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
class Main():
    def __init__(self, path):
        v = VideoFitting(path)
        videoloader = DataLoader(v, batch_size=1, pin_memory=True, num_workers=0)
        model_input, ground_truth = next(iter(videoloader))
        model_input, ground_truth = model_input[0].cuda(), ground_truth[0].cuda()
        ground_truth = ground_truth[:, 0:1]  # 将RGB图像转换为灰度图
        self.v=v
        self.model_input=model_input
        self.ground_truth=ground_truth

        #######################################################################

        self.f_soft=Layer()
        self.f_rigid=Layer()

        self.f2 = Siren(in_features=3, out_features=1, hidden_features=128,
                   hidden_layers=4, outermost_linear=True)
        self.f2.cuda()
        self.parameters=[
            self.f2.parameters()
        ]+self.f_rigid.parameters + self.f_soft.parameters

    def synthesis(self,xyt): # soft, rigid, fluid
        # print()
        # 1.刚体
        # self.f_rigid(xyt)
        # h_global = self.g_global(xyt[:, [-1]])#整体位移
        # h_local = self.g(xyt)#局部位移
        # o_scene = torch.sigmoid(self.f1(xyt[:, :-1] + h_local+ h_global))
        o_rigid ,p_rigid= self.f_rigid.forward0(xyt)

        # # 2.软体
        # h_global = self.g_global(xyt[:, [-1]])  # 整体位移
        # h_local = self.g(xyt)  # 局部位移
        # o_scene = torch.sigmoid(self.f1(xyt[:, :-1] + h_local + h_global))
        o_soft, p_soft = self.f_soft.forward0(xyt)

        # 3.流体
        o_fluid = torch.sigmoid(self.f2(xyt))
        # o_obst_fluid = torch.sigmoid(self.f3(xyt))
        o = o_rigid * o_soft * (1.0-o_fluid)
        return o , [o_rigid  , o_soft, o_fluid] ,{
            "h_local":p_rigid["h_local"]
        }

    def loss(self,xyt,ground_truth,step,start,end):
        o, layers, p = self.synthesis(xyt)
        o_soft  = layers[1]
        o_fluid = layers[2]

        # loss_recon = ((o - (1.0-o_fluid)*ground_truth[start:end]) ** 2).mean()*10.
        loss_recon = ((o - ground_truth[start:end]) ** 2).mean() * 10.
        loss_soft  = 0.05 * (1.-o_soft).abs().mean() #减少软体的信息量
        loss_fluid = 0.02 * o_fluid.abs().mean()     #减少流体的信息量
        loss_rigid = 0.20 * p["h_local"].abs().mean()#减少刚体的信息量

        loss = loss_recon + loss_soft + loss_fluid + loss_rigid

        self.layers=layers
        # print("step:", step)
        if not step % 200:
            print("Step [%04d]: loss=%0.8f, recon=%0.8f, loss_soft=%0.8f, loss_fluid=%0.8f, loss_rigid=%0.4f" % (
                step, loss, loss_recon, loss_soft , loss_fluid , loss_rigid))
        return loss

    def train(self,total_steps):
        model_input=self.model_input
        ground_truth=self.ground_truth

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

myMain=Main('data/in')
myMain.train(5000)
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
    o, layers, _=myMain.synthesis(xyt) # h = g(xyt)
    print("o",type(o),o.shape)
    save0(o, "recon")

    save0(layers[0], "rigid")
    save0(layers[1], "soft")
    save0(layers[2], "fluid")

    orig = orig.permute(0, 2, 3, 1).detach().numpy()
    orig = (orig * 255).astype(np.uint8)
    save2img(orig[:, :, :, 0], os.path.join(outpath, 'orig'))

