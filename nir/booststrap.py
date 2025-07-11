import os
import numpy as np
from itertools import chain
import cv2
import torch
from torch.utils.data import DataLoader
import imageio
import argparse
from model import Siren
from util import get_mgrid, jacobian, VideoFitting

import yaml
# 指定 YAML 文件路径
script_path = os.path.abspath(__file__)
ROOT1 = os.path.dirname(script_path)
file_path = os.path.join(ROOT1,'../','./confs/newConfig.yaml')
# 打开并读取 YAML 文件
with open(file_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
nir_image_path = os.path.join(config["my"]["filePathRoot"],config["my"]["subPath"]["nirIn"])

def train_fence(path, total_steps, lambda_interf=0.5, lambda_flow=0.5, verbose=True, steps_til_summary=100):
    g = Siren(in_features=3, out_features=2, hidden_features=256,
              hidden_layers=4, outermost_linear=True)#in:(x,y,t) out:(dx,dy)
    g.cuda()
    f1 = Siren(in_features=2, out_features=3, hidden_features=256,
               hidden_layers=4, outermost_linear=True, first_omega_0=90.)
    f1.cuda()#in:(x+dx,y+dy) out:(r,g,b) #场景计算
    f2 = Siren(in_features=3, out_features=4, hidden_features=256, 
               hidden_layers=4, outermost_linear=True)
    f2.cuda() #in:(x,y,t) out:(a,b,c,d)干扰计算

    optim = torch.optim.Adam(lr=1e-4, params=chain(g.parameters(), f1.parameters(), f2.parameters()))

    v = VideoFitting(path)
    videoloader = DataLoader(v, batch_size=1, pin_memory=True, num_workers=0)
    model_input, ground_truth = next(iter(videoloader))
    model_input, ground_truth = model_input[0].cuda(), ground_truth[0].cuda()

    batch_size = (v.H * v.W) // 8
    for step in range(total_steps):
        start = (step * batch_size) % len(model_input)
        end = min(start + batch_size, len(model_input))

        xyt = model_input[start:end].requires_grad_()
        xy, t = xyt[:, :-1], xyt[:, [-1]]
        h = g(xyt)#in:(x,y,t) out:(dx,dy)
        xy_ = xy + h #(x2,y2)=(x1,y1)+(dx,dy)
        o_scene = torch.sigmoid(f1(xy_))#场景
        o_obst = torch.sigmoid(f2(xyt))#干扰
        o_obst, alpha = o_obst[:, :-1], o_obst[:, [-1]]
        o = (1 - alpha) * o_scene + alpha * o_obst
        loss_recon = ((o - ground_truth[start:end]) ** 2).mean()
        loss_interf = alpha.abs().mean()
        loss_flow = jacobian(h, xyt).abs().mean()
        loss = loss_recon + lambda_interf * loss_interf + lambda_flow * loss_flow
        if not step % steps_til_summary:
            print("Step [%04d/%04d]: recon=%0.8f, interf=%0.4f, flow=%0.4f" % (step, total_steps, loss_recon, loss_interf, loss_flow))

        optim.zero_grad()
        loss.backward()
        optim.step()

    return g, f1, f2, v.video
def channel_stack(data,filePathRoot):
    parent_path = data[:9]
    # print(parent_path)
    # print("filePathRoot",filePathRoot)
    folder_path = os.path.join("xca_dataset", parent_path, 'images', data)
    folder_path = os.path.join(filePathRoot,"xca_dataset",parent_path,'images',data)
    folder_path = os.path.join(ROOT1, "../", config["my"]["datasetPath"],parent_path,'images',data)
    # print(folder_path,"folder_path")
    # print("datasetPath:",config["my"]["datasetPath"])
    # exit(0)
    count = 0
    for filename in sorted(os.listdir(folder_path)):
        f = os.path.join(folder_path,filename)
        # print("f",f)
        gray_image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        # print(sorted(os.listdir(folder_path)))
        rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        path = os.path.join(nir_image_path,data,filename)
        cv2.imwrite(path,rgb_image)
        count +=1
        if count > 13:
            break
    return True

from PIL import Image
def save2img(imgs,path):
    if not os.path.exists(path):os.makedirs(path)
    for i in range(imgs.shape[0]):
        image_array = imgs[i]
        image = Image.fromarray(image_array, mode='L')
        image.save(os.path.join(path, str(i).zfill(5) + '.png'))

def main(args):
    data = args.data
    # base_path = os.path.join(config["my"]["filePathRoot"],"nir_image")#
    base_path = nir_image_path
    if not os.path.exists( base_path ): os.mkdir( base_path )
    path = os.path.join(base_path,data)#这应该是输入数据路径
    os.makedirs(path,exist_ok=True)
    stack_flag = channel_stack(data,args.filePathRoot)
    total_steps=3 if config["my"]["TestFlag"] else 3000
    print("total_steps",total_steps)
    g, f1, f2, orig = train_fence(path, total_steps)
    #g:相机运动记录, f1:场景获取器, f2:干扰获取器, orig:原输入视频
    with torch.no_grad():
        N, _, H, W = orig.size()#512*512*5，包含5帧图片的视频
        xyt = get_mgrid([H, W, N]).cuda()#shape=[1310720=512*512*5, 3] 整个视频所有点的三维坐标(x,y,t)
        h = g(xyt) #torch.Size([1310720, 2]) 给出每个像素的(dx,dy)
        o_scene = torch.sigmoid(f1(xyt[:, :-1] + h))
        # xyt[:, :-1].shape = [1310720, 2]
        o_obst = torch.sigmoid(f2(xyt))
        o_obst = o_obst[:, :-1] * o_obst[:, [-1]]
        o_scene = o_scene.view(H, W, N, 3).permute(2, 0, 1, 3).cpu().detach().numpy()
        o_obst = o_obst.view(H, W, N, 3).permute(2, 0, 1, 3).cpu().detach().numpy()
        o_scene = (o_scene * 255).astype(np.uint8)
        save2img(o_scene[:, :, :, 0], os.path.join(args.outpath, "my_test", 'scene_r'))
        save2img(o_scene[:, :, :, 1], os.path.join(args.outpath, "my_test", 'scene_g'))
        save2img(o_scene[:, :, :, 2], os.path.join(args.outpath, "my_test", 'scene_b'))
        o_obst = (o_obst * 255).astype(np.uint8)
        save2img(o_obst[:, :, :, 0], os.path.join(args.outpath, "my_test", 'obstruct_r'))
        save2img(o_obst[:, :, :, 1], os.path.join(args.outpath, "my_test", 'obstruct_g'))
        save2img(o_obst[:, :, :, 2], os.path.join(args.outpath, "my_test", 'obstruct_b'))
        o_recon=o_scene+o_obst
        save2img(o_recon[:, :, :, 0], os.path.join(args.outpath, "my_test", 'recon_r'))
        save2img(o_recon[:, :, :, 1], os.path.join(args.outpath, "my_test", 'recon_g'))
        save2img(o_recon[:, :, :, 2], os.path.join(args.outpath, "my_test", 'recon_b'))
        o_scene = [o_scene[i] for i in range(len(o_scene))]
        o_obst = [o_obst[i] for i in range(len(o_obst))]#干扰数据没使用
        orig = orig.permute(0, 2, 3, 1).detach().numpy()
        save2img(orig[:, :, :, 0], os.path.join(args.outpath, "my_test", 'orig_r'))
        save2img(orig[:, :, :, 1], os.path.join(args.outpath, "my_test", 'orig_g'))
        save2img(orig[:, :, :, 2], os.path.join(args.outpath, "my_test", 'orig_b'))
        orig = (orig * 255).astype(np.uint8)
        orig = [orig[i] for i in range(len(orig))]#原视频数据没使用
    p = os.path.join(args.outpath, data)
    # p = os.path.join("nirs",data)
    os.makedirs(p,exist_ok=True)
    name =os.path.join(p,"scene.png")
    # o_scene是一个包含5张图片的list
    # o_scene[-1].shape=(512, 512, 3)
    cv2.imwrite(name,o_scene[-1])#应该是只使用了最后一帧的场景图片
    # print("o_scene(2):",type(o_scene),len(o_scene))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data")
    parser.add_argument("--outpath")
    parser.add_argument("--filePathRoot")
    args = parser.parse_args()
    # args.filePathRoot=""
    # args.outpath="nirs"
    main(args)

'''

export PATH="~/anaconda3/bin:$PATH"
source activate DNVR
python nir/booststrap.py --filePathRoot log --data CVAI-2828RAO11_CRA11 --outpath log/nirs
python nir/booststrap.py --filePathRoot log --data CVAI-2855LAO26_CRA31 --outpath log/nirs

'''
