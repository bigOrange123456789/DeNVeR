import os
import subprocess
import torch
from concurrent import futures
script_path = os.path.abspath(__file__)
ROOT = os.path.dirname(script_path)
def runRAFT(rgb_dir,out_dir,out_img_dir,gap):
    path0 = os.path.join(ROOT, "../scripts")
    batch_size = 8
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    if not os.path.exists(out_img_dir): os.makedirs(out_img_dir)
    cmd = f"cd {path0} && python run_raft.py {rgb_dir} {out_dir} -I {out_img_dir} --gap {gap} -b {batch_size}"
    subprocess.call(cmd, shell=True)

from nir.myLib.mySave import save2img

import numpy as np
import data
import utils
import torch.nn.functional as F
class Track():
    def save(self, data, path):
        data = ( data.cpu().detach().numpy() * 255).astype( np.uint8 )
        save2img(data, path)
    def showDatasetOcc(self,dset,path):
        data_list = self.getDatasetOcc(dset)
        self.save(data_list[:,0],path)
    def getDatasetOcc(self,dset):
        data_list = []
        i = 0
        for occ in dset:
            data_list.append(occ[0])
            i = i + 1
            if i >= len(dset): break
        return torch.stack(data_list, dim=0)
    def getInv(self,masks, gap):
        masks = masks[:,[[0]]]
        batch_in = self.batch
        if gap > 0:  # 正向光流
            ok0, flow = batch_in["fwd"]  # (B, 2, H, W) #没有找到对应的文件才会不OK
            # ok=[ True,  True,  True,  True,  True,  True,  True,  True, False]
            # flow.shape=[9, 2, 128, 128] 光流图是双通道
            occ_map = batch_in["occ"][0]  # occ.shape=[9, 1, 128, 128] #前向遮挡图
            ok = ok0[:-gap]  # 去除最后一个false
            F12 = flow[:-gap].permute(0, 2, 3, 1)  # 0 ... B-1-gap #[8,128,128,2] #有效的光流图视频
            O12 = occ_map[:-gap]  # [8,1,128,128] #遮挡图视频
            M1 = masks[:-gap, :, 0, ...]  # 0 ... B-1-gap #[8, 2, 128, 128] #去除最后一张MASK图
            M2 = masks[gap:, :, 0, ...]  # gap ... B-1   #[8, 2, 128, 128] #去除第一张MASK图
        else:  # 逆向光流
            ok0, flow = batch_in["bck"]
            occ_map = batch_in["disocc"][0]
            ok = ok0[-gap:]
            F12 = flow[-gap:].permute(0, 2, 3, 1)  # gap ... B-1
            O12 = occ_map[-gap:]
            M1 = masks[-gap:, :, 0, ...]  # gap ... B-1
            M2 = masks[:gap, :, 0, ...]  # 0 ... B-1-gap

        M1, M2 = M1[ok], M2[ok]  # 输出的MASK
        F12, O12 = F12[ok], O12[ok]  # 输入的光流图和遮挡图

        # mask 1 resampled from mask 2
        # print("masks", masks.shape)
        # print("M2, F12, O12", M2.shape, F12.shape, O12.shape)
        # exit(0)
        M2_new = utils.inverse_flow_warp(M2, F12, O12) # 光流图记录了起始图片中每个像素的新坐标
        # M2_new = torch.where(~(O12 == 1), M2_new, M2)
        # print("M2",M2.min().item(),O12 is not None,"M2_new",M2_new.min().item())
        masks2 = masks[:,0] # [16, 1, 512, 512] <= [16, 1, 1, 512, 512]
        myone = torch.ones_like(masks2[[0]])
        if gap > 0:  #逆向预测 #补充后面 #这里这考虑了+1或-1的情况，没有考虑gap为其他整数的情况
            M2_new  = torch.cat([M2_new, masks2[-gap:]], dim=0) # [16, 1, 512, 512] <= [15+1, 1, 512, 512]
            O12_new = torch.cat([O12   , myone        ], dim=0) # [16, 1, 512, 512] <= [15+1, 1, 512, 512]
        else: #正向预测 #补充前面
            M2_new  = torch.cat([masks2[0:-gap], M2_new], dim=0)
            O12_new = torch.cat([myone         , O12   ], dim=0) # [16, 1, 512, 512] <= [15+1, 1, 512, 512]
        # M2_new[O12_new==1]=
        if True: #被遮挡的区域预测结果不变
            M2_new = torch.where(O12_new == 1, masks2, M2_new) #被遮挡使用原数据,否则使用预测数据
        else:#被遮挡的区域预测结果倾向于背景 #效果不佳
            M2_new = torch.where(O12_new == 1, 0.8*masks2, M2_new)
        return ok0, M2_new  # 根据光流图，从终点MASK2得到起始MASK1
    def initTrack(self,imgs0): # imgs = [10, 1, 128, 128]
        def shift(A,tag="imgsF"): #适用于imgsF
            n = A.size(0)
            shifted = torch.zeros_like(A)  # 初始化全 0
            shiftType = "ROW2"
            if shiftType == "ROW":# 逐行移动 #第一行表示原始数据 #没有检测是否正确
                for i in range(n):  #右移 i 位, 第 0 行不动
                    if tag=="imgsF": #
                        shifted[i, i:] = A[i, :n - i]  # 直接赋值，左侧补 0
                    elif tag=="imgsB":
                        shifted[i, :n - i] = A[i, i:]  # 直接赋值，右侧补 0
            elif shiftType == "COL":# 逐列移动 #对角线表示原始数据 #错误
                for i in range(n):
                    if tag=="imgsF": #
                        NUM = n - i # 移动元素个数
                        LEN = i   # 移动距离
                    elif tag=="imgsB":
                        NUM = i + 1  # 移动元素个数
                        LEN = n - (i + 1)  # 移动距离
                    shifted[LEN:, i] = A[:NUM, i]
            elif shiftType == "ROW2":# 逐行移动
                for i in range(n):
                    if tag=="imgsF": #
                        NUM = n - i # 移动元素个数
                        LEN = i   # 移动距离
                    elif tag=="imgsB":
                        NUM = i + 1  # 移动元素个数
                        LEN = n - (i + 1)  # 移动距离
                    shifted[i, LEN:] = A[i, :NUM]
            return shifted

        # 针对这组图片预测出所有情况
        imgs0=imgs0[:,[0]] # [10, 1, 128, 128] <= [10, 3, 128, 128]
        # print("imgs0", imgs0.shape)
        # exit(0)
        imgsF=[imgs0.clone()]
        imgs=imgs0.clone()
        for i in range(self.length - 1):  # 正向预测
            ok, imgs = self.getInv(imgs, -1) # imgs = [10, 1, 128, 128]
            imgsF.append(imgs.clone())
        imgsF = torch.cat(imgsF, dim=1) #!!!第一列是原始数据,这列添加
        imgsF = shift(imgsF,"imgsF")

        imgsB = [imgs0.clone()]
        imgs = imgs0.clone()
        for i in range(self.length - 1):  # 逆向预测
            ok, imgs = self.getInv(imgs, 1)  # imgs = [10, 1, 128, 128]
            imgsB.append(imgs.clone())
        imgsB = torch.cat(imgsB, dim=1) #第一层每一个初始帧、第二层是行进步长
        imgsB = shift(imgsB,"imgsB")
        # imgsB = imgsB.flip(0) # imgsF = imgsF.permute(1, 0, 2, 3)
        diag_mat = torch.eye(self.length).view(self.length, self.length, 1, 1)
        imgs_track = ( 1- diag_mat ) * imgsF + imgsB.flip(1) # imgsB.flip(0)
        '''
            shiftType == "ROW":
                imgs_track 第0行是原始数据，第1行是经过1步预测的结果，...
            shiftType == "COL":
                每一行是一帧
        '''
        return imgs_track


    def __init__(self,rgb_dir, fwd_dir, bck_dir, mask_dir): #对mask中的每个像素进行追踪
        # outpath = './data'
        # rgb_dir = os.path.join(ROOT, outpath,     tag)
        # fwd_dir = os.path.join(ROOT, outpath,     tag+"_fwd")
        # bck_dir = os.path.join(ROOT, outpath,     tag+"_bck")
        # mask_dir = os.path.join(ROOT, outpath, "A.mask_nr2", "filter")
        # fwd_img_dir = os.path.join(ROOT, outpath, tag+"_fwdImg")
        # bck_img_dir = os.path.join(ROOT, outpath, tag+"_bckImg")

        fwd_img_dir=fwd_dir+"Img"
        bck_img_dir=bck_dir+"Img"
        runRAFT(rgb_dir, fwd_dir, fwd_img_dir, +1) #计算正向光流图
        runRAFT(rgb_dir, bck_dir, bck_img_dir, -1) #计算逆向光流图
        # return

        rgb_dset = data.RGBDataset(rgb_dir)
        fwd_dset = data.FlowDataset(fwd_dir, +1, rgb_dset=rgb_dset)  # 前向光流
        bck_dset = data.FlowDataset(bck_dir, -1, rgb_dset=rgb_dset)  # 后向光流
        occ_dset = data.OcclusionDataset(fwd_dset, bck_dset)  # 遮挡(行进顺序-前后)
        disocc_dset = data.OcclusionDataset(bck_dset, fwd_dset)  # 遮挡(行进顺序-后前)
        epi_dset = data.RGBDataset(mask_dir)  # 黑塞矩阵MASK

        dset = data.CompositeDataset({
            "rgb": rgb_dset,  # custom_videos/PNGImages #原视频
            "fwd": fwd_dset,  # custom_videos/raw_flows_gap1  #正向光流
            "bck": bck_dset,  # custom_videos/raw_flows_gap-1 #逆向光流
            "epi": epi_dset,  # preprocess/--/binary  #黑塞MASK
            "occ": occ_dset,  # 遮挡(行进顺序-前后)
            "disocc": disocc_dset
        })
        self.length=len(fwd_dset)
        val_loader = data.get_ordered_loader(  # 我猜这里是有序加载
            dset,
            self.length,
            preloaded=True
        )
        for batch in val_loader:
            self.batch = batch
            self.vessel = batch["epi"]
            self.rgb = batch["rgb"]

        # 分析追踪的结果是否正确
        def inverse_sigmoid(y: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
            """
            对 sigmoid 输出 y 做逆变换
            :param y: 必须在 (0, 1) 区间
            :param eps: 防止除零 / log(0) 的小常数
            :return: logit(y)
            """
            y = y.clamp(min=eps, max=1 - eps)  # 数值稳定
            return torch.log(y / (1 - y))
        with torch.no_grad():  # 对MASK视频进行导管(追踪)抑制的效果
            # print("vessel_track_occ",vessel_track_occ.mean(),vessel_track_occ.min(),vessel_track_occ.max(),vessel_track_occ.std())
            # exit(0)
            vessel_track = self.initTrack(self.vessel)
            # myone = torch.ones_like(self.vessel)
            # vessel_track_occ = self.initTrack(myone)
            vessel_track_occ = torch.ones_like(vessel_track)

            vessel_track_g=vessel_track.clone()
            vessel_track_occ_g = vessel_track_occ.clone()
            def GaussianWeight(x, sigma=3): #基于高斯函数进行加权
                return np.exp(-(x ** 2) / (2 * sigma ** 2))
            for i in range(vessel_track.shape[0]):
                for j in range(vessel_track.shape[1]):
                    vessel_track_g[i, j] = GaussianWeight(i - j) * vessel_track[i, j]
                    vessel_track_occ_g[i, j] = GaussianWeight(i - j) * vessel_track_occ[i, j]

            vessel_track_gd = vessel_track.clone() # 对角线
            vessel_track_occ_gd = vessel_track_occ.clone()
            def GaussianWeight2(x):  #
                if x==0:return 1
                else:   return 0
            for i in range(vessel_track.shape[0]):
                for j in range(vessel_track.shape[1]):
                    vessel_track_gd[i, j] = GaussianWeight2(i - j) * vessel_track[i, j]
                    vessel_track_occ_gd[i, j] = GaussianWeight2(i - j) * vessel_track_occ[i, j]

            vessel_track_gn = vessel_track.clone() #只考虑相邻的前后帧率
            vessel_track_occ_gn = vessel_track_occ.clone()
            def GaussianWeight3(x):  #
                if abs(x) <= 1:return 1
                else:          return 0
            for i in range(vessel_track.shape[0]):
                for j in range(vessel_track.shape[1]):
                    vessel_track_gn[i, j] = GaussianWeight3(i - j) * vessel_track[i, j]
                    vessel_track_occ_gn[i, j] = GaussianWeight3(i - j) * vessel_track_occ[i, j]

            vessel_track_gn2 = vessel_track.clone()  # 只考虑相邻的前后帧率
            vessel_track_occ_gn2 = vessel_track_occ.clone()
            def GaussianWeight4(x):  #
                if abs(x) == 0: return 0.5
                elif abs(x) == 1: return 0.25
                else: return 0
            for i in range(vessel_track.shape[0]):
                for j in range(vessel_track.shape[1]):
                    vessel_track_gn2[i, j] = GaussianWeight4(i - j) * vessel_track[i, j]
                    vessel_track_occ_gn2[i, j] = GaussianWeight4(i - j) * vessel_track_occ[i, j]

            self.vessel = vessel_track.sum(dim=1)/(vessel_track_occ.sum(dim=1) + 10**-10)

            self.vessel_g = vessel_track_g.sum(dim=1)/(vessel_track_occ_g.sum(dim=1) + 10**-10)

            self.vessel_gd = vessel_track_gd.sum(dim=1) #/ (vessel_track_occ_gd.sum(dim=1) + 10 ** -10) #对角线

            self.vessel_gn = vessel_track_gn.sum(dim=1) / (vessel_track_occ_gn.sum(dim=1) + 10 ** -10) #相邻帧

            self.vessel_gn2 = vessel_track_gn2.sum(dim=1) / (vessel_track_occ_gn2.sum(dim=1) + 10 ** -10)  # 相邻帧

            self.vessel_gn_max = torch.clamp(vessel_track_gn.sum(dim=1), max=1)

            self.vessel3 = torch.clamp(vessel_track.sum(dim=1), max=1)

            vessel_track = inverse_sigmoid(vessel_track)
            vessel2 = vessel_track.sum(dim=1) / (vessel_track_occ.sum(dim=1) + 10 ** -10)
            self.vessel2 = F.sigmoid(vessel2)

            vessel_track_g = inverse_sigmoid(vessel_track_g)
            vessel2_g = vessel_track_g.sum(dim=1) / (vessel_track_occ_g.sum(dim=1) + 10 ** -10)
            self.vessel2_g = F.sigmoid(vessel2_g)

            self.result = {
                "raft": self.vessel,
                "raft2": self.vessel2,
                "raft_g": self.vessel_g,
                "raft2_g": self.vessel2_g,
                "raft_gd": self.vessel_gd,
                "raft_gn": self.vessel_gn,
                "raft_gn2": self.vessel_gn2, #new
                "raft_gn_max": self.vessel_gn_max,
                "raft3": self.vessel3
            }

            # dataSave = self.rgb.clone().permute(0, 2, 3, 1)
            # dataSave[:,:,:,0] = vessel
            # self.save(dataSave, os.path.join(ROOT, outpath, tag + "_vessel"))
            # self.save(vessel, os.path.join(ROOT, outpath, tag + "_vessel2"))
            # for i in range(self.length):
            #     # dataSave = self.rgb.clone().permute(0, 2, 3, 1)
            #     # dataSave[:,:,:,0]=vessel_track_occ[i]
            #     # self.save(dataSave, os.path.join(ROOT, outpath, tag + "_pre"+str(i)))
            #     self.save(vessel_track[:,i], os.path.join(ROOT, outpath, tag + "_test" + str(i)))
            #     # break
            # vessel = self.vessel_gd
            # vessel[vessel<0.5]=0
            # vessel[vessel>=0.5]=1
            # self.save(vessel, os.path.join(ROOT, outpath, tag + "_vessel_new3"))

if __name__ == "__main__":
    # Test("recon_non2")
    print("version: 15:18")
    outpath = './data'
    tag = "CVAI-1207LAO44_CRA29"
    rgb_dir = os.path.join(ROOT, outpath, tag)
    fwd_dir = os.path.join(ROOT, outpath, tag + "_fwd")
    bck_dir = os.path.join(ROOT, outpath, tag + "_bck")
    mask_dir = os.path.join(ROOT, outpath, "A.mask.main_nr2", "filter")
    Track(rgb_dir, fwd_dir, bck_dir, mask_dir)
