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
    print(cmd)
    subprocess.call(cmd, shell=True)
    # return
    # exe = os.path.join(ROOT, "../scripts", "run_raft.py")
    # batch_size = 8
    # cmd = f"python {exe} {rgb_dir} {out_dir} -I {out_img_dir} --gap {gap} -b {batch_size}"
    # print(cmd)
    # subprocess.call(cmd, shell=True)

from nir.myLib.mySave import save2img

import numpy as np
import data
import utils
class Test():
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
        M2_new = utils.inverse_flow_warp(M2, F12, O12)
        masks2=masks[:,0]
        if gap > 0:  #逆向预测 #补充后面 # 这里这考虑了+1或-1的情况，没有考虑gap为其他整数的情况
            M2_new=torch.cat([M2_new,masks2[-gap:]], dim=0)
        else: # 正向预测 #补充前面
            M2_new = torch.cat([masks2[0:-gap], M2_new], dim=0)
        return ok0, M2_new  # 根据光流图，从终点MASK2得到起始MASK1

    def initPredict(self,imgs0): # imgs = [10, 1, 128, 128]
        def shift_rows(A):
            n = A.size(0)
            shifted = torch.zeros_like(A)  # 初始化全 0
            for i in range(n): # 右移 i 位（第 0 行不动）
                shifted[i, i:] = A[i, :n - i]  # 直接赋值，左侧补 0
            return shifted
        # 针对这组图片预测出所有情况
        imgs0=imgs0[:,[0]]
        imgsF=[imgs0.clone()]
        imgs=imgs0.clone()
        for i in range(self.length - 1):  # 正向预测
            ok, imgs = self.getInv(imgs, -1) # imgs = [10, 1, 128, 128]
            imgsF.append(imgs.clone())
        imgsF = torch.cat(imgsF, dim=1)
        imgsF = shift_rows(imgsF)

        imgsB = [imgs0.clone()]
        imgs = imgs0.clone()
        for i in range(self.length - 1):  # 逆向预测
            ok, imgs = self.getInv(imgs, 1)  # imgs = [10, 1, 128, 128]
            imgsB.append(imgs.clone())
        imgsB = torch.cat(imgsB, dim=1) #第一层每一个初始帧、第二层是行进步长
        imgsB = shift_rows(imgsB)
        # print("F",imgsF.shape)
        # print("B",imgsB.shape)
        # print("len",self.length)
        # print("eye",torch.eye(self.length).shape)
        imgsF = imgsF.permute(1, 0, 2, 3)
        diag_mat = 1 - torch.eye(self.length).view(self.length, self.length, 1, 1)
        imgsPre = imgsF + imgsB * diag_mat
        return imgsPre

    def __init__(self,tag,outpath = './data/myReflection4_03'):
        rgb_dir = os.path.join(ROOT, outpath,     tag)
        fwd_dir = os.path.join(ROOT, outpath,     tag+"_fwd")
        fwd_img_dir = os.path.join(ROOT, outpath, tag+"_fwdImg")
        bck_dir = os.path.join(ROOT, outpath,     tag+"_bck")
        bck_img_dir = os.path.join(ROOT, outpath, tag+"_bckImg")
        occ_dir = os.path.join(ROOT, outpath,     tag+"_occ")
        disocc_dir = os.path.join(ROOT, outpath,  tag+"_disocc")
        mask_dir = os.path.join(ROOT, outpath, "..", "mask","binary")

        runRAFT(rgb_dir, fwd_dir, fwd_img_dir, +1)
        runRAFT(rgb_dir, bck_dir, bck_img_dir, -1)

        rgb_dset = data.RGBDataset(rgb_dir)
        fwd_dset = data.FlowDataset(fwd_dir, +1, rgb_dset=rgb_dset)  # 前向光流
        bck_dset = data.FlowDataset(bck_dir, -1, rgb_dset=rgb_dset)  # 后向光流
        occ_dset = data.OcclusionDataset(fwd_dset, bck_dset)  # 遮挡(行进顺序-前后)
        disocc_dset = data.OcclusionDataset(bck_dset, fwd_dset)  # 遮挡(行进顺序-后前)
        epi_dset = data.RGBDataset(mask_dir)  # 黑塞矩阵MASK

        # batch={
        #     "fwd":self.getDataset(fwd_dset),
        #     "bck":self.getDataset(bck_dset),
        #     "occ":self.getDataset(occ_dset),
        #     "disocc":self.getDataset(disocc_dset),
        # }
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
            self.masks = batch["epi"]
            self.rgb =batch["rgb"]

        def test01():
            masks = self.masks
            ok, masks_inv = self.getInv(masks, +1) #逆向前进
            self.save(masks_inv[:,0], os.path.join(ROOT, outpath, tag + "_maskF"))
            ok, masks_inv = self.getInv(masks, -1) #正向前进
            self.save(masks_inv[:,0], os.path.join(ROOT, outpath, tag + "_maskB"))

        def test02(): # 对MASK视频进行导管(追踪)抑制的效果
            masks = self.masks
            imgs = [masks[[0],0]]
            for i in range(self.length-1):#正向预测
                ok, masks = self.getInv(masks, -1)
                imgs.append(masks[[i+1],0])
            imgs = torch.cat(imgs, dim=0) # imgs.shape=[10, 128, 128]

            self.save(
                imgs,
                os.path.join(ROOT, outpath, tag+"_headTest")
            )
            masks2 = self.masks[:,0] - 0.5 * imgs
            masks2[ masks2<0.2 ] = 0.2
            self.save(
                masks2,
                os.path.join(ROOT, outpath, tag + "_masks2")
            )

        def test03():
            self.showDatasetOcc(occ_dset, occ_dir)
            self.showDatasetOcc(disocc_dset, disocc_dir)

        # 分析追踪的结果是否正确
        with torch.no_grad():  # 对MASK视频进行导管(追踪)抑制的效果
            imgsPre = self.initPredict(self.masks)
            for i in range(self.length):
                dataSave = self.rgb.clone().permute(0, 2, 3, 1)
                dataSave[:,:,:,0]=imgsPre[i]
                self.save(dataSave, os.path.join(ROOT, outpath, tag + "_pre"+str(i)))

if __name__ == "__main__":
    if False:Test("orig")
    Test("recon_non2")

    # myDataset = data.CompositeDataset({
    #     "fwd": fwd_dset,  # custom_videos/raw_flows_gap1  #正向光流
    #     "bck": bck_dset,  # custom_videos/raw_flows_gap-1 #逆向光流
    #     "occ": occ_dset,  # 遮挡(行进顺序-前后)
    #     "disocc": disocc_dset,  # 遮挡(行进顺序-后前)
    # })

'''
export PATH="~/anaconda3/bin:$PATH"
source activate DNVR
python -m nir.pixelTracking
'''
