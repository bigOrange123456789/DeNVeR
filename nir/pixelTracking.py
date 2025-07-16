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

def analysis(gap, batch_in):
    if gap > 0:  # 正向光流
        ok, flow = batch_in["fwd"]  # (B, 2, H, W) #没有找到对应的文件才会不OK
        # ok=[ True,  True,  True,  True,  True,  True,  True,  True, False]
        # flow.shape=[9, 2, 128, 128] 光流图是双通道
        occ_map = batch_in["occ"][0]  # occ.shape=[9, 1, 128, 128] #前向遮挡图
        ok = ok[:-gap]  # 去除最后一个false
        F12 = flow[:-gap].permute(0, 2, 3, 1)  # 0 ... B-1-gap #[8,128,128,2] #有效的光流图视频
        O12 = occ_map[:-gap]  # [8,1,128,128] #遮挡图视频
        # M1 = masks[:-gap, :, 0, ...]  # 0 ... B-1-gap #[8, 2, 128, 128] #去除最后一张MASK图
        # M2 = masks[gap:, :, 0, ...]  # gap ... B-1   #[8, 2, 128, 128] #去除第一张MASK图
    else:  # 逆向光流
        ok, flow = batch_in["bck"]
        occ_map = batch_in["disocc"][0]
        ok = ok[-gap:]
        F12 = flow[-gap:].permute(0, 2, 3, 1)  # gap ... B-1
        O12 = occ_map[-gap:]
        # M1 = masks[-gap:, :, 0, ...]  # gap ... B-1
        # M2 = masks[:gap, :, 0, ...]  # 0 ... B-1-gap

    # M1, M2 = M1[ok], M2[ok]  # 输出的MASK
    F12, O12 = F12[ok], O12[ok]  # 输入的光流图和遮挡图

from nir.myLib.mySave import save2img

def save(o_scene, path):
    o_scene = o_scene.cpu().detach().numpy()
    o_scene = (o_scene * 255).astype(np.uint8)
    save2img(o_scene[:, 0], path)
def showDataset(dset,path):
    data_list = []
    i = 0
    for occ in dset:
        data_list.append(occ[0])
        i = i + 1
        if i >= len(dset): break
    data_list = torch.stack(data_list, dim=0)
    save(data_list,path)

import numpy as np
import data
def main(tag):
    outpath = './data/myReflection4_03'
    rgb_dir = os.path.join(ROOT, outpath,     tag)
    fwd_dir = os.path.join(ROOT, outpath,     tag+"_fwd")
    fwd_img_dir = os.path.join(ROOT, outpath, tag+"_fwdImg")
    bck_dir = os.path.join(ROOT, outpath,     tag+"_bck")
    bck_img_dir = os.path.join(ROOT, outpath, tag+"_bckImg")
    occ_dir = os.path.join(ROOT, outpath,     tag+"_occ")
    disocc_dir = os.path.join(ROOT, outpath,  tag+"_disocc")

    # runRAFT(rgb_dir, fwd_dir, fwd_img_dir, +1)
    # runRAFT(rgb_dir, bck_dir, bck_img_dir, -1)
    with futures.ProcessPoolExecutor(max_workers=1) as ex:#只有一个编号为0的GPU
        ex.submit(
            runRAFT,
            rgb_dir, fwd_dir, fwd_img_dir, +1
        )
    with futures.ProcessPoolExecutor(max_workers=1) as ex:#只有一个编号为0的GPU
        ex.submit(
            runRAFT,
            rgb_dir, bck_dir, bck_img_dir, -1
        )

    print("fwd_dir",fwd_dir)
    fwd_dset = data.FlowDataset(fwd_dir, +1)  # 前向光流
    bck_dset = data.FlowDataset(bck_dir, -1)  # 后向光流
    occ_dset = data.OcclusionDataset(fwd_dset, bck_dset)  # 遮挡(行进顺序-前后)
    disocc_dset = data.OcclusionDataset(bck_dset, fwd_dset)  # 遮挡(行进顺序-后前)
    showDataset(occ_dset, occ_dir)
    showDataset(disocc_dset, disocc_dir)
if __name__ == "__main__":
    #cd nir python pixelTracking.py
    main("orig")
    main("recon_non2")

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
