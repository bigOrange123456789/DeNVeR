import os
import subprocess
import torch
from concurrent import futures
script_path = os.path.abspath(__file__)
ROOT = os.path.dirname(script_path)
def runRAFT(rgb_dir,out_dir,out_img_dir,gap):
    path0 = os.path.join(ROOT, "../scripts")
    batch_size = 165 #造影视频含有164帧数据 #8 #100
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    if not os.path.exists(out_img_dir): os.makedirs(out_img_dir)
    cmd = f"cd {path0} && python run_raft.py {rgb_dir} {out_dir} -I {out_img_dir} --gap {gap} -b {batch_size}"
    print(cmd)
    subprocess.call(cmd, shell=True)

if __name__ == "__main__":
    tag =  "recon_non2" # "orig"
    outpath = './data/myReflection4_03'
    rgb_dir = os.path.join(ROOT, outpath, tag)
    fwd_dir = os.path.join(ROOT, outpath, tag + "_fwd")
    fwd_img_dir = os.path.join(ROOT, outpath, tag + "_fwdImg")

    runRAFT(rgb_dir, fwd_dir, fwd_img_dir, +1)

'''
export PATH="~/anaconda3/bin:$PATH"
source activate DNVR
python -m nir.raftSim 
'''
