import os
import subprocess
import torch
# from concurrent import futures
script_path = os.path.abspath(__file__)
ROOT = os.path.dirname(script_path)
def runRAFT(rgb_dir,out_dir,out_img_dir,gap):
    path0 = os.path.join(ROOT, "../scripts")
    batch_size = 8#165 #造影视频含有164帧数据 #8 #100
    # if not os.path.exists(out_dir): os.makedirs(out_dir)
    # if not os.path.exists(out_img_dir): os.makedirs(out_img_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_img_dir, exist_ok=True)
    cmd = f"cd {path0} && python run_raft.py {rgb_dir} {out_dir} -I {out_img_dir} --gap {gap} -b {batch_size}"
    # print(cmd)
    subprocess.call(cmd, shell=True)
from tqdm import tqdm
def traverseVideo():
    datasetPath=os.path.join(ROOT,"../","../DeNVeR_in/xca_dataset")
    # paramPath = "../DeNVeR_in/models_config/freecos_Seg.pt"
    patient_names = [name for name in os.listdir(datasetPath)
                 if os.path.isdir(os.path.join(datasetPath, name))]
    CountSum = 0
    for patientID in patient_names:
        patient_path = os.path.join(datasetPath, patientID, "images")
        video_names = [name for name in os.listdir(patient_path)
                     if os.path.isdir(os.path.join(patient_path, name))]
        CountSum = CountSum + len(video_names)

    print("tag=fwd")
    with tqdm(total=CountSum) as pbar:
     for patientID in patient_names:
        patient_path = os.path.join(datasetPath, patientID, "images")
        video_names = [name for name in os.listdir(patient_path)
                     if os.path.isdir(os.path.join(patient_path, name))]
        for videoId in video_names:
            inpath = os.path.join(datasetPath, patientID, "images", videoId)
            outpath = os.path.join(datasetPath, patientID, "daft", videoId)
            pbar.set_postfix(videoId=f"{videoId}")
            runRAFT(inpath, outpath+"_fwd", outpath+"_fwdImg", +1)
            pbar.update(1)  # 每次增加 1
    print("tag=bck")
    with tqdm(total=CountSum) as pbar:
        for patientID in patient_names:
            patient_path = os.path.join(datasetPath, patientID, "images")
            video_names = [name for name in os.listdir(patient_path)
                           if os.path.isdir(os.path.join(patient_path, name))]
            for videoId in video_names:
                inpath = os.path.join(datasetPath, patientID, "images", videoId)
                outpath = os.path.join(datasetPath, patientID, "daft", videoId)
                pbar.set_postfix(videoId=f"{videoId}")
                runRAFT(inpath, outpath + "_bck", outpath + "_bckImg", -1)
                pbar.update(1)  # 每次增加 1


if __name__ == "__main__":
    traverseVideo()
    # tag =  "recon_non2" # "orig"
    # outpath = './data/myReflection4_03'
    # rgb_dir = os.path.join(ROOT, outpath, tag)
    # fwd_dir = os.path.join(ROOT, outpath, tag + "_fwd")
    # fwd_img_dir = os.path.join(ROOT, outpath, tag + "_fwdImg")
    #
    # runRAFT(rgb_dir, fwd_dir, fwd_img_dir, +1)

'''
export PATH="~/anaconda3/bin:$PATH"
source activate DNVR
python -m RAFT.raftBatch 
'''
