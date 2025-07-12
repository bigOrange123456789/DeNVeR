import glob
import os
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from skimage import color, data, filters, graph, measure, morphology
import preprocess
import subprocess
import argparse
import time
# 记录开始时间
time_log = {
    "time_pre" : time.time()/60 ,
    "起始时刻":time.time()/60 
}
def saveTime(tag):
    time_log[tag]=time.time()/60
    print(time_log)
    print(tag,time_log[tag]-time_log["time_pre"])
    time_log["time_pre"]=time_log[tag]


ROOT = os.path.abspath("__file__/..") #ROOT: /home/lzc/桌面/DNVR

def skeltoize(path="CVAI-2829RAO9_CRA37",ROOT=""):
    imfiles = sorted(glob.glob(f"{ROOT}/preprocess/datasets/{path}/binary/*"))#获取binary中全部图片的路径
    # glob.glob函数的作用是根据提供的路径模式，返回一个包含所有匹配文件路径的列表。
    os.makedirs(f"{ROOT}/custom_videos/skeltoize/{path}", exist_ok=True)#生成一个文件夹
    for image_file in imfiles:#逐个文件进行读取
        file_namge = image_file.split("/")[-1]#获取文件名
        binary_image_ori = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE) #读取MASK文件
        skeletonized_image = morphology.skeletonize(binary_image_ori)#输入和输出都是(512, 512)的numpy
        '''
            morphology.skeletonize函数:
                将二值图像中的连通区域转换为单像素宽的骨架。它是一种形态学操作，通常用于图像分析和特征提取。
        '''
        binary_image = np.array(skeletonized_image, dtype=np.uint8)
        binary_image = 1 - binary_image #二值MASK图像
        # distance_transform = binary_image*255
        distance_transform = distance_transform_edt(binary_image)#计算每个像素到最近骨架像素的距离。
        distance_transform = np.clip(distance_transform, 0, 65.0)
        distance_transform = 255 - distance_transform
        distance_transform = cv2.normalize(
            distance_transform, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #将其缩放到指定的范围(0到255)，同时将数据类型转换为8位无符号整数（uint8）
        name = os.path.join(f"{ROOT}/custom_videos/skeltoize/{path}", file_namge)
        cv2.imwrite(name, distance_transform)
    print(f"{path} done!")


class Main():
    def __init__(self, config,data_name):
        if not os.path.exists(config["my"]["filePathRoot"]):
            os.mkdir(config["my"]["filePathRoot"])
        self.run(config,data_name)
    def run(self, config,data_name):
        # data_name = config["my"]["videoId"]#args.data
        # data_name: CVAI-2828RAO2_CRA32
        # preprocess
        if True: #需要将下面这个操作替换为FreeCOS方法
            preprocess.filter_extract(data_name,
                                  filePathRoot=config["my"]["filePathRoot"],
                                  inPath=config["my"]["datasetPath"]
                                  )#通过“黑塞矩阵+区域生长”生成MASK，并存入“preprocess/--/binary”
        saveTime("黑塞矩阵+区域生长")
        skeltoize(data_name,ROOT=os.path.join(ROOT, config["my"]["filePathRoot"])) # 获取图片的骨架，并存入custom_videos/skeltoize
        saveTime("获取骨架")
        # exit(0)

        # run raft #RAFT是方法简称
        # print(ROOT)
        # cmd = f"python {ROOT}/scripts/dataset_raft.py  --root ../custom_videos/ --dtype custom --seqs {data_name}"
        # cmd = f"cd scripts && python dataset_raft.py  --root ../custom_videos/ --dtype custom --seqs {data_name}"
        # cmd = f"cd scripts && python dataset_raft.py  --root ../custom_videos/ --dtype custom --seqs {data_name}"
        custom_videos_path = os.path.join("../", config["my"]["filePathRoot"],"custom_videos")
        cmd = f"cd {ROOT}/scripts && python dataset_raft.py --root { custom_videos_path } --dtype custom --seqs {data_name}"
        # cmd = f"python {ROOT}/dataset_raft.py  --root {custom_videos_path} --dtype custom --seqs {data_name}"
        # cmd = f"cd {ROOT} python ./scripts/dataset_raft.py  --root {custom_videos_path} --dtype custom --seqs {data_name}"
        # cmd = f"cd scripts && python dataset_raft.py  --root { custom_videos_path } --dtype custom --seqs {data_name}"
        print(cmd) # cd scripts && python dataset_raft.py  --root ../custom_videos/ --dtype custom --seqs CVAI-2828RAO2_CRA32
        subprocess.call(cmd, shell=True) # 计算光流数据，并存入custom_videos中
        # exit(0)
        saveTime("计算光流")

        # stage 1                     # 获取背景的静态全景图
        nirs_path = os.path.join(ROOT, config["my"]["filePathRoot"],"nirs")
        print("nirs_path:",nirs_path)
        filePathRoot2=os.path.join(ROOT, config["my"]["filePathRoot"])#用于加载数据集
        cmd = f"python {ROOT}/nir/booststrap.py --filePathRoot {filePathRoot2} --data {data_name} --outpath {nirs_path}"
        print(cmd) # python nir/booststrap.py --data CVAI-2828RAO2_CRA32
        subprocess.call(cmd, shell=True) # 计算背景图片，并存入nirs中
        saveTime("NIR前/背景分离")
        # exit(0)
        # stage 2
        cmd = f"python {ROOT}/run_opt.py data=custom data.seq={data_name}"
        print(cmd) # python run_opt.py data=custom data.seq=CVAI-2828RAO11_CRA11
        subprocess.call(cmd, shell=True)
        saveTime("执行完毕")

import yaml
if __name__ == "__main__":
    print("version:2025.06.14 15:46")
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data")
    args = parser.parse_args()
    # 指定 YAML 文件路径
    script_path = os.path.abspath(__file__)
    ROOT1 = os.path.dirname(script_path)
    file_path = os.path.join(ROOT1, './confs/newConfig.yaml')
    # 打开并读取 YAML 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        Main(config,args.data)
    # main(args)
    # main()

'''

pip install pillow==10.2.0 scikit-image==0.22.0 scipy==1.12.0 matplotlib==3.8.3 opencv-python==4.9.0.80 tensorboard==2.16.2 torch==2.2.1 torchvision==0.17.1 tqdm==4.66.2 hydra-core==1.3.2
export PATH="~/anaconda3/bin:$PATH"
source activate DNVR
python main.py -d CVAI-2855LAO26_CRA31
python main.py -d CVAI-2828RAO11_CRA11

获取光流图
cd scripts && python dataset_raft.py  --root ../custom_videos/ --dtype custom --seqs CVAI-2828RAO11_CRA11

获取背景的静态全景图
python nir/booststrap.py --filePathRoot log --data CVAI-2828RAO11_CRA11 --outpath log/nirs

三分支的视频分割框架
python run_opt.py data=custom data.seq=CVAI-2828RAO2_CRA32
python run_opt.py data=custom data.seq=CVAI-2855LAO26_CRA31
python run_opt.py data=custom data.seq=CVAI-2828RAO11_CRA11

'''
