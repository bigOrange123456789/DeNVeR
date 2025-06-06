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

ROOT = os.path.abspath("__file__/..") #ROOT: /home/lzc/桌面/DNVR

def skeltoize(path="CVAI-2829RAO9_CRA37"):
    imfiles = sorted(glob.glob(f"{ROOT}/preprocess/datasets/{path}/binary/*"))
    os.makedirs(f"{ROOT}/custom_videos/skeltoize/{path}", exist_ok=True)
    for image_file in imfiles:
        file_namge = image_file.split("/")[-1]
        binary_image_ori = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        skeletonized_image = morphology.skeletonize(binary_image_ori)
        binary_image = np.array(skeletonized_image, dtype=np.uint8)
        binary_image = 1 - binary_image
        distance_transform = distance_transform_edt(binary_image)
        distance_transform = np.clip(distance_transform, 0, 65.0)
        distance_transform = 255 - distance_transform
        distance_transform = cv2.normalize(
            distance_transform, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        name = os.path.join(
            f"{ROOT}/custom_videos/skeltoize/{path}", file_namge)
        cv2.imwrite(name, distance_transform)
    print(f"{path} done!")


def main(args):
    data_name = args.data
    # data_name: CVAI-2828RAO2_CRA32
    # preprocess
    preprocess.filter_extract(data_name)
    skeltoize(data_name)

    # run raft

    cmd = f"cd scripts && python dataset_raft.py  --root ../custom_videos/ --dtype custom --seqs {data_name}"
    print(cmd)
    subprocess.call(cmd, shell=True)

    # stage 1
    cmd = f"python nir/booststrap.py --data {data_name}"
    print(cmd)
    subprocess.call(cmd, shell=True)
    # stage 2
    cmd = f"python run_opt.py data=custom data.seq={data_name}"
    print(cmd)
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data")
    args = parser.parse_args()
    main(args)

'''
pip install pillow==10.2.0 scikit-image==0.22.0 scipy==1.12.0 matplotlib==3.8.3 opencv-python==4.9.0.80 tensorboard==2.16.2 torch==2.2.1 torchvision==0.17.1 tqdm==4.66.2 hydra-core==1.3.2
export PATH="~/anaconda3/bin:$PATH"

'''
