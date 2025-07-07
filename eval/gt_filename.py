import os
import glob
import numpy as numpy
import argparse

BASE_DIR = os.path.abspath("__file__/..")#当前文件的路径的上一级路径
BASE_DIR="./"
ROOT_DIR = os.path.dirname(BASE_DIR)
'''
    os.path.abspath(__file__): /home/lzc/桌面/DeNVeR/__file__
    BASE_DIR: /home/lzc/桌面/DeNVeR
    ROOT_DIR: /home/lzc/桌面
'''

def get_subfolder_names(path):
    subfolders = []
    for f in os.listdir(path):
        if os.path.isdir(os.path.join(path, f)):
            p = os.path.join(path,f)
            subfolders.append(p)
    return subfolders
def get_subfolder_names_cath(path):
    subfolders = []
    for f in os.listdir(path):
        if os.path.isdir(os.path.join(path, f)) and not os.path.join(path, f).endswith("CATH"):
            p = os.path.join(path,f)
            subfolders.append(p)
    return subfolders
def main(args):
    eval_data = []
    data_path = f"{ROOT_DIR}/job_specs/vessel.txt" #所有源视频文件夹的路径
    # ROOT_DIR: /home/lzc/桌面
    # data_path: /home/lzc/桌面/job_specs/vessel.txt
    with open(data_path, "r") as file:
        for line in file: # 逐行读取列表文件中的信息
            eval_data.append(line.strip())
            # print("line",line)
    # print(eval_data[0][:9])

    path = f"{ROOT_DIR}/xca_dataset"
    output_txt_path = "gt_path.txt"
    output_predict_path = "out_path.txt"
    all_data=[]

    for i in range(len(eval_data)):#遍历读取的列表
        newpath = os.path.join(path,eval_data[i][:9],"ground_truth")#使用列表文件的前9个字符作为根路径。
        data = os.path.join(newpath,eval_data[i]) #存储了所有人工标注MASK的路径(不包含导管)。
        all_data.append(data) #如果有两个路径相同怎么办？回答：newpath有时会重复、但data不会重复。
    # print(all_data)
    image_data = []
    for i in all_data:#遍历所有人工标注的文件夹
        p = os.path.join(i,"*")
        print("p:",p)
        p="./xca_dataset/CVAI-2828/ground_truth/CVAI-2828RAO2_CRA32/*"
        tmp = sorted(glob.glob(p))#加载全部图片
        image_data += tmp
    print("!!!--...由于本地数据不足，本地无法执行代码sorted(glob.glob(p))...--!!!")
    with open(output_txt_path, "w") as file:
        for d in image_data:
            file.write(d + "\n")
    out_data = []
    for image_id in image_data:
        image_name = image_id.split("/")[-1]
        file_name  = image_id.split("/")[-2]
        predict_path = f"{ROOT_DIR}/outputs/dev/custom-{file_name}-gap1-2l"
        output_name = args.dir #输出路径
        print("predict_path,output_name",predict_path,output_name)
        predict_path = os.path.join(predict_path,output_name)
        for f in os.listdir(predict_path):
            if f.endswith("val_refine"):
                p = os.path.join(predict_path,f,"masks_0",image_name)
                out_data.append(p)
    with open(output_predict_path, "w") as file:
        for d in out_data:
            file.write(d + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dir")
    args = parser.parse_args()
    main(args)