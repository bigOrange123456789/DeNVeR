import os
import glob
import numpy as numpy
import argparse

BASE_DIR = os.path.abspath("__file__/..")
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

from main_batch import getVideoId
script_path = os.path.abspath(__file__)
ROOT1 = os.path.dirname(script_path)
file_path = os.path.join(ROOT1, '../confs/newConfig.yaml')
class SavePath():
    def __init__(self, path,eval_data,PredictRootPath):#输入路径、文件名列表、输出路径
        self.gts=SavePath.initGTS(file_path)
        self.saveTxt(path,eval_data,PredictRootPath)
    @staticmethod
    def initGTS(file_path):
        gts = [] #self.gts=[]

        # 打开并读取 YAML 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            import yaml
            config = yaml.safe_load(file)
            eval_data = getVideoId(config["my"]["datasetPath"])#'job_specs/vessel.txt'
            for i in range(len(eval_data)):  # 遍历读取的列表
                path = config["my"]["datasetPath"]
                newpath = os.path.join(path, eval_data[i][:9], "ground_truth")  # 使用列表文件的前9个字符作为根路径。
                data = os.path.join(newpath, eval_data[i])  # 存储了所有人工标注MASK的路径(不包含导管)。
                p = os.path.join(data, "*")
                tmp = sorted(glob.glob(p))  # 加载全部图片
                gts += tmp # self.gts += tmp
        return gts
        # print("self.gts:",self.gts)
        # for i in self.gts:
        #     print(i.split("/")[-1],i.split("/")[-2])


    def saveTxt(self, path,eval_data,PredictRootPath):
        output_txt_path = os.path.join(PredictRootPath,"gt_path.txt")     #所有人工标签图片的路径
        output_predict_path = os.path.join(PredictRootPath,"out_path.txt")#所有分割掩码图片的路径
        output_json_path = os.path.join(PredictRootPath, "path.json")     #所有分割掩码图片的路径
        result={}
        all_data=[]

        for i in range(len(eval_data)):#遍历读取的列表
            newpath = os.path.join(path,eval_data[i][:9],"ground_truth")#使用列表文件的前9个字符作为根路径。
            data = os.path.join(newpath,eval_data[i]) #存储了所有人工标注MASK的路径(不包含导管)。
            all_data.append(data) #如果有两个路径相同怎么办？回答：newpath有时会重复、但data不会重复。
        # print(all_data)
        image_data = []
        for i in all_data:#遍历所有人工标注的文件夹
            p = os.path.join(i,"*")
            # print("p:",p)
            # p="./xca_dataset/CVAI-2828/ground_truth/CVAI-2828RAO2_CRA32/*"
            tmp = sorted(glob.glob(p))#加载全部图片
            print("tmp:",tmp)
            image_data += tmp
        # print("!!!--...由于本地数据不足，本地无法执行代码sorted(glob.glob(p))...--!!!")
        with open(output_txt_path, "w") as file:
            for d in image_data:
                file.write(d + "\n")
        out_data = []
        for image_id in image_data:
            image_name = image_id.split("/")[-1]
            file_name  = image_id.split("/")[-2]
            # print("image_id:",image_id,"file_name:",file_name)
            predict_path = f"{ROOT_DIR}/outputs/dev/custom-{file_name}-gap1-2l"
            predict_path = os.path.join(PredictRootPath, f"outputs/dev/custom-{file_name}-gap1-2l")
            output_name = "{exp_name}"
            predict_path = os.path.join(predict_path,output_name)
            for f in os.listdir(predict_path):
                if f.endswith("val_refine"):
                    p = os.path.join(predict_path,f,"masks_0",image_name)
                    out_data.append(p)
                    result[p]=image_id
        with open(output_predict_path, "w") as file:
            for d in out_data:
                file.write(d + "\n")
        import json # 打开文件并写入 JSON 数据
        with open(output_json_path, 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dir")
    args = parser.parse_args()
    #################################################
    data_path = f"{ROOT_DIR}/job_specs/vessel.txt"  # 所有源视频文件夹的路径
    eval_data = []
    with open(data_path, "r") as file:
        for line in file: # 逐行读取列表文件中的信息
            eval_data.append(line.strip())
    #################################################
    path = f"{ROOT_DIR}/xca_dataset"
    SavePath(path,eval_data,ROOT_DIR)

'''
python gt_filename.py -d init_model
'''
