
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


def getVideoId(folder_path):
    print("folder_path",folder_path)
    NUMBER_video = 0
    NUMBER_gt = 0
    my_list = []
    for item in os.listdir(folder_path): #遍历所有患者
        if item==".gitkeep":
            continue
        path = folder_path + "/" + item + "/ground_truth"  # /CVAI-1207LAO44_CRA29"
        gts = os.listdir(path)
        for gts0 in gts: #遍历所有视频的标注
            if len(gts0.split("CATH")) == 1: #只看二分类标注的结果
                NUMBER_video = NUMBER_video + 1 #视频数量
                source_path = os.path.join(folder_path + "/" + item + "/images", gts0)
                NUMBER_gt = NUMBER_gt + len(os.listdir(source_path)) #有标注图像数量
                my_list.append(gts0)
    if False:
        print("视频总个数:",NUMBER_video)
        print("有标注视频的总帧数:", NUMBER_gt)
    return my_list

script_path = os.path.abspath(__file__)
ROOT1 = os.path.dirname(script_path)
file_path = os.path.join(ROOT1, '../confs/newConfig.yaml')
class SavePath():
    def __init__(self, path,eval_data,PredictRootPath):#输入路径、文件名列表、输出路径
        # self.gts=SavePath.initGTS(file_path)
        self.saveTxt(path,eval_data,PredictRootPath)
    @staticmethod
    def initGTS(file_path):
        gts = [] #self.gts=[]


        # 打开并读取 YAML 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            import yaml
            config = yaml.safe_load(file)
            eval_data = getVideoId(os.path.join(ROOT1,"../",config["my"]["datasetPath"]))#'job_specs/vessel.txt'
            for i in range(len(eval_data)):  # 遍历读取的列表
                path = config["my"]["datasetPath"]
                newpath = os.path.join(ROOT1,"../",path, eval_data[i][:9], "ground_truth")  # 使用列表文件的前9个字符作为根路径。
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
        self.pathJson=result


##################################################################################3

import cv2
import numpy as np
import json
from PIL import Image
import yaml
import csv
class Evaluate():
    def __init__(self):
        # 定义两个文件夹的路径
        file_path = os.path.join(ROOT1, '../confs/newConfig.yaml')

        with open(file_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
            self.PredictRootPath = self.config['my']["filePathRoot"]
        self.temp={} # 用于暂时存储这个视频的测试结果


    def getPath(self):
        binary_folder = os.path.join(self.PredictRootPath,"gt_path.txt")     #所有人工标签图片的路径
        ground_truth_folder = os.path.join(self.PredictRootPath,"out_path.txt")#所有分割掩码图片的路径
        binary_files = []
        with open(binary_folder, "r") as file:
            for line in file:
                binary_files.append(line.strip())
        ground_truth_files = []
        with open(ground_truth_folder, "r") as file:
            for line in file:
                ground_truth_files.append(line.strip())
        print("binary_files", len(binary_files))
        print("ground_truth_files", len(ground_truth_files))

    def get(self):
        config = self.config
        lines = getVideoId(config["my"]["datasetPath"])
        pathJson = SavePath(config["my"]["datasetPath"], lines, config["my"]["filePathRoot"]).pathJson
        binary_files = []
        ground_truth_files = []
        for key, value in pathJson.items():
            print(key,value)
            binary_files.append(key)
            ground_truth_files.append(value)

        accuracies = []
        recalls = []
        precisions = []
        f1_scores = []
        ious = []
        specificities = []
        # 遍历每一对图像
        for binary_file, gt_file in zip(binary_files, ground_truth_files):
            # 读取二元图像和对应的地面真值（ground truth）图像
            binary_image = cv2.imread(os.path.join(
                binary_file), cv2.IMREAD_GRAYSCALE)
            ground_truth = cv2.imread(os.path.join(
                gt_file), cv2.IMREAD_GRAYSCALE)
            ind=self.getIndicators(binary_image, ground_truth)
            accuracies.append(ind['accuracy'])
            recalls.append(ind['recall'])
            precisions.append(ind['precision'])
            f1_scores.append(ind['f1'])
            ious.append(ind['iou'])
            specificities.append(ind['specificity'])
        # 计算平均值
        average_accuracy = np.mean(accuracies)
        average_recall = np.mean(recalls)
        average_precision = np.mean(precisions)
        average_f1_score = np.mean(f1_scores)
        average_iou = np.mean(ious)
        average_specificity = np.mean(specificities)
        # 计算标准差
        std_accuracy = np.std(accuracies)
        std_recall = np.std(recalls)
        std_precision = np.std(precisions)
        std_f1_score = np.std(f1_scores)
        std_iou = np.std(ious)
        std_specificity = np.std(specificities)
        # 打印结果
        print(f"Accuracy: {average_accuracy:.4f} +- {std_accuracy:.4f}")   #Acc
        print(f"Recall: {average_recall:.4f} +- {std_recall:.4f}")         #Sn
        print(f"Precision: {average_precision:.4f} +- {std_precision:.4f}")#Pr
        print(f"F1 Score: {average_f1_score:.4f} +- {std_f1_score:.4f}")   #Dice
        print(f"IoU: {average_iou:.4f} +- {std_iou:.4f}")                  #JC
        print(f"Specificity: {average_specificity:.4f} +- {std_specificity:.4f}")#Sp

    def getIndicators(self,binary_image, ground_truth): #对于指标来说预测结果和真值不对称
        x, y = binary_image.shape
        g_x, g_y = ground_truth.shape
        if x != g_x and y != g_y:
            binary_image = binary_image.astype(np.uint8)
            binary_image = cv2.resize(binary_image, (g_x, g_y))
        binary_image = np.where(binary_image < 1, 0, 255)
        # 计算真正例、假正例、真负例和假负例的数量
        true_positive = np.logical_and(
            binary_image == 255, ground_truth == 255).sum()
        false_positive = np.logical_and(
            binary_image == 255, ground_truth == 0).sum()
        true_negative = np.logical_and(binary_image == 0, ground_truth == 0).sum()
        false_negative = np.logical_and(
            binary_image == 0, ground_truth == 255).sum()
        # print(binary_file)
        # print(true_positive+false_positive+true_negative+false_negative)
        # 计算精确度
        accuracy = (true_positive + true_negative) / (true_positive +
                                                      false_positive + true_negative + false_negative)
        # 计算召回率
        recall = true_positive / (true_positive + false_negative)
        # 计算精确率
        precision = true_positive / (true_positive + false_positive)
        # 计算F1分数
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        # 计算IoU
        iou = true_positive / (true_positive + false_positive + false_negative)
        specificity = true_negative / (true_negative + false_positive)

        return {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1": f1, "iou": iou,
            "specificity": specificity
        }

    def analysis(self,tag,id,imgs,time_gap):

        def initCSV():
            self.csv_file_path = os.path.join(ROOT1, "../", self.PredictRootPath, "experiment_results.csv")
            self.csv_header=["binary_mask_flag","tag","id","frameId","accuracy","recall","precision","f1","iou","specificity","time_gap"]
            # 写入 CSV 文件
            # with open(self.csv_file_path, mode="w", newline="", encoding="utf-8") as file:
            if not os.path.exists(self.csv_file_path,):#如果不存在这个文件就创建一个
             with open(self.csv_file_path, mode="a+", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file,self.csv_header)
                writer.writeheader()  # 写入表头
        def save2CVS(config0,frameId,time_gap,binary_mask_flag):
            if not hasattr(self, 'csv_header'): initCSV()
            with open(self.csv_file_path, 'a+', newline='', encoding="utf-8") as file:
                csv_writer = csv.writer(file)
                arr=[]
                for i in self.csv_header:
                    if i=="tag":arr.append(tag)
                    elif i=="id":arr.append(id)
                    elif i=="frameId":arr.append(frameId)
                    elif i=="time_gap":arr.append(time_gap)
                    elif i=="binary_mask_flag":arr.append(binary_mask_flag)
                    else: arr.append(config0[i])
                csv_writer.writerow(arr)

        path=os.path.join(ROOT1,"../",self.PredictRootPath,self.config['my']["subPath"]["outputs"],id,tag)
        print(os.path.exists(path))
        if not os.path.exists(path):
            os.makedirs(path)
        #     print("true")
        # print("path:",path)
        images_tensor = imgs.cpu().byte()  # 转换为 [0, 255] 范围的 8 位整数
        # 遍历 tensor 中的每个图像
        for i in range(images_tensor.size(0)):  # size(0) 是 tensor 的第一个维度，即图像数量
            image_tensor = images_tensor[i] # 提取单个图像
            image_array = image_tensor.numpy() # 将 tensor 转换为 NumPy 数组
            # print("image_array",image_array.shape,i)
            image = Image.fromarray(image_array, mode='L')  # 'L' 表示灰度图像
            image.save(os.path.join(path, str(i).zfill(5) + '.png'))
            # image.save(path,os.path.join(str(i).zfill(5)+'.png')) # 保存为 PNG 文件
        gts=SavePath.initGTS(file_path)
        for gt in gts:
            if gt.split("/")[-2]==id:
                frameName=gt.split("/")[-1]
                frameId=int(frameName.split(".")[0])
                ground_truth = cv2.imread(
                    os.path.join(gt),
                    cv2.IMREAD_GRAYSCALE
                )
                image_predict = images_tensor[frameId].numpy()
                ind = self.getIndicators(image_predict, ground_truth)
                save2CVS(ind,frameId,time_gap,False)
                # image_binary = torch.where(image_predict >= 0.5, 1, 0)
                image_binary = (image_predict >= 0.5).astype(int)
                ind2 = self.getIndicators(image_binary, ground_truth)
                save2CVS(ind2, frameId, time_gap, True)


if __name__ == "__main__":
    # 打开并读取 JSON 文件
    # with open( "./log/path.json" , 'r', encoding='utf-8') as file:
    #     pathJson = json.load(file)
    #     Evaluate().get(pathJson)
    Evaluate().get()


