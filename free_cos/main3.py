import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# 假设您的模型定义在某个模块中，请根据实际情况导入
# from your_model_file import ModelSegment

class SegmentationDataset(Dataset):
    """自定义分割数据集，假设图像和标签均为灰度PNG文件，文件名一一对应"""
    def __init__(self, img_dir, gt_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_names = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])
        # 确保标签文件存在（简单检查第一个文件）
        if len(self.img_names) == 0:
            raise RuntimeError(f"No PNG images found in {img_dir}")
        # sample_gt = os.path.join(gt_dir, self.img_names[0])
        # if not os.path.exists(sample_gt):
        #     raise FileNotFoundError(f"Ground truth file {sample_gt} not found. Ensure filenames match.")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        gt_path = os.path.join(self.gt_dir, img_name)

        image = Image.open(img_path).convert('L')   # 灰度图
        mask = Image.open(gt_path).convert('L')     # 标签图

        if self.transform:
            image = self.transform(image)
        else:
            # 默认转为Tensor，数值范围[0,1]
            image = transforms.ToTensor()(image)

        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            # 标签也转为Tensor，并保持为0~1（假设标签值为0或255）
            mask = transforms.ToTensor()(mask)
            # 将255的像素映射为1（二分类）
            mask = (mask > 0.5).float()

        return image, mask

class SegmentationDatasetNoGT(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        # self.gt_dir = gt_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_names = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        # 确保标签文件存在（简单检查第一个文件）
        if len(self.img_names) == 0:
            raise RuntimeError(f"No PNG images found in {img_dir}")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        # gt_path = os.path.join(self.gt_dir, img_name)

        image = Image.open(img_path).convert('L')   # 灰度图
        # mask = Image.open(gt_path).convert('L')     # 标签图

        if self.transform:
            image = self.transform(image)
        else:
            # 默认转为Tensor，数值范围[0,1]
            image = transforms.ToTensor()(image)

        # if self.target_transform:
        #     mask = self.target_transform(mask)
        # else:
        #     # 标签也转为Tensor，并保持为0~1（假设标签值为0或255）
        #     mask = transforms.ToTensor()(mask)
        #     # 将255的像素映射为1（二分类）
        #     mask = (mask > 0.5).float()

        return image#, mask
    



import numpy as np
import data
import utils
import torch.nn.functional as F
import subprocess
import torch
from concurrent import futures
script_path = os.path.abspath(__file__)
ROOT = os.path.dirname(script_path)
# def runRAFT(rgb_dir="",out_dir="",out_img_dir="",gap=1):
#     path0 = os.path.join(ROOT, "../scripts")
#     batch_size = 8
#     if not os.path.exists(out_dir): os.makedirs(out_dir)
#     if not os.path.exists(out_img_dir): os.makedirs(out_img_dir)
#     cmd = f"cd {path0} && python run_raft.py {rgb_dir} {out_dir} -I {out_img_dir} --gap {gap} -b {batch_size}"
#     print("cmd:",cmd)
#     subprocess.call(cmd, shell=True)
#     print("finished!")
def initValLoader(rgb_dir, batch_size=1, gt_dir="temp_gt", fwd_dir="temp_fwd", bck_dir="temp_fwd", mask_dir="temp_fwd"):
    # 一、生成分割图
    def segImg(rgb_dir, gt_dir, mask_dir):
        test_img_dir=rgb_dir
        test_gt_dir=gt_dir
        test_dataset = SegmentationDatasetNoGT(test_img_dir, #test_gt_dir, 
                                               transform=transform, target_transform=target_transform)
        test_loader = DataLoader(test_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
        finetuner.seg(test_loader, output_dir=mask_dir)
    segImg(rgb_dir, gt_dir, mask_dir )
    # 二、生成光流图
    # if True:
    #     fwd_img_dir=fwd_dir+"Img"
    #     bck_img_dir=bck_dir+"Img"
    #     runRAFT(rgb_dir=rgb_dir, out_dir=fwd_dir, out_img_dir=fwd_img_dir, gap=+1) #计算正向光流图
    #     runRAFT(rgb_dir=rgb_dir, out_dir=bck_dir, out_img_dir=bck_img_dir, gap=-1) #runRAFT(rgb_dir, bck_dir, bck_img_dir, -1) #计算逆向光流图
    #     exit(0)


    # print("rgb_dir:", rgb_dir)
    rgb_dset = data.RGBDataset(rgb_dir)
    fwd_dset = data.FlowDataset(fwd_dir, +1, rgb_dset=rgb_dset)  # 前向光流
    bck_dset = data.FlowDataset(bck_dir, -1, rgb_dset=rgb_dset)  # 后向光流
    occ_dset = data.OcclusionDataset(fwd_dset, bck_dset)     # 遮挡(行进顺序-前后)
    disocc_dset = data.OcclusionDataset(bck_dset, fwd_dset)  # 遮挡(行进顺序-后前)
    epi_dset = data.RGBDataset(mask_dir)  # 黑塞矩阵MASK

    # print({
    #     "rgb_dir":rgb_dir,
    #     "fwd_dir":fwd_dir,
    #     "bck_dir":bck_dir,
    #     "mask_dir":mask_dir
    # })
    # print({
    #     "rgb": rgb_dset,  # custom_videos/PNGImages #原视频
    #     "fwd": fwd_dset,  # custom_videos/raw_flows_gap1  #正向光流
    #     "bck": bck_dset,  # custom_videos/raw_flows_gap-1 #逆向光流
    #     "epi": epi_dset,  # preprocess/--/binary  #黑塞MASK
    #     "occ": occ_dset,  # 遮挡(行进顺序-前后)
    #     "disocc": disocc_dset
    # })
    dset = data.CompositeDataset({
        "rgb": rgb_dset,  # custom_videos/PNGImages #原视频
        "fwd": fwd_dset,  # custom_videos/raw_flows_gap1  #正向光流
        "bck": bck_dset,  # custom_videos/raw_flows_gap-1 #逆向光流
        "epi": epi_dset,  # preprocess/--/binary  #黑塞MASK
        "occ": occ_dset,  # 遮挡(行进顺序-前后)
        "disocc": disocc_dset
    })
    length=len(fwd_dset)
    val_loader = data.get_ordered_loader(  # 我猜这里是有序加载
        dset,
        batch_size,#length,#batch_size
        preloaded=True
    )
    return val_loader 


import torch.backends.cudnn as cudnn
from free_cos.ModelSegment import ModelSegment
def getModel(pathParam):
    os.environ['MASTER_PORT'] = '169711' #“master_port”的意思是主端口
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    cudnn.benchmark = True #benchmark的意思是基准


    n_channels = 1
    num_classes =  1
    Segment_model = ModelSegment(n_channels, num_classes)

    # 有1个cuda 。torch.cuda.device_count()=1
    if torch.cuda.is_available():
        # print("cuda_is available")
        Segment_model = Segment_model.cuda() # 分割模型


    ##############################   predictor.lastInference()   ##############################
    checkpoint = torch.load(pathParam, map_location=torch.device('cpu'))  # 如果模型是在GPU上训练的，这里指定为'cpu'以确保兼容性
    Segment_model.load_state_dict(checkpoint['state_dict'])  # 提取模型状态字典并赋值给模型

    return Segment_model

def calculate_mean_variance(image_folder):
    # 初始化变量
    total_pixels = 0
    sum_pixels = 0.0
    sum_squared_pixels = 0.0

    # 获取文件夹中所有图片文件
    image_files = [f for f in os.listdir(image_folder) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not image_files:
        print("文件夹中没有找到图片文件")
        return None, None

    # 遍历所有图片
    for img_file in image_files:
        try:
            img_path = os.path.join(image_folder, img_file)
            img = Image.open(img_path).convert('L')  # 确保是灰度图
            img_array = np.array(img).astype(np.float32)/255.0
            # print("img_array",np.max(img_array),np.min(img_array))
            # exit(0)

            # 更新统计量
            num_pixels = img_array.size
            total_pixels += num_pixels
            sum_pixels += np.sum(img_array)
            sum_squared_pixels += np.sum(img_array.astype(np.float64) ** 2)

        except Exception as e:
            print(f"处理图片 {img_file} 时出错: {e}")
            continue

    if total_pixels == 0:
        print("没有有效的像素数据")
        return None, None

    # 计算均值和方差
    mean = sum_pixels / total_pixels
    variance = (sum_squared_pixels / total_pixels) - (mean ** 2)

    return mean, variance**0.5

def getDice(
    img_dir= "../DeNVeR_in/xca_dataset/CVAI-1207/images/CVAI-1207LAO44_CRA29",
    gt_dir = "../DeNVeR_in/xca_dataset/CVAI-1207/ground_truth/CVAI-1207LAO44_CRA29",
    fwd_dir="../DeNVeR_in/custom_videos/raw_flows_gap1/CVAI-1207LAO44_CRA29" ,
    bck_dir="../DeNVeR_in/custom_videos/raw_flows_gap-1/CVAI-1207LAO44_CRA29",
    pathParam = "../DeNVeR_in/models_config/freecos_Seg.pt",
    batch_size = 4 #10 #14 #16
    ):

    # 定义数据预处理（与推理代码保持一致，计算均值和标准差）
    # 注意：您可以使用 calculate_mean_variance 函数计算训练集的均值和标准差
    # 此处简单使用默认的ToTensor()，即归一化到[0,1]
    mean, std = calculate_mean_variance(img_dir)
    # print("{mean:",mean, "std:", std, "}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        # 如果需要标准化，可以添加 transforms.Normalize(mean, std)
    ])
    target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        lambda x: (x > 0.5).float()  # 将标签转换为0/1
    ])

    # 创建数据集和数据加载器
    val_dataset = SegmentationDataset(img_dir, gt_dir, transform=transform, target_transform=target_transform)  # 这里用测试集作为验证集示例，实际应划分验证集
    val_loader =  DataLoader(val_dataset,    batch_size=batch_size, shuffle=False, num_workers=2)

    # 初始化微调器
    # 假设 ModelSegment 已在某处定义，此处用占位
    # from your_model import ModelSegment
    # 如果您没有该类的定义，需要先导入或定义。这里假设存在。
    segment_model = getModel(pathParam)
    
    # 使用占位模型演示，实际应替换为您的 ModelSegment
    finetuner = FineTuner(segment_model)

    # 测试 # 79%
    finetuner.test3(val_loader, output_dir="inference_new2") #finetuner.test(test_loader)

    
    
    
    
from free_cos.FineTuner import FineTuner
# ========== 使用示例 ==========
if __name__ == "__main__":
    datasetPath="../DeNVeR_in/xca_dataset"
    for userId in os.listdir(datasetPath):
        for videoId in os.listdir(os.path.join(datasetPath,userId,"images")):
            print(videoId)
    
