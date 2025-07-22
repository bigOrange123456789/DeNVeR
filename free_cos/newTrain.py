import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from free_cos.ModelSegment import ModelSegment
import cv2

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import numpy as np

import random
import matplotlib.pyplot as plt
import time
import math

##############################################################################################################
from skimage import measure
class ConnectivityAnalyzer:
    def __init__(self,mask_tensor):
        self.mask_tensor=mask_tensor
        self.allObj=torch.where(mask_tensor > 0.5, torch.ones_like(mask_tensor),
                                  torch.zeros_like(mask_tensor))
        self.mainObj=self.getMainObj(self.allObj)
    def getMainObj(self, mask_tensor): #这里的输入是0/1标签
        mask_tensor=mask_tensor.cpu() # 我猜.转换到CPU当中之后就不会计算梯度了
        # mask_tensor = torch.where(mask_tensor > 0.5, torch.ones_like(mask_tensor),
        #                              torch.zeros_like(mask_tensor))

        # 将PyTorch张量转换为NumPy数组，保持单通道维度
        mask_array = mask_tensor.numpy().astype(np.uint8)

        # 创建一个空列表来存储处理后的MASK，保持与输入相同的shape
        processed_masks = []

        # print(mask_tensor.shape, mask_array.shape[0],type(mask_array.shape[0]))
        # exit(0)
        # entropyList=[]
        entropy = 0 #所有图片的连通熵
        # 遍历每张MASK图片（保持单通道维度）
        i = 0
        for mask in mask_array: #对每个批次中的图片逐个进行处理
            # 挤压掉单通道维度以进行连通性检测，但之后要恢复
            mask_squeeze = mask.squeeze()
            if mask_squeeze.sum()==0:#这个对象为空
                processed_masks.append(mask)
                i=i+1
                continue

            # 进行连通性检测，返回标记数组和连通区域的数量
            labeled_array, num_features = measure.label(mask_squeeze, connectivity=1, return_num=True)
            entropy0=self.__computeEntropy(
                labeled_array,
                num_features,
                self.mask_tensor[i,0,:,:]
            )
            i = i+1
            entropy+=entropy0
            # entropyList.append(entropy0)

            # 创建一个字典来存储每个标签的像素数
            region_sizes = {}
            for region in range(1, num_features + 1):
                # 计算每个连通区域的像素数
                region_size = np.sum(labeled_array == region)
                region_sizes[region] = region_size

            # 找到像素数最多的连通区域及其标签
            max_region = max(region_sizes, key=region_sizes.get)

            # 创建一个新的MASK，只保留像素数最多的连通区域，并恢复单通道维度
            processed_mask = np.zeros_like(mask)
            processed_mask[0, labeled_array == max_region] = 1

            # 将处理后的MASK添加到列表中
            processed_masks.append(processed_mask)


        # 将处理后的MASK列表转换回PyTorch张量
        processed_masks_tensor = torch.tensor(processed_masks, dtype=torch.float32)

        # 检查shape是否保持不变
        assert processed_masks_tensor.shape == mask_tensor.shape, "Processed masks tensor shape does not match original."

        if torch.cuda.is_available():# 检查CUDA是否可用
            device = torch.device("cuda")  # 创建一个表示GPU的设备对象
        else:
            device = torch.device("cpu")  # 如果没有GPU，则使用CPU

        # self.entropyList=torch.tensor(entropyList).to(device)
        self.entropy = entropy/mask_array.shape[0] # 每个图片信息熵的均值

        return processed_masks_tensor.to(device)
    def __computeEntropy(self, labeled_array,NUM,img_score): #单张图片的连通熵
        '''
        labeled_array, #标出了连通区域
        num,           #连通区域个数
        img_score,     #每个像素的打分
        img_vessel     #血管的mask图片
        '''
        score_all = img_score[img_score > 0.5].sum() #总分数
        entropy_all = 0
        if score_all!=0:
            for region_id in range(1, NUM + 1):
                # 计算每个连通区域的像素数
                # region_size = np.sum(labeled_array == region_id)
                # 创建一个与标签图像尺寸相同的布尔数组，并初始化为False
                # mask = np.zeros_like(img_score, dtype=bool)
                score_region = img_score[labeled_array == region_id].sum()
                # print("score_region",score_region)
                # print("(img_score * mask)",(img_score * mask).shape)
                # print("score_all+epsilon",score_all+epsilon)
                # exit(0)
                if score_region!=0:
                    p = score_region / score_all #区域的概率
                    entropy_region = -p * torch.log(p)
                    entropy_all = entropy_all + entropy_region
        return entropy_all
# loss_conn2 = ConnectivityAnalyzer(pred_target).connectivityLoss('entropy')  # 无/伪监督
# loss_conn2 = ConnectivityAnalyzer(pred_target).entropy
##############################################################################################################


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target): #预测值、目标值
        # predict:[4,1,256,256]
        assert predict.size() == target.size(), "the size of predict and target must be equal."

        num = predict.size(0) # 4

        pre = predict.view(num, -1) # [ 4 , 256 * 256 ]
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score

class MyDataset(Dataset):
    def add_noise(self,image_tensor):
        cMax = 0.60 #0.6
        cMin = 0.40
        cMean = 0.50
        cStd = 0.10  # 0.05#0.22

        if False: #不使用椒盐噪声
            # 创建随机噪声掩码
            prob = 0.01
            H, W = image_tensor.shape[1], image_tensor.shape[2]

            # 盐噪声掩码 (白色像素)
            salt_mask = torch.rand(H, W) < prob
            image_tensor[0, salt_mask] = cMax  # 假设图像值在[0,1]范围

            # 椒噪声掩码 (黑色像素)
            pepper_mask = torch.rand(H, W) < prob
            image_tensor[0, pepper_mask] = cMin

        """
        # 高斯噪声生成函数 - 针对Tensor格式
        给单通道Tensor图像添加高斯噪声
        参数:
            image_tensor: 形状为 [1, H, W] 的Tensor
            mean: 噪声均值 (通常为0)
            sigma_range: 噪声标准差范围 (min_sigma, max_sigma)
        返回:
            添加噪声后的Tensor
        """
        mean = cMean
        sigma_range = (cStd, cStd)

        # 克隆原始Tensor以避免修改原始数据
        noisy_tensor = image_tensor.clone()

        # 随机选择当前噪声的标准差
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        # 生成与图像相同形状的高斯噪声
        noise = torch.randn_like(noisy_tensor) * sigma + mean

        # 添加噪声并裁剪到[0,1]范围
        noisy_tensor = (noisy_tensor+noise)/2
        noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)

        return noisy_tensor

    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform

        # 获取配对的文件名（假设同名文件是配对数据）
        self.noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*")))
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, "*")))

        # 验证文件配对
        assert len(self.noisy_files) == len(self.clean_files), "文件数量不匹配"
        for nf, cf in zip(self.noisy_files, self.clean_files):
            assert os.path.basename(nf) == os.path.basename(cf), f"文件不匹配: {nf} vs {cf}"

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):

        # 加载图像对
        noisy_img = Image.open(self.noisy_files[idx])
        clean_img = Image.open(self.clean_files[idx])

        # 转换为RGB（处理灰度图）
        noisy_img = noisy_img.convert('L') # ('RGB')
        clean_img = clean_img.convert('L') # ('RGB')
        # print("noisy_img", type(noisy_img))
        # exit(0)

        # 应用变换
        img = self.transform(noisy_img)
        clean_img = self.transform(clean_img)

        img = img*2

        # 添加高斯噪声
        img = self.add_noise(img) #sigma_range=(0.01, 0.2)
        img = img * 0.5
        # print("img", img.shape, img.min(), img.max(),img.mean(), img.std())

        # noisy_img = (noisy_img - torch.mean(noisy_img)) / torch.std(noisy_img)
        return img, clean_img

class MyDatasetUnsup(Dataset):
    def __init__(self, imgs_dir, transform=None):
        self.imgs_dir = imgs_dir
        self.transform = transform
        self.imgs_files = sorted(glob.glob(os.path.join(imgs_dir, "*"))) # 获取配对的文件名（假设同名文件是配对数据）

    def __len__(self):
        return len(self.imgs_files)

    def __getitem__(self, idx):
        img = Image.open(self.imgs_files[idx]) # 加载图像
        img = img.convert('L') # ('RGB') # 转换为RGB（处理灰度图）
        return self.transform(img) # 应用变换

from tqdm import tqdm
def train_model(model, train_loader,train_loaderUnsup, criterion, optimizer, device, num_epochs=10):
    model.train()
    unsupervised_dataloader = iter(train_loaderUnsup)  # 无监督的数据加载器
    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(total=len(train_loader)) as pbar:
         for i, (imgs, clean) in enumerate(train_loader):
            imgs = imgs.to(device)[:,[0]]
            gts = clean.to(device)[:,[0]]
            if False:
                imgs = (imgs - torch.mean(imgs)) / torch.std(imgs)
            if False:
                while True: # 获取一个batch的无监督数据
                    try: # 这段代码尝试从unsupervised_dataloader迭代器中获取下一个数据批次。
                        unsup_minibatch = next(unsupervised_dataloader)
                    except StopIteration: # 如果迭代器已经耗尽（引发了StopIteration异常）
                        unsupervised_dataloader = iter(train_loaderUnsup) # 则重新初始化迭代器
                        unsup_minibatch = next(unsupervised_dataloader) # 再次尝试获取下一个数据批次
                    if not torch.isnan(torch.sum(unsup_minibatch)): break
                loss_conn = ConnectivityAnalyzer(unsup_minibatch).entropy#连通熵一直为0?
                print("loss_conn:",loss_conn)

            # 前向传播
            outputs = model(imgs)["pred"]
            with torch.no_grad():  # 禁用梯度计算
                weight_mask = gts.clone().detach()
                weight_mask[weight_mask == 0] = 0.05  # 0.1  # 值为0的元素设为0.1
                weight_mask[weight_mask > 0] = 1  # 1  # 值为1的元素保持不变
                criterion_bce = nn.BCELoss(weight=weight_mask)
            gts[gts <  0.25] = 0 #背景
            gts[gts >= 0.25] = 1 #血管、导管
            loss = criterion(outputs, gts) + 3.0*criterion_bce(outputs, gts) #测试

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.update(1)  # 每次增加 1
         pbar.set_postfix(loss=f"{running_loss:.4f}", epoch=f"{epoch + 1}/{num_epochs}")

    print('训练完成!')
    return model

def start(pathParam,pathIn,pathOut):
    os.makedirs(pathOut, exist_ok=True)
    os.environ['MASTER_PORT'] = '169711' #“master_port”的意思是主端口
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    cudnn.benchmark = True #benchmark的意思是基准


    n_channels = 1
    num_classes =  1
    Segment_model = ModelSegment(n_channels, num_classes)

    # 有1个cuda 。torch.cuda.device_count()=1
    if torch.cuda.is_available():
        print("cuda_is available")
        Segment_model = Segment_model.cuda() # 分割模型


    ##############################   predictor.lastInference()   ##############################
    pathParam = pathParam#"./logs/FreeCOS48.log/best_Segment.pt"#os.path.join('logs', "FreeCOS48.log", "best_Segment.pt")
    pathOut = pathOut#"./logs/FreeCOS48.log/inference" #os.path.join('logs', "FreeCOS48.log", "inference")
    pathIn = pathIn#"./DataSet-images/test/img"
    print("pathOut",pathOut)
    print("pathParam",pathParam)
    print("pathIn",pathIn)
    checkpoint = torch.load(pathParam)  # 如果模型是在GPU上训练的，这里指定为'cpu'以确保兼容性
    Segment_model.load_state_dict(checkpoint['state_dict'])  # 提取模型状态字典并赋值给模型

    #########################################################################################################
    # 设置参数
    noisy_dir = "../DeNVeR_in/datasetSysthesis/vessel_3D"  # 噪声图像目录
    clean_dir = "../DeNVeR_in/datasetSysthesis/label_3D"  # 干净图像目录
    batch_size = 1#10#10#11#4#8
    num_epochs = 1#10#20#10#10
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
    ])

    # 创建数据集和数据加载器
    dataset = MyDataset(noisy_dir, clean_dir, transform=transform)
    datasetUnsup = MyDatasetUnsup(pathIn, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_loaderUnsup = DataLoader(datasetUnsup, batch_size=batch_size, shuffle=True, num_workers=4)

    # 初始化模型
    model = Segment_model#DenoiseNet().to(device)

    # 加载预训练权重（如果有）
    # pretrained_path = "path/to/pretrained_model.pth"
    # model.load_state_dict(torch.load(pretrained_path))

    # 定义损失函数和优化器
    criterion = DiceLoss() #nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        train_loaderUnsup=train_loaderUnsup,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs
    )

    # 保存模型
    torch.save(trained_model.state_dict(), os.path.join(pathOut,"freecos_Seg_new.pt"))
    print("模型参数已保存!")
    #########################################################################################################

    import numpy as np
    from PIL import Image
    os.makedirs(pathOut, exist_ok=True)
    os.makedirs(os.path.join(pathOut, "filter"), exist_ok=True)
    os.makedirs(os.path.join(pathOut, "binary"), exist_ok=True)
    if True:
        os.makedirs(os.path.join(pathOut, "connect"), exist_ok=True)
        os.makedirs(os.path.join(pathOut, "connect_maxbox"), exist_ok=True)
    Segment_model.eval()

    # 定义转换流程
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor并自动归一化到[0,1]
    ])
    from free_cos.main import calculate_mean_variance
    mean, var = calculate_mean_variance(pathIn)
    std = var**0.5
    with torch.no_grad():
     for filename in os.listdir(pathIn):# 获取所有PNG文件
        file_path = os.path.join(pathIn, filename)

        img = Image.open(file_path).convert('L')
        img = transform(img)*2
        tensor=(img-mean)/ std#tensor=(img - torch.mean(img)) / torch.std(img) #这对提升最后的效果很有用
        val_imgs = tensor.unsqueeze(0)
        val_imgs = val_imgs.cuda(non_blocking=True)  # NCHW
        result = Segment_model(val_imgs, mask=None, trained=False, fake=False)
        val_pred_sup_l, sample_set_unsup = result["pred"], result["sample_set"]
        val_pred_sup_l = val_pred_sup_l.detach() * 255
        images_np = val_pred_sup_l.cpu().numpy().squeeze(axis=1).astype(np.uint8)
        # print("1images_np", images_np.shape)
        images_np = images_np[0]

        # 查找连通区域
        binary_image = images_np.copy()
        binary_image[binary_image>0] = 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        # 创建一个与原图同样大小的彩色图像
        colored_image = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)
        # 为每个连通区域分配不同的颜色
        for label in range(1, num_labels):  # 跳过背景（标签0）
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            colored_image[labels == label] = color
        # 保存标注后的图像
        cv2.imwrite(os.path.join(pathOut, "connect", filename), colored_image)

        # 初始化最大包围框面积和对应标签
        max_bbox_area = -1
        max_label = -1
        # 跳过背景(0)，从1开始遍历
        for label in range(1, num_labels):
            # stats结构: [x0, y0, width, height, area]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            bbox_area = w * h  # 计算包围框面积
            if bbox_area > max_bbox_area:# 更新最大区域
                max_bbox_area = bbox_area
                max_label = label
        # 创建只包含最大连通区域的图像
        result_image = np.zeros_like(binary_image)
        if max_label != -1:  # 确保找到有效区域
            result_image[labels == max_label] = 255
        # 保存结果
        cv2.imwrite(os.path.join(pathOut, "connect_maxbox", filename), result_image)
        result_image[result_image>0]=1
        maxRegion = result_image

        # images_np = images_np*maxRegion
        image = Image.fromarray(images_np.copy(), mode='L')
        image.save(os.path.join(pathOut, "filter", filename))
        img2=images_np.copy()
        img2[img2>=255*0.5]=255
        img2[img2<255*0.5]=0
        image2 = Image.fromarray(img2, mode='L')
        image2.save(os.path.join(pathOut, "binary", filename))

import argparse
if __name__ == '__main__':
    print("01:初次测试", "只能分割无噪合成图、无法正确分割真实图")
    print("02:添加高斯噪声","训练的时候没有除以均值、测试的时候除以均值", "能够正确分割、但是无法去除根部")
    print("03:添加了DICE损失函数、训练时不除以标准差", "视觉上没有显著变化")
    print("04:添加了椒盐噪声(不进行血管/导管区分)", "查全率大大下降")#椒盐噪声起了消极作用
    print("05:不使用椒盐噪声(不进行血管/导管区分)", "查全率依旧非常低")#除以方差后效果很好
    print("06:训练和测试的时候都除以标准差", "分割不出任何东西")
    print("07:训练的时候不除以标准差", "")

    parser = argparse.ArgumentParser()
    parser.add_argument("--pathParam")
    parser.add_argument("--pathIn")
    parser.add_argument("--pathOut")
    args = parser.parse_args()
    # train01(args.pathParam,args.pathIn,args.pathOut)
    start("../DeNVeR_in/models_config/freecos_Seg.pt",
            "./free_cos/data/in",
            "./free_cos/data/out07")

'''
    export PATH="~/anaconda3/bin:$PATH"
    source activate FreeCOS
    python -m free_cos.newTrain

'''
