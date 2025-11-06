import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF

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
        if True:#进行造影剂浓度的扰动 # light = 1 * np.exp(-1 * self.thickness * delta)
            delta0 = 0.03  # random.uniform(0.027, 0.036)
            thickness = torch.log(image_tensor)/(-1*delta0)
            # delta2 = torch.rand(1) * (0.035 - 0.020) + 0.020  # 进行数据扰动
            delta2 = torch.rand(1) * (0.040 - 0.030) + 0.030  # 进行数据扰动
            # delta2 = 0.06
            image_tensor = 1 * torch.exp(-1 * thickness * delta2)
        cMax = 0.60 #0.6
        cMin = 0.40
        cMean = 0.50 #这里实际上应该是1
        cStd = 0.10  # 0.05#0.22

        if True: #使用椒盐噪声
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
        # 确保输入是单通道Tensor
        assert image_tensor.dim() == 3 and image_tensor.shape[0] == 1, "输入应为单通道Tensor [1, H, W]"

        if torch.rand(1) > 0.5: #有1/10的概率使用真噪声
            noise = self.getVideoHead()
        else:
            # 随机选择当前噪声的标准差
            sigma = random.uniform(sigma_range[0], sigma_range[1])
            # 生成与图像相同形状的高斯噪声
            noise = (torch.randn_like(image_tensor) * sigma + mean)/2

        # 添加噪声并裁剪到[0,1]范围
        if False:
            image_tensor = image_tensor/2+ noise
            image_tensor = torch.clamp(image_tensor, 0.0, 1.0)#*0.5
        else:
            noise = torch.clamp(noise, 0.0, 1.0)  # *0.5
            image_tensor = noise * image_tensor
        rand_tensor = torch.rand(1) * (1.50 - 0.25) + 0.25 #进行数据扰动
        image_tensor = image_tensor * rand_tensor
        # print("img",image_tensor.shape,image_tensor[0,0,0:50])
        # exit(0)

        return image_tensor
    def getVideoHead(self):
        from pathlib import Path
        path0 = "../DeNVeR_in/firstImg"

        folder = Path(path0)  # ← 换成你的目录
        # 1. 仅保留文件，过滤掉子目录
        files = [p for p in folder.iterdir() if p.is_file()]


        import random
        fileName = random.choice(files)
        # print(fileName)
        # print(self.videoPath, fileName)
        img = Image.open(fileName)

        if torch.rand(1) > 0.5:
            if torch.rand(1) > 0.5:
                img = img.rotate(90, expand=False)  # 逆时针 90°
            else:
                img = img.rotate(180, expand=False)  # 逆时针 180°
        elif torch.rand(1) > 0.5:
            img = img.rotate(-90, expand=False)  # 顺时针 90°

        img = self.transform(img.convert('L'))

        if torch.rand(1)>0.5: img = TF.hflip(img) # 水平翻转
        if torch.rand(1)>0.5: img = TF.vflip(img) # 垂直翻转

        return img

    def getVideoHeadOld(self):
        from pathlib import Path

        folder = Path(self.videoPath)  # ← 换成你的目录
        # 1. 仅保留文件，过滤掉子目录
        files = [p for p in folder.iterdir() if p.is_file()]
        # 2. 按“数字文件名”排序（忽略扩展名，也能处理 1.txt, 2.png 等）
        files.sort(key=lambda p: int(p.stem))  # p.stem 去掉扩展名后的纯文件名

        # 3. 取前 5 个文件名（不含路径）
        first_five = [f.name for f in files[:5]]

        import random
        fileName = random.choice(first_five)
        # print(self.videoPath, fileName)
        img = Image.open(os.path.join(self.videoPath,fileName))

        if torch.rand(1) > 0.5:
            if torch.rand(1) > 0.5:
                img = img.rotate(90, expand=False)  # 逆时针 90°
            else:
                img = img.rotate(180, expand=False)  # 逆时针 180°
        elif torch.rand(1) > 0.5:
            img = img.rotate(-90, expand=False)  # 顺时针 90°

        img = self.transform(img.convert('L'))

        if torch.rand(1)>0.5: img = TF.hflip(img) # 水平翻转
        if torch.rand(1)>0.5: img = TF.vflip(img) # 垂直翻转

        return img

    def __init__(self, noisy_dir, clean_dir, videoPath, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.videoPath = videoPath #用来获取前几帧作为背景
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

        # img = img*2

        # 添加高斯噪声
        img = self.add_noise(img) #sigma_range=(0.01, 0.2)
        # img = img * 0.5
        # print("img", img.shape, img.min(), img.max(),img.mean(), img.std())

        # noisy_img = (noisy_img - torch.mean(noisy_img)) / torch.std(noisy_img)
        return img, clean_img

from tqdm import tqdm
def train_model(model, train_loader, criterion, optimizer, device, pathOut , num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        test_i=0
        os.makedirs(os.path.join(pathOut,"train_epoch"+str(epoch)), exist_ok=True)
        running_loss = 0.0
        with tqdm(total=len(train_loader)) as pbar:
         for i, (imgs, clean) in enumerate(train_loader):
            imgs = imgs.to(device)[:,[0]]
            gts = clean.to(device)[:,[0]]
            # print("imgs:",imgs.shape)
            # from Test import Test
            # Test(0).show_images(imgs[:,0])
            # exit(0)

            # 前向传播
            imgs[gts<0.25]=0 #后面要测试一下这里为1是否结果会更好
            outputs = model(imgs)["pred"]
            '''
            with torch.no_grad():  # 禁用梯度计算
                weight_mask = gts.clone().detach()
                weight_mask[gts == 0] = 0.05#0.05  # 0.1  # 值为0的元素设为0.1
                weight_mask[gts > 0] = 1  # 1  # 值为1的元素保持不变
                criterion_bce = nn.BCELoss(weight=weight_mask)
            '''
            with torch.no_grad():  # 禁用梯度计算
                weight_mask = gts.clone().detach()
                weight_mask[gts >= 0.75] = 0.05  # 血管
                weight_mask[(gts >= 0.25) & (gts < 0.75)] = 1  # 导管
                weight_mask[gts < 0.25] = 0  # 背景
                criterion_bce = nn.BCELoss(weight=weight_mask)
                # mask = torch.ones_like(gts)
                # mask[gts < 0.25]  = 0  #背景不进行监督
            outputs[gts < 0.25]  = 0 # 背景
            gts[gts < 0.25] = 0 # 背景
            gts[gts >= 0.75] = 0  # 血管
            gts[(gts >= 0.25) & (gts < 0.75)] = 1 #导管
            ###########################################################
            # from Test
            ###########################################################
            '''
            gts[gts <  0.25] = 0 #背景
            gts[gts >= 0.75] = 1 #血管
            gts[(gts >= 0.25) & (gts < 0.75)] = 0 #导管被视为背景
            '''

            loss = criterion(outputs, gts) + 3.0*criterion_bce(outputs, gts)

            for i in range(1):#(len(outputs)):
                Image.fromarray((outputs[i,0] * 255).cpu().byte().numpy(), mode='L').save(
                    os.path.join(pathOut,"train_epoch"+str(epoch),str(test_i)+".jpg"))
                Image.fromarray((gts[i,0] * 255).cpu().byte().numpy(), mode='L').save(
                    os.path.join(pathOut, "train_epoch" + str(epoch), str(test_i) + "o.jpg"))
                Image.fromarray((imgs[i, 0]/(imgs[i, 0].max()+10**-10) * 255).cpu().byte().numpy(), mode='L').save(
                    os.path.join(pathOut, "train_epoch" + str(epoch), str(test_i) + "i.jpg"))
                test_i = test_i + 1

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
    # checkpoint = torch.load(os.path.join(pathOut,"freecos_Seg_new.pt"))#checkpoint = torch.load(pathParam)  # 如果模型是在GPU上训练的，这里指定为'cpu'以确保兼容性
    # Segment_model.load_state_dict(checkpoint)  # 提取模型状态字典并赋值给模型
    checkpoint = torch.load(pathParam)  # 如果模型是在GPU上训练的，这里指定为'cpu'以确保兼容性
    Segment_model.load_state_dict(checkpoint['state_dict'])  # 提取模型状态字典并赋值给模型
    # evaluate(pathOut, Segment_model,"orig")

    #########################################################################################################
    # 设置参数
    noisy_dir = "../DeNVeR_in/datasetSysthesis/vessel_3D_2"  # 噪声图像目录
    clean_dir = "../DeNVeR_in/datasetSysthesis/label_3D_2"  # 干净图像目录
    batch_size = 10#10#10#10#10#10#10#10#11#4#8
    num_epochs = 1#10#10#20#10#10
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
    ])

    # 创建数据集和数据加载器
    dataset = MyDataset(noisy_dir, clean_dir,videoPath=pathIn, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 初始化模型
    model = Segment_model#DenoiseNet().to(device)

    # 定义损失函数和优化器
    criterion = DiceLoss() #nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        pathOut=pathOut
    )

    if False:# 保存模型
        #['state_dict']
        # torch.save(trained_model.state_dict(), os.path.join(pathOut,"freecos_Seg_new.pt"))
        torch.save({"state_dict":trained_model.state_dict(),}, os.path.join(pathOut, "freecos_Seg_new.pt"))
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

    evaluate(pathOut,Segment_model)
    # 定义转换流程
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor并自动归一化到[0,1]
    ])
    from free_cos.main import calculate_mean_variance
    mean, std = calculate_mean_variance(pathIn)
    #std=std**0.5
    with torch.no_grad():
     for filename in os.listdir(pathIn):# 获取所有PNG文件
        file_path = os.path.join(pathIn, filename)

        img = Image.open(file_path).convert('L')
        img = transform(img)#*2
        tensor=img#(img-mean)/ std#tensor=(img - torch.mean(img)) / torch.std(img)
        val_imgs = tensor.unsqueeze(0)
        val_imgs = val_imgs.cuda(non_blocking=True)  # NCHW
        # result = Segment_model(val_imgs, mask=None, trained=False, fake=False)
        result = model(val_imgs)
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
        # maxRegion = result_image

        # images_np = images_np*maxRegion
        image = Image.fromarray(images_np.copy(), mode='L')
        image.save(os.path.join(pathOut, "filter", filename))
        img2=images_np.copy()
        img2[img2>=255*0.5]=255
        img2[img2<255*0.5]=0
        image2 = Image.fromarray(img2, mode='L')
        image2.save(os.path.join(pathOut, "binary", filename))


from nir.myLib.mySave import check
from free_cos.main import mainFreeCOS

def getIndicators(binary_image, ground_truth):  # 对于指标来说预测结果和真值不对称
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
                                                  false_positive + true_negative + false_negative + 1e-10)
    # 计算召回率
    recall = true_positive / (true_positive + false_negative + 1e-10)
    # 计算精确率
    precision = true_positive / (true_positive + false_positive + 1e-10)
    # 计算F1分数
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    # 计算IoU
    iou = true_positive / (true_positive + false_positive + false_negative + 1e-10)
    specificity = true_negative / (true_negative + false_positive + 1e-10)

    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1, "iou": iou,
        "specificity": specificity
    }
import csv
def initCSV(path,head): # 写入 CSV 文件
    csv_file_path = path#os.path.join(path, "experiment_results.csv")
    csv_header = head#["binary_mask_flag", "tag", "id", "frameId", "accuracy", "recall", "precision", "f1", "iou","specificity", "time_gap"]
    if os.path.exists(csv_file_path):  os.remove(csv_file_path)
    with open(csv_file_path, mode="a+", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, csv_header)
        writer.writeheader()  # 写入表头
def save2CVS(path,head,data):
    with open(path, 'a+', newline='', encoding="utf-8") as file:
        csv_writer = csv.writer(file)
        arr = []
        for i in head: arr.append(data[i])
        csv_writer.writerow(arr)
def evaluate(pathOut,model,tag="eval"):
    os.makedirs(os.path.join(pathOut,tag), exist_ok=True)
    head = ["fileName","accuracy","recall","precision","f1","iou","specificity"]
    path = os.path.join(pathOut, "experiment_results.csv")
    initCSV(path, head)
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
    ])
    datasetPath = "../DeNVeR_in/xca_dataset_video"
    sum_recall = 0
    sum_precision = 0
    sum_f1 = 0
    for name in os.listdir(os.path.join(datasetPath,"img")):
        path_img = os.path.join(datasetPath,"img",name)
        img = Image.open( path_img ).convert('L')
        img = transform(img).unsqueeze(0).cuda()
        path_m0 = os.path.join(datasetPath, "A.mask_nr2", name)
        m0 = Image.open(path_m0).convert('L')
        m0 = transform(m0).unsqueeze(0).cuda()
        img[m0<0.5]=0
        pred = model(img)["pred"]

        pred[pred >= 0.5] = 1
        pred[pred <  0.5] = 0
        # output_org=output.clone()
        m0[m0<0.5] = 0
        m0[m0>=0.5] = 1
        output = (1-pred)*m0 #去除m0中的导管
        # output2 = output.clone()
        # output2[m0==1&pred==0] = 0.5
        # output[output >= 0.5] = 1
        # output[output < 0.5] = 0
        path_gt = os.path.join(datasetPath, "gt", name)
        gt = Image.open(path_gt).convert('L')
        gt = transform(gt).unsqueeze(0).cuda()
        ind=getIndicators(
            output[0,0].detach().cpu()*255,
            gt[0,0].detach().cpu()*255
        )
        ind["fileName"]=name.split(".png")[0]
        output2 = output.clone()
        output2[m0 == 1] = 1 #血管和导管
        output2[(m0 == 1) & (pred == 0)] = 0.5 #血管区域
        Image.fromarray(
            (output2[0,0].detach().cpu()*255).numpy().astype(np.uint8), mode='L'
        ).save(os.path.join(pathOut, "eval", name))
        save2CVS(path, head, ind)
        sum_recall += ind["recall"]
        sum_precision += ind["precision"]
        sum_f1 += ind["f1"]
    print("f1:",sum_f1/len(os.listdir(datasetPath+"/img")),
          "pr:",sum_precision/len(os.listdir(datasetPath+"/img")),
          "sn:",sum_recall/len(os.listdir(datasetPath+"/img")))
    print("f1:", sum_f1.item() / len(os.listdir(datasetPath + "/img")),
          "pr:", sum_precision.item() / len(os.listdir(datasetPath + "/img")),
          "sn:", sum_recall.item() / len(os.listdir(datasetPath + "/img")))
    # paramPath = os.path.join(pathOut,"freecos_Seg_new.pt")
    # patient_names = [name for name in os.listdir(datasetPath)
    #              if os.path.isdir(os.path.join(datasetPath, name))]
    ####################################################################
    # if os.path.exists(path0):
    #     mainFreeCOS(paramPath, datasetPath, os.path.join(pathOut,"eval",videoId))
    #     check(os.path.join(pathOut,"eval",videoId), videoId, "newTrain.30.recon_non2")


import argparse
if __name__ == '__main__':
    '''
    print("01:初次测试", "只能分割无噪合成图、无法正确分割真实图")
    print("02:添加高斯噪声","训练的时候没有除以均值、测试的时候除以均值", "能够正确分割、但是无法去除根部")
    print("03:添加了DICE损失函数、训练时不除以标准差", "视觉上没有显著变化")
    print("04:添加了椒盐噪声(不进行血管/导管区分)", "查全率大大下降")#椒盐噪声起了消极作用
    print("05:不使用椒盐噪声(不进行血管/导管区分)", "查全率依旧非常低")#除以方差后效果很好
    print("06:训练和测试的时候都除以标准差", "分割不出任何东西")
    ###########################################################################
    print("07:测试训练0次的效果", "可以分割出正确的效果")
    print("08:测试导管剔除训练[0.1、1、1;0、1、1]=>[0.1、1、1;0、1、0]", "训练完成后没有分割出任何东西","BUG:1没有除标准差;2没有考虑新合成图是白背景")
    print("09:解决了08号实验的BUG", "训练完成后没有分割出任何东西")
    print("10:将导管视为血管", "失败:基本上也是全黑")#不合理、很奇怪(为什么训练后无法分割)
    print("11:训练的时候不除以方差", "失败")
    print("12:回退到5号实验", "")
    print("13:训练数据像素亮度可以大于0.5", "效果非常差")
    print("14:(1)训练数据像素亮度不大于0.5;(2)验证时不除以标准差", "")
    print("15:分析微调过程是否正确", "")
    print("16:训练过程中区分血管和导管", "能够很容易的去除训练集中的导管")
    print("17:对图片的亮度值进行随机扰动", "能够正确识别到血管、但是背景噪声较为严重")
    print("18:(1)背景和血管使用相同的权重;(2)添加椒盐噪声", "背景噪声减少了一点、但仍然非常严重")
    print("19:将添加高斯噪声的方式由相加改为了相乘", "过拟合、识别到的血管区域极少")
    print("20:添加造影剂浓度扰动", "和19的区别不大")
    print("21:使用2号血管", "过拟合减轻了一些")
    print("22:测试造影剂浓度增大后的效果", "过拟合、模型变得破碎、不连贯")
    print("23:添加噪声、而不是乘上噪声", "")
    print("24:有1/10的概率使用真实背景噪声", "失败：只能分割出轮廓")
    print("25:不使用真实背景噪声", "")
    print("26:使用真实背景噪声、但背景处的损失函数为1/20", "失败：还是只能分割出轮廓")
    print("27:提高造影剂浓度", "失败：还是只能分割出轮廓")
    print("28:乘上噪声、而不是添加噪声", "基本正确、但没有去除乳头和导管")
    print("29:有1/2的概率使用真实背景噪声", "失败：查准率较低")
    '''
    # 基于去除刚体：F1:0.764351423 Pr:0.691800375 Sn:0.860701029
    print("30:使用规模为111张图片的噪声集", "成功:查全率和查准率粗看是正确的")
    # f1:0.6433  pr: 0.5364  sn: 0.8438
    print("31:改变思路：接下来只识别导管", "不能对去除刚体的效果进行提升")
    # f1:0.7618  pr: 0.7004 sn: 0.8421
    print("32:将MASK信息输入神经网络中", "")
    # f1: tensor(0.7390) pr: tensor(0.7089) sn: tensor(0.7788)
    print("33:修复训练过程中的一个BUG", "")
    # f1: tensor(0.7689)  pr: tensor(0.7063) sn: tensor(0.8502)
    # f1: 0.7666700544400452 pr: 0.7093585950756505 sn: 0.84076963398791
    # print("34:关注于纯血管的学习、而不是导管", "")

    parser = argparse.ArgumentParser()
    parser.add_argument("--pathParam")
    parser.add_argument("--pathIn")
    parser.add_argument("--pathOut")
    args = parser.parse_args()
    # train01(args.pathParam,args.pathIn,args.pathOut)
    start("../DeNVeR_in/models_config/freecos_Seg.pt",
            "./free_cos/data/in",
            "./free_cos/data/out33")

'''
    export PATH="~/anaconda3/bin:$PATH"
    source activate FreeCOS

'''
