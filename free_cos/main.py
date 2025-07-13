import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


from free_cos.ModelSegment import ModelSegment
import cv2
def mainFreeCOS(pathParam,pathIn,pathOut):
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
    checkpoint = torch.load(pathParam, map_location=torch.device('cpu'))  # 如果模型是在GPU上训练的，这里指定为'cpu'以确保兼容性
    Segment_model.load_state_dict(checkpoint['state_dict'])  # 提取模型状态字典并赋值给模型

    import numpy as np
    from PIL import Image
    os.makedirs(pathOut, exist_ok=True)
    os.makedirs(os.path.join(pathOut, "filter"), exist_ok=True)
    os.makedirs(os.path.join(pathOut, "binary"), exist_ok=True)
    if False:
        os.makedirs(os.path.join(pathOut, "connect"), exist_ok=True)
        os.makedirs(os.path.join(pathOut, "connect_maxbox"), exist_ok=True)
    Segment_model.eval()

    from torchvision import transforms

    # 定义转换流程
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor并自动归一化到[0,1]
    ])
    with torch.no_grad():
     for filename in os.listdir(pathIn):# 获取所有PNG文件
        file_path = os.path.join(pathIn, filename)

        img = Image.open(file_path).convert('L')
        img = transform(img)
        tensor=(img - torch.mean(img)) / torch.std(img)
        val_imgs = tensor.unsqueeze(0)
        val_imgs = val_imgs.cuda(non_blocking=True)  # NCHW
        result = Segment_model(val_imgs, mask=None, trained=False, fake=False)
        val_pred_sup_l, sample_set_unsup = result["pred"], result["sample_set"]
        val_pred_sup_l = val_pred_sup_l.detach() * 255
        images_np = val_pred_sup_l.cpu().numpy().squeeze(axis=1).astype(np.uint8)
        # print("1images_np", images_np.shape)
        images_np = images_np[0]

     # if False:
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

        images_np = images_np*maxRegion
        image = Image.fromarray(images_np.copy(), mode='L')
        image.save(os.path.join(pathOut, "filter", filename))
        img2=images_np.copy()
        img2[img2>=255*0.5]=255
        img2[img2<255*0.5]=0
        image2 = Image.fromarray(img2, mode='L')
        image2.save(os.path.join(pathOut, "binary", filename))



import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathParam")
    parser.add_argument("--pathIn")
    parser.add_argument("--pathOut")
    args = parser.parse_args()
    mainFreeCOS(args.pathParam,args.pathIn,args.pathOut)

'''
    export PATH="~/anaconda3/bin:$PATH"
    source activate FreeCOS
    python train_DA_contrast_liot_finalversion.py 
    python FreeCOS/main.py --pathParam ./logs/FreeCOS48.log/best_Segment.pt --pathIn ./DataSet-images/test/img --pathOut ./pathOut
    python main.py --pathParam ./logs/FreeCOS48.log/best_Segment.pt --pathIn ./DataSet-images/test/img --pathOut ./pathOut
    
    python main.py 
    --pathParam ./logs/FreeCOS48.log/best_Segment.pt 
    --pathIn ./DataSet-images/test/img 
    --pathOut ./pathOut
    
    python main.py 
    --pathParam ../logs/FreeCOS48.log/best_Segment.pt 
    --pathIn ./DataSet-images/test/img 
    --pathOut ./pathOut
    
    python main.py --pathParam ../../DeNVeR_in/models_config/freecos_Seg.pt --pathIn ../../DeNVeR_in/xca_dataset/CVAI-2828/images/CVAI-2828RAO11_CRA11 --pathOut ../log/preprocess/datasets/CVAI-2828RAO11_CRA11
    python ./FreeCOS/main.py --pathParam ../DeNVeR_in/models_config/freecos_Seg.pt --pathIn ../DeNVeR_in/xca_dataset/CVAI-2828/images/CVAI-2828RAO11_CRA11 --pathOut ./log/preprocess/datasets/CVAI-2828RAO11_CRA11


'''
