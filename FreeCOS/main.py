import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


from lib.ModelSegment import ModelSegment
import cv2
def main(args):
    os.environ['MASTER_PORT'] = '169711' #“master_port”的意思是主端口
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # args = parser.parse_args()
    cudnn.benchmark = True #benchmark的意思是基准


    n_channels = 1
    num_classes =  1
    Segment_model = ModelSegment(n_channels, num_classes)

    # 有1个cuda 。torch.cuda.device_count()=1
    if torch.cuda.is_available():
        print("cuda_is available")
        Segment_model = Segment_model.cuda() # 分割模型


    ##############################   predictor.lastInference()   ##############################
    pathParam = args.pathParam#"./logs/FreeCOS48.log/best_Segment.pt"#os.path.join('logs', "FreeCOS48.log", "best_Segment.pt")
    pathOut = args.pathOut#"./logs/FreeCOS48.log/inference" #os.path.join('logs', "FreeCOS48.log", "inference")
    pathIn = args.pathIn#"./DataSet-images/test/img"
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
    os.makedirs(os.path.join(pathOut, "connect"), exist_ok=True)
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
        image = Image.fromarray(images_np.copy(), mode='L')
        # image.save(os.path.join(pathOut, filename))
        image.save(os.path.join(pathOut, "filter", filename))
        img2=images_np.copy()
        img2[img2>0]=255
        image2 = Image.fromarray(img2, mode='L')
        image2.save(os.path.join(pathOut, "binary", filename))

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

        # 跳过背景（标签0），从第1个开始找最大面积
        max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        max_area = stats[max_label, cv2.CC_STAT_AREA]
        # 创建一个全黑的图像，只保留最大连通区域
        max_region_mask = (labels == max_label).astype(np.uint8) * 255
        # 保存最大连通区域的掩码图像
        cv2.imwrite(os.path.join(pathOut, "max_region", filename), max_region_mask)



import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathParam")
    parser.add_argument("--pathIn")
    parser.add_argument("--pathOut")
    args = parser.parse_args()
    main(args)

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
    
'''
