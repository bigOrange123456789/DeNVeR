import cv2
import numpy as np
import os
import argparse
from pathlib import Path

def load_images_from_folder(folder_path):
    """从文件夹加载图像并按数字顺序排序"""
    images = []
    filenames = []
    
    # 获取所有png文件并按数字顺序排序
    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.png'):
            filenames.append(file)
    
    # 按数字顺序排序
    filenames.sort(key=lambda x: int(x.split('.')[0]))
    
    for filename in filenames:
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
        else:
            print(f"警告: 无法读取图像 {img_path}")
    
    return images, filenames

def perform_dsa(mask_image, contrast_images):
    """执行DSA处理"""
    dsa_results = []
    
    # 将掩模图像转换为浮点数以便进行减法运算
    mask_float = mask_image.astype(np.float32)/255
    
    for contrast_img in contrast_images:
        # 将对比图像转换为浮点数
        contrast_float = contrast_img.astype(np.float32)/255
        
        # 执行减法: 对比图像 - 掩模图像
        # subtracted = contrast_float - mask_float
        subtracted = contrast_float / mask_float
        
        # 处理负值（将负值设为0）
        subtracted[subtracted < 0] = 0
        
        # # 归一化到0-255范围
        # if np.max(subtracted) > 0:
        #     normalized = (subtracted / np.max(subtracted)) * 255
        # else:
        #     normalized = subtracted
        normalized = subtracted
        # subtracted = subtracted/255
        # normalized = 255 - subtracted
        normalized = 255*normalized/2
        # normalized = 255*normalized
        
        # 转换为8位无符号整数
        result = normalized.astype(np.uint8)
        
        dsa_results.append(result)
    
    return dsa_results

def clean_dir(path: str, recursive: bool = True):
    import os, shutil, sys
    if not os.path.isdir(path):
        print(f"目录不存在：{path}")
        sys.exit(1)

    for entry in os.listdir(path):
        full = os.path.join(path, entry)
        if os.path.isfile(full) or os.path.islink(full):
            os.remove(full)
        elif recursive and os.path.isdir(full):
            shutil.rmtree(full)

def save_dsa_results(dsa_results, filenames, save_path):
    """保存DSA结果图像"""
    # 创建保存目录（如果不存在）
    os.makedirs(save_path, exist_ok=True)
    
    for i, (result, filename) in enumerate(zip(dsa_results, filenames)):
        # if i == 0:  # 跳过掩模图像
        #     continue
        
        # 生成输出文件名
        output_filename = f"{filename}" #f"dsa_{filename}"
        output_path = os.path.join(save_path, output_filename)
        
        # 保存图像
        cv2.imwrite(output_path, result)
        print(f"已保存: {output_path}")

def main(inPath, savePath):
    """主函数"""
    print(f"输入路径: {inPath}")
    print(f"保存路径: {savePath}")
    
    # 加载图像
    print("正在加载图像...")
    images, filenames = load_images_from_folder(inPath)
    
    if len(images) < 2:
        print("错误: 需要至少2张图像（掩模图像和至少一张对比图像）")
        return
    
    print(f"成功加载 {len(images)} 张图像")
    
    # 提取掩模图像（第一张）和对比图像（其余所有）
    mask_image = images[0]
    contrast_images = images#[1:]
    
    print(f"使用 {filenames[0]} 作为掩模图像")
    print(f"处理 {len(contrast_images)} 张对比图像")
    
    # 执行DSA处理
    print("正在执行DSA处理...")
    dsa_results = perform_dsa(mask_image, contrast_images)
    
    # 保存结果
    print("正在保存结果...")
    save_dsa_results(dsa_results, 
                     filenames,#[1:], 
                     savePath)
    
    print("DSA处理完成！")


from tqdm import tqdm
if __name__ == "__main__":
    # # 设置命令行参数解析
    # parser = argparse.ArgumentParser(description='数字减影血管造影(DSA)处理')
    # parser.add_argument('inPath', type=str, help='输入图像文件夹路径')
    # parser.add_argument('savePath', type=str, help='结果保存路径')
    
    # args = parser.parse_args()
    
    # # 检查输入路径是否存在
    # if not os.path.exists(args.inPath):
    #     print(f"错误: 输入路径 {args.inPath} 不存在")
    #     exit(1)
    
    # main(args.inPath, args.savePath)
    # inPath = "../DeNVeR_in/xca_dataset_sub1/CVAI-1207/"
    # main(args.inPath, args.savePath)

    datasetPath = "../DeNVeR_in/xca_dataset_sub1"
    patient_names = [name for name in os.listdir(datasetPath)
                 if os.path.isdir(os.path.join(datasetPath, name))]
    CountSum=0
    for patientID in patient_names:
        patient_path = os.path.join(datasetPath, patientID, "images")
        video_names = [name for name in os.listdir(patient_path)
                     if os.path.isdir(os.path.join(patient_path, name))]
        for videoId in video_names:
            CountSum=CountSum+1
    with tqdm(total=CountSum) as pbar:
     for patientID in patient_names:
        patient_path = os.path.join(datasetPath, patientID, "images")
        video_names = [name for name in os.listdir(patient_path)
                     if os.path.isdir(os.path.join(patient_path, name))]
        for videoId in video_names:
            pbar.set_postfix(videoId=f"{videoId}")
            pbar.update(1)  # 每次增加 1
            if len(videoId.split("CATH"))==1:
                inPath = os.path.join(datasetPath, patientID, "images", videoId)
                savePath = os.path.join(datasetPath, patientID, "decouple", videoId,"0.TDSA")
                # clean_dir(savePath)
                main(inPath, savePath)
        # exit(0)


