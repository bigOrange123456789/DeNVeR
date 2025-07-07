from .calculate_intensity import *
from .sato import *
from .grow_and_connect import *
import os
import shutil
from tqdm import tqdm
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
ROOT = os.path.abspath("__file__/..")
# ROOT应该是文件所在的绝对路径：/home/lzc/桌面/DNVR

def extract_images(input_folder, output_folder, index):
    os.makedirs(output_folder, exist_ok=True) #创建文件夹
    image_files = sorted([os.path.join(input_folder, filename)
                         for filename in os.listdir(input_folder)])
    index = min(index, len(image_files)-1)
    selected_images = image_files[:index + 1]

    for image_path in selected_images:
        shutil.copy(image_path, output_folder) #复制图片


def filter_extract(dir_name="CVAI-2828RAO2_CRA32", base_path="datasets"):
    parent_folder = dir_name[:9]#获取前9个字符：CVAI-2828
    base_path = os.path.join(f"{ROOT}/preprocess", base_path)#/home/lzc/桌面/DNVR/preprocess/datasets
    output_folder_filter = os.path.join(base_path, dir_name, "filter")#filter输出路径
    output_folder_mask = os.path.join(base_path, dir_name, "binary")#binary输出路径
    new_input_folder = os.path.join(base_path, dir_name, dir_name)#输入数据缓存路径
    deforamble_sprite_folder = os.path.join(#扭曲图存储路径
        f"{ROOT}/custom_videos/PNGImages", dir_name)
    input_folder = os.path.join(#原始输入数据路径
        f"{ROOT}/xca_dataset/{parent_folder}/images/{dir_name}")
    # print("output_folder_filter",output_folder_filter)
    process_images(input_folder, output_folder_filter)#获取黑塞矩阵处理后的图像，存入filter文件夹
    thresholds, cut_position, maximum_position = find_cut_position(
        output_folder_filter) #每张图的相对暗度，最后一张图的索引，最亮的图的索引

    extract_images(input_folder, new_input_folder, cut_position) #preprocess/--/CVAI-2828RAO2_CRA32
    extract_images(input_folder, deforamble_sprite_folder, cut_position) #custom_videos/--/CVAI-2828RAO2_CRA32

    filter_images = sorted(os.listdir(output_folder_filter))
    filter_images = filter_images[:cut_position + 1]
    # print(len(filter_images))

    os.makedirs(output_folder_mask, exist_ok=True) #preprocess/--/binary
    for i, filename in tqdm(enumerate(filter_images)): #遍历所有的滤图片 #preprocess/--/filter
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(output_folder_filter, filename)
            image = cv2.imread(image_path, 0) #图片读取
            image_processed = image
            threshold = np.percentile(image, thresholds[i]) #计算这个图片的阈值
            ''' percentile这个单词的含义是“百分位数”
                image：输入的图像数据（多维数组会被自动展平为一维处理）
                thresholds[i]：98.14。指定的百分位数值（0~100之间的标量）
                返回一个阈值数值，使得图像中 thresholds[i]% 的像素值小于或等于该阈值。
                #################
                这行代码的作用是计算图像(或数组)image中指定百分位数对应的阈值。
                返回一个阈值数值，使得图像中 thresholds[i]% 的像素值小于或等于该阈值。
            '''
            # 一、将图中98%的位置设为0
            image_processed = np.where(image < threshold, 0, image) #将滤波图中98%的位置设置为0。
            visited = np.zeros_like(image_processed, dtype=bool)
            connected_regions = []
            intensities = []
            edge_points_each_region = []

            # 二、遍历连通区域
            '''
                最终MASK为：
                    1.主连通区域(A.总分最高)
                    2.总分大于1000
                        平均分大于主区域(B.平均分较高)
                        总分不比主区域小太多
                            靠近主区域(C.靠近主区域)
            '''
            for x in range(image_processed.shape[0]):
                for y in range(image_processed.shape[1]):
                    if image_processed[x, y] > 0 and not visited[x, y]: #如果这个像素的评分不为零且还没有放入MASK中。
                        connected_region, intensity, edge_points = region_grow(
                            image_processed, x, y, visited)#这个连通区域，这个连通区域的总亮度，这个区域的边界
                        connected_regions.append(connected_region)
                        intensities.append(intensity)
                        edge_points_each_region.append(edge_points)

            max_intensity_index = np.argmax(intensities)#找出总亮度最高的连通区域
            for x, y in connected_regions[max_intensity_index]:
                image_processed[x, y] = 255 #将分数最高的连通区域的每个点的分数都设为255
            if len(connected_regions) >= 2: #如果有多个连通区域
                for j, region in enumerate(connected_regions):
                    if j != max_intensity_index and intensities[j] > 1000:#如果新的连通区域分数大于1000
                        if intensities[j] / len(connected_regions[j]) > intensities[max_intensity_index] / len(connected_regions[max_intensity_index]):
                            #如果该区像素的亮度均值比主区域大
                            for x, y in region:
                                image_processed[x, y] = 255
                        elif intensities[j] >= 0.1 * intensities[max_intensity_index]: #如果亮度大于0.1倍的主区亮度
                            min_distance, ptA, ptB = find_closest_points(
                                max_intensity_index, j, edge_points_each_region)
                            '''寻找两个区域的最近距离
                            input:
                                max_intensity_index:主区域索引
                                j:当前区域索引
                                edge_points_each_region：每个区域的边缘
                            output:
                                min_distance:最近距离
                                ptA:区域A上的点
                                ptB:区域B上的点
                            '''
                            # print("image", i, "min distance:", min_distance, ptA, ptB)
                            # print(intensities[max_intensity_index] / len(connected_regions[max_intensity_index]))
                            # print(intensities[j] / len(connected_regions[j]))
                            for x, y in region:#遍历这个区域
                                if min_distance < 100 or intensities[j] / len(connected_regions[j]) > intensities[max_intensity_index] / len(connected_regions[max_intensity_index]):
                                    #如果距离主区域比较近或者比主区域更亮
                                    image_processed[x, y] = 255
                                else:
                                    image_processed[x, y] = 0
                    elif intensities[j] <= 1000:#如果连通区域的总亮度小于1000就将其全部点的分数都设为0
                        for x, y in region:
                            image_processed[x, y] = 0

            output_path = os.path.join(output_folder_mask, filename) # preprocess/--/binary
            cv2.imwrite(output_path, image_processed)
