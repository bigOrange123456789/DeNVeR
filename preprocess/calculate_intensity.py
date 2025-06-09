import os
from PIL import Image
import numpy as np


def calculate_intensity(image_path):#计算一张图片所有像素的亮度值之和
    img = Image.open(image_path)
    img = img.convert('L')
    img = list(img.getdata())#len(img)=262144=512*512
    intensity = sum(img)

    return intensity

def get_thresholds(folder_path):
    intensity_values = []
    image_names = []

    for filename in sorted(os.listdir(folder_path)):#遍历文件夹中的全部文件
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            intensity = calculate_intensity(image_path)#计算像素值之和
            intensity_values.append(intensity)#记录图片亮度
            image_names.append(filename)#记录文件名

    min_intensity = min(intensity_values)
    max_intensity = max(intensity_values)
    maximum_position = intensity_values.index(max_intensity)
    normalized_intensity = [(intensity - min_intensity) / (max_intensity - min_intensity) * (99.9 - 92) + 92 for intensity in intensity_values]
    # 归一化后的值会被映射到一个新的范围（92到99.9）#为啥要大于92？
    normalized_intensity = np.array(normalized_intensity)

    return 191.9 - normalized_intensity, maximum_position #191.9-x: 92->99.9, 99.9->92


def find_cut_position(folder_path):
    thresholds, maximum_position = get_thresholds(folder_path)#衡量每张图片亮度的相对强弱，获取最亮图片的索引。
    print("The index of image that Frangi/Sato filter favors:", maximum_position) #Frangi/Sato滤镜偏好的图像索引

    cut_position = len(thresholds)-1
    for i in range(maximum_position + 1, len(thresholds)):
        if thresholds[i] >= 94.0:
            cut_position = i
            # print("The position that we cut the dataset in half:", cut_position)
            break
    cut_position = len(thresholds)-1
    return thresholds, cut_position, maximum_position #每张图的相对暗度，最后一张图的索引，最亮的图的索引