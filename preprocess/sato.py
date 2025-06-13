import os
import numpy as np
from skimage import io, filters
from tqdm import tqdm


def process_images(input_folder, output_folder, border_thickness=3):
    os.makedirs(output_folder, exist_ok=True)#创建文件夹

    image_files = sorted([os.path.join(input_folder, filename) for filename in os.listdir(input_folder)])
    #整理一个路径序列列表(根据文件名/时间排序) #/home/lzc/桌面/DNVR/xca_dataset/CVAI-2828/images/CVAI-2828RAO2_CRA32

    for i, image_path in enumerate(tqdm(image_files, desc="Processing")):
        '''
            enumerate用于将一个可迭代对象（如列表、元组等）组合为一个索引序列，同时列出数据和数据下标。
            tqdm用于在命令行或终端中显示进度条。
            desc="***"输出tqdm的描述文本。
        '''
        file_name = image_path.split("/")[-1]  # 获取文件名
        image = io.imread(image_path, as_gray=True).astype(np.uint8)#读取图片 # type=numpy shape=(512, 512)
        sato_result = filters.sato(image, black_ridges=True, sigmas=range(1, 5), mode="reflect", cval=0)
        ''' 滤波器输入和输出的图像shape相同
            filters模块：包含各种图像滤波和增强功能，其中filters.sato是用于增强图像中管状结构的函数。
            filters.sato函数会增强图像中的管状结构（如血管、纤维等）。
                black_ridges:
                    当black_ridges=True时，函数会增强图像中的暗管状结构（即比周围背景更暗的管状结构）。
                    如果设置为False，则会增强亮管状结构（比周围背景更亮的管状结构）。
                sigmas=range(1, 5):
                    指定了用于多尺度分析的标准差（σ）范围。
                    这里的标准差为1到4之间的整数（即1, 2, 3, 4）
            mode="reflect"：
                "reflect"表示在边界处使用反射模式，即图像边界外的像素值会通过镜像反射的方式进行扩展。
                例如，如果图像的边界像素是[1, 2, 3]，反射模式下边界外的像素会扩展为[1, 2, 3, 2, 1]。
                其他常见的边界处理模式还包括：
                    "constant"：用常数值填充边界外的像素（默认值为0）。
                    "nearest"：用最近的边界像素值填充边界外的像素。
                    "wrap"：将图像边界外的像素值环绕到图像的另一侧。
            cval=0：
                这个参数仅在mode="constant"时生效，表示边界外的像素值用什么常数填充。
                在这里，虽然mode="reflect"，但cval=0仍然被指定，可能是为了代码的通用性，或者是为了避免后续修改时出现意外。
                    如果mode改为"constant"，则边界外的像素值将被填充为0。
        '''
        sato_result = sato_result.astype(np.uint8)

        h, w = sato_result.shape
        for x in range(h):
            for y in range(w):
                if x < border_thickness or x >= h - border_thickness or y < border_thickness or y >= w - border_thickness:
                    sato_result[x, y] = 0 # 将MASK图片的边缘置零：前面滤波过程中对边缘进行了镜像扩充，因此边缘检测可能出错。

        output_filename = file_name#f"{i:05d}.png"
        io.imsave(os.path.join(output_folder, output_filename), sato_result)
        # /home/lzc/桌面/DNVR/preprocess/datasets/CVAI-2828RAO2_CRA32/filter/00000.png

    print("Done.")
