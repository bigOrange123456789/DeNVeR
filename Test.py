import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



class Test():
    def __init__(self,tensor):
        # print()
        # self.save_tensor_as_images(tensor)
        self.show_images(tensor)

    def save_tensor_as_images(self,tensor):  # 显示图片的一个测试脚本
        batch_size, channels, height, width = tensor.shape # 获取张量的形状

        # 遍历每个样本
        for i in range(batch_size):
            img_tensor = tensor[i] # 提取单个样本
            # 确保通道数为 1 或 3
            if channels == 1: # 如果是单通道，直接转换为灰度图像
                img_array = img_tensor[0, :, :].cpu().detach().numpy()  # 去掉通道维度
            elif channels == 3: # 如果是三通道，直接转换为 RGB 图像
                img_array = img_tensor.permute(1, 2, 0).cpu().numpy()  # 调整维度顺序
            else:
                raise ValueError("Unsupported number of channels. Only 1 or 3 channels are supported.")

            # 将张量值缩放到 [0, 1] 范围
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())

            # 显示图像
            plt.figure()
            if channels == 1:
                plt.imshow(img_array, cmap='gray')  # 灰度图像
            else:
                plt.imshow(img_array)  # 彩色图像
            plt.title(f"Image {i + 1}")
            plt.axis('off')  # 关闭坐标轴

        plt.show() # 显示所有图像

    def show_images(self,tensor):  # 显示灰度图片组的一个测试脚本
        batch_size, height, width = tensor.shape # 获取张量的形状

        # 遍历每个样本
        for i in range(batch_size):
            img_tensor = tensor[i] # 提取单个样本
            img_array = img_tensor[:, :].cpu().detach().numpy()  # 去掉通道维度

            # 将张量值缩放到 [0, 1] 范围
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())

            # 显示图像
            plt.figure()
            plt.imshow(img_array, cmap='gray')  # 灰度图像
            plt.title(f"Image {i + 1}")
            plt.axis('off')  # 关闭坐标轴

        plt.show() # 显示所有图像


