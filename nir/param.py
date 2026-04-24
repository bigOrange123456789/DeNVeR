import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch


from nir.new_batch_topK_lib import Config, ImageLoader, NormalizationCalculator, ModelManager, Evaluator, StatisticalAnalyzer, ResultVisualizer, ResultSaver, denoising

import cv2
import numpy as np
from pathlib import Path     
import os   


import os
# import shutil
from nir.paramPy.config_fluid00 import config_fluid00
from nir.paramPy.config_rigid11 import config_rigid11
from nir.paramPy.config_rigid12 import config_rigid12
from nir.paramPy.config_sim import config_sim
from nir.paramPy.config_soft00 import config_soft00

if True:
# def main(): 

    # 初始化配置和各个管理器
    config = Config()
    model_manager = ModelManager()
    image_loader = ImageLoader(config.dataset_path, model_manager.transform)
    norm_calculator = NormalizationCalculator(config.dataset_path, image_loader)
    #os.makedirs(save_masks_dir, exist_ok=True)
    
    # 设置参数
    threshold = 0.5
    block_cath = True    
    
    # 定义多个配置
    configs = [#在短视频数据上的测试结果
        
        # ***--sim--*** ////基于刚体提取//// 
        # config_sim

        # ***--一、提取刚体--***
        # 1.1 双层刚体 CVAI-1257RAO0_CRA1
        # --test4 : 效果较差
        # --test5 : 使用了渐进式流体、效果略微提升
        # --test6 : 只使用刚体、不使用流体
        # --test7 : 流体只拟合血管区域
        # --test8 : 流体使用位置编码，刚体效果相当不错
        # --test9 : 软体拟合背景（软体没有拟合出任何有效信息）
        # --test10: 背景损失函数 MSE=>myLog
        # --test11: 两个刚体层
        # config_rigid11,

        # 1.2 单层刚体 CVAI-2855RAO4_CRA35
        config_rigid12, # --test12: 单个刚体层 # 最终采用了12号运动#流体层进行渐进式学习 #有一个较好的效果
        # --test13: 增加优化次数 epochs 0.625=>1.0
        # --test14: 刚体纹理网络的深度增加 2=>3 ❌️
        # --test15: 刚体运动网络的宽度减半 2*128 => 128 (对抖动没有明显影响)
        # --test16: 刚体关注全部信息=>只关注背景信息
        # --test17: 更改流体对应的损失函数：MSE=>myLog
        # --test18: 流体层不进行渐进式遮挡
        # --test19: 尝试优化流体效果❌️ （流体效果不错、但刚体效果很差）
        # --test20: 再次尝试优化流体效果（流体效果虽然好，但是有波纹、这个波纹会和刚体相互影响造成瑕疵）
        # --test21: 流体模块进行渐进式遮挡 流体效果不错
        # --test22: 流体网络的宽度减半、并启用位置编码❌️ 流体拟合失败
        # --test23: 关闭流体拟合 ❌️ 刚体效果很差
        # --test24: 两个刚体层（流体的缺乏有效信息）
        # --test25: 模糊辅助刚体，增加主刚体的深度、关闭流体位置编码 流体深度减半(主刚-辅刚-流体 3-2-2)❌️ 流体信息过多
        #               似乎深度增加会降低模型的学习欲望
        # --test26: 主刚-辅刚-流体 3-2-2 => 2-3-4 {效果：流体包含了较多的纹理}
        # --test27: 主刚-辅刚-流体 2-3-4 => 3-3-8  流体宽度减半❌️较深的网络无法获取刚体信息
        # --test28: 主刚-辅刚-流体 2-2-4
        #                较高的层数容易出现螺纹，或许位置编码可以缓解这一现象{❌️❌️运动不正确}
        # --test29: 增加主刚的宽度 {结果：有变化、但看不出质量的提升}
        # --test30: 流体只学习血管 {刚体上有噪点}
        # --test31: 降低流体宽度❌️❌️运动不正确
        # --test32: 主刚运动网络的宽度加倍
        # --test33: 主刚运动网络的宽度加倍、batch减半
        # --test34: 增加主刚纹理信息

        # ***--二、提取流体--***
        # config_fluid00,

        # ***--三、提取软体--***
        # config_soft00

    ]#不要重复做实验，要相信之前的结果
    #分析一下batch

    for c in configs:
        print({"n":norm_calculator.calculate_mean_variance})
        print("c[norm_method]",c["norm_method"])
        c["norm_method"] = norm_calculator.calculate_mean_variance
        c["binarize"] = False
        c["inferenceAll"] = False
        c["mergeMask"] = False
    '''
    需要验证的小点:
        刚体重构损失约束 MSE=>mylog
        流体层拟合血管的时候 背景也添加微弱的约束
        提高concise损失的权重
        对concise损失进行加权
        对运动也进行PE编码
    还没有实现的功能：
        像素语义特征
        实时推理的MASK: epoch>800的时候,每隔200epoch更新一次MASK
    要添加的负影响功能（通过约束进行现在）
        刚体整体运动
        软体局部运动
    想要测试的功能：
        将激活函数替换为RELU   
    '''




