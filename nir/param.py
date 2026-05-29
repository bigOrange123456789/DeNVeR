import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch


from nir.new_batch_topK_lib import Config, ImageLoader, NormalizationCalculator, ModelManager, Evaluator, StatisticalAnalyzer, ResultVisualizer, ResultSaver, denoising

import cv2
import numpy as np
from pathlib import Path      


# import shutil
from nir.paramPy.config_fluid00 import config_fluid00
from nir.paramPy.config_rigid11 import config_rigid11 #双刚体
from nir.paramPy.config_rigid12 import config_rigid12 #单刚体
from nir.paramPy.config_soft00 import config_soft00
from nir.paramPy.config_A26_03 import config_A26_03 #最好的效果
from nir.paramPy.config_A26_03_01 import config_A26_03_01 #和03一样
from nir.paramPy.config_A26_03_01B import config_A26_03_01B #
from nir.paramPy.config_A26_03_01B2 import config_A26_03_01B2
from nir.paramPy.config_A26_03_01C import config_A26_03_01C #没软体            #下降、当相对流体消融下降的很少
from nir.paramPy.config_A26_03_01D import config_A26_03_01D #没软体和流体       #显著下降(流体很重要?)
from nir.paramPy.config_A26_03_01E import config_A26_03_01E #只有宽度减半的刚体    #显著下降
from nir.paramPy.config_A26_03_01F import config_A26_03_01F #用渐进式破坏软体干扰
from nir.paramPy.config_A26_03_01F2 import config_A26_03_01F2 #关闭软体运动
from nir.paramPy.config_A26_03_01G import config_A26_03_01G #自适应遮挡（指标不够高、但对照组有明显下降）
from nir.paramPy.config_A26_03_01H import config_A26_03_01H #自适应遮挡的初值为1
from nir.paramPy.config_A26_03_01I import config_A26_03_01I #自适应遮挡的初值为1+快收敛(指标和G相似)
from nir.paramPy.config_A26_03_01I2 import config_A26_03_01I2 #高迭代数         # myLastMethod
from nir.paramPy.config_A26_03_01I2_2 import config_A26_03_01I2_2 #减少血管MASK更新次数
from nir.paramPy.config_A26_03_01I3 import config_A26_03_01I3 #超高迭代数
# from nir.paramPy.config_A26_03_01I4 import config_A26_03_01I4 #高迭代数 + 不确定学习 
from nir.paramPy.config_A26_03_01J import config_A26_03_01J #自适应中加强软体运动
from nir.paramPy.config_A26_03_01J2 import config_A26_03_01J2 #高迭代数         # *
from nir.paramPy.config_A26_03_01J3 import config_A26_03_01J3 #高迭代数+大batch
# from nir.paramPy.config_A26_03_01K import config_A26_03_01K #不确定学习
from nir.paramPy.config_A26_03_01L import config_A26_03_01L #rv:Log=>MSE
# from nir.paramPy.config_A26_03_01M import config_A26_03_01M #不确定: {S、F}
# from nir.paramPy.config_A26_03_01N import config_A26_03_01N #不确定: &sigmoid激活 [失败]
# from nir.paramPy.config_A26_03_01O import config_A26_03_01O #不确定: (便于后续消融分析的版本) [失败]
# from nir.paramPy.config_A26_03_01O1 import config_A26_03_01O1 #不确定: (只作用于刚体)
# from nir.paramPy.config_A26_03_01O2 import config_A26_03_01O2 #不确定: (只作用于软体)
# from nir.paramPy.config_A26_03_01O3 import config_A26_03_01O3 #不确定: (只作用于流体)
# from nir.paramPy.config_A26_03_01O4 import config_A26_03_01O4 #不确定: (add混合法)
# from nir.paramPy.config_A26_03_01O5 import config_A26_03_01O5 #不确定: (只作用于流体)(MSE_noUL)
# from nir.paramPy.config_A26_03_01P import config_A26_03_01P # 复现J2        # myLastMethod
# from nir.paramPy.config_A26_03_01P1 import config_A26_03_01P1 # 刚体:微弱的不确定性
# # from nir.paramPy.config_A26_03_01P2 import config_A26_03_01P2 # follow P (使用输入数据本身进行归一化)
# # from nir.paramPy.config_A26_03_01P3 import config_A26_03_01P3 # follow P1(使用输入数据本身进行归一化)
# from nir.paramPy.config_A26_03_01P4 import config_A26_03_01P4 # 不确定: 刚体:微弱的不确定性(纠正后)
# from nir.paramPy.config_A26_03_01P5 import config_A26_03_01P5 # 减少训练数据
# from nir.paramPy.config_A26_03_01P6 import config_A26_03_01P6 # 全部使用myLog
# from nir.paramPy.config_A26_03_01P7 import config_A26_03_01P7 # myLog_UL刚体
# from nir.paramPy.config_A26_03_01P8 import config_A26_03_01P8 # 关闭软体PE
# from nir.paramPy.config_A26_03_01P9 import config_A26_03_01P9 # MSE_UL刚体(sigmoid2)
# from nir.paramPy.config_A26_03_01P10 import config_A26_03_01P10 # MSE_UL刚体(square)
# from nir.paramPy.config_A26_03_01P11 import config_A26_03_01P11 # myLogSquare_noUL流体
# from nir.paramPy.config_A26_03_01P12 import config_A26_03_01P12 # 复现01P
# from nir.paramPy.config_A26_03_01P13 import config_A26_03_01P13 # 不确定: 前后帧变化快的地方不确定
# from nir.paramPy.config_A26_03_01P14 import config_A26_03_01P14 # 不确定: 加强不确定效果
from nir.paramPy.config_A26_03_01Q import config_A26_03_01Q #降低刚体运动的总复杂度
from nir.paramPy.config_A26_03_01Q1 import config_A26_03_01Q1 #流体层只关注血管
from nir.paramPy.config_A26_03_01Q2 import config_A26_03_01Q2 #完全复现01I2（myLastMethod）
from nir.paramPy.config_A26_03_01Q3 import config_A26_03_01Q3 #流体网络宽度加倍

# “特征向量遮挡”模块的消融测试
from nir.paramPy.config_A26_03_02 import config_A26_03_02 #去除渐进式特征
from nir.paramPy.config_A26_03_02B import config_A26_03_02B
from nir.paramPy.config_A26_03_02F import config_A26_03_02F
from nir.paramPy.config_A26_03_02G import config_A26_03_02G #去除自适应特征

# # “联合重构约束”模块的消融测试(这里是否有效不重要)
# from nir.paramPy.config_A26_03_03G import config_A26_03_03G

# # “不确定学习”模块的消融测试
# from nir.paramPy.config_A26_03_04N import config_A26_03_04N # rv_eps=0
# from nir.paramPy.config_A26_03_04N1 import config_A26_03_04N1 # rv_eps=0.5

from nir.paramPy.config_test import config_test
c0 = config_A26_03_01Q3

# 目前最重要的是获取论文所需的量化结果
if True:
# def main(): 

    # 初始化配置和各个管理器
    config = Config()
    model_manager = ModelManager()
    image_loader = ImageLoader(config.dataset_path, model_manager.transform)
    # norm_calculator = NormalizationCalculator(config.dataset_path, image_loader)
    #os.makedirs(save_masks_dir, exist_ok=True)
    
    # 设置参数
    threshold = 0.5
    block_cath = True    
    
    # ***--零、一些测试--*** ////基于刚体提取//// 
    # config_sim

    # ***--一、提取刚体--***
    # 1.1 双层刚体 CVAI-1257RAO0_CRA1
    '''
        --test4 : 效果较差
        --test5 : 使用了渐进式流体、效果略微提升
        --test6 : 只使用刚体、不使用流体
        --test7 : 流体只拟合血管区域
        --test8 : 流体使用位置编码，刚体效果相当不错
        --test9 : 软体拟合背景（软体没有拟合出任何有效信息）
        --test10: 背景损失函数 MSE=>myLog
        --test11: 两个刚体层
    '''    
    # config_rigid11,

    # 1.2 单层刚体 CVAI-2855RAO4_CRA35
    '''
        --test12: 单个刚体层 # 最终采用了12号运动#流体层进行渐进式学习 #有一个较好的效果 #测试了刚体是否运动对整体指标的影响
        --test13: 增加优化次数 epochs 0.625=>1.0
        --test14: 刚体纹理网络的深度增加 2=>3 ❌️
        --test15: 刚体运动网络的宽度减半 2*128 => 128 (对抖动没有明显影响)
        --test16: 刚体关注全部信息=>只关注背景信息
        --test17: 更改流体对应的损失函数：MSE=>myLog
        --test18: 流体层不进行渐进式遮挡
        --test19: 尝试优化流体效果❌️ （流体效果不错、但刚体效果很差）
        --test20: 再次尝试优化流体效果（流体效果虽然好，但是有波纹、这个波纹会和刚体相互影响造成瑕疵）
        --test21: 流体模块进行渐进式遮挡 流体效果不错
        --test22: 流体网络的宽度减半、并启用位置编码❌️ 流体拟合失败
        --test23: 关闭流体拟合 ❌️ 刚体效果很差
        --test24: 两个刚体层（流体的缺乏有效信息）
        --test25: 模糊辅助刚体，增加主刚体的深度、关闭流体位置编码 流体深度减半(主刚-辅刚-流体 3-2-2)❌️ 流体信息过多
                      似乎深度增加会降低模型的学习欲望
        --test26: 主刚-辅刚-流体 3-2-2 => 2-3-4 {效果：流体包含了较多的纹理}
        --test27: 主刚-辅刚-流体 2-3-4 => 3-3-8  流体宽度减半❌️较深的网络无法获取刚体信息
        --test28: 主刚-辅刚-流体 2-2-4
                       较高的层数容易出现螺纹，或许位置编码可以缓解这一现象{❌️❌️运动不正确}
        --test29: 增加主刚的宽度 {结果：有变化、但看不出质量的提升}
        --test30: 流体只学习血管 {刚体上有噪点}
        --test31: 降低流体宽度❌️❌️运动不正确
        --test32: 主刚运动网络的宽度加倍
        --test33: 主刚运动网络的宽度加倍、batch减半
        --test34: 增加主刚纹理信息
    ''' 
    # config_rigid12
    
    # ***--二、提取流体--***
    # config_fluid00,

    # ***--三、提取软体--***
    # config_soft00

    # ***--四、最佳效果--***
    # c0 = config_A26_03_01P1_1
    print("config-name:",c0["name"])
    os.makedirs('config_zip', exist_ok=True)
    os.system("zip -r config_zip/"+c0["name"]+".zip nir") #脚本存档
    
    # 定义多个配置
    configs = [#在短视频数据上的测试结果
        c0
    ]#不要重复做实验，要相信之前的结果
    #分析一下batch

    
    for c in configs:
        if not "normalization" in c:
            c["normalization"] = "orig"
        # print({"n":norm_calculator.calculate_mean_variance})
        # print("c[norm_method]",c["norm_method"])
        norm_calculator = NormalizationCalculator(
            config.dataset_path, 
            image_loader,
            None if c["normalization"] == "orig" else c["input_mode"] # 使用原始数据还是输入数据来计算归一化的均值和方差
            )
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

