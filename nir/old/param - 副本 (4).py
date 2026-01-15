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
import shutil


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

        # { #复现最佳的刚体解耦效果2  [follow 018_04 1S1R、静止(没有follow)、不用遮挡拟合]
        #     "decouple":{ # 解耦
        #         "tag":"A21-01-best",#只兼容了startDecouple1
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数
        #             # 纹理
        #             'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #             'hidden_features_map': 256,#64,#8,#2*4*512,#16*4*512,#128,#512, #128,
        #             # 'hidden_features_map': 512, #将隐含层特征维度变为1/8
        #             # 整体运动
        #             'hidden_layers_global':4,
        #             'hidden_features_global':128,
        #             # 局部运动
        #             'hidden_layers_local':4,
        #             'hidden_features_local':128,
        #         }, 
        #         "configRigids":{ # 整个刚体层模块的参数
        #             "loss_recon_all_type":"MSE",#{"myLog" 学习能力不如MSE, "MSE", "atten_d"} #我猜测均方误差更关注背景、注意力损失更关注血管
        #         }, 
        #         "openLocalDeform":False,#True,
        #         "stillness":    False,#True,#False,#不取消运动约束
        #         "stillnessFristLayer":False,#True,#False,#并无意义，要和stillness保持一致
        #         # 1.2 软体模块
        #         "NUM_soft":1,#1,
        #         "configSofts":{ # 软体
        #             "layer":{
        #                 # 纹理
        #                 'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #                 'hidden_features_map': 512, #将隐含层特征维度变为1/8
        #                 # 整体运动
        #                 'hidden_layers_global':4,
        #                 'hidden_features_global':128,
        #                 # 局部运动
        #                 'hidden_layers_local':4,
        #                 'hidden_features_local':128, # Mask遮挡
        #             },
        #             "useSoftMask" : False, #无法生成有意义的MASK
        #             "layerMask":{ #无效
        #                 "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #                 "hidden_layers": 2, 
        #                 "use_residual": False, # 似乎还有负面作用
        #                 "posEnc":{ # 有显著作用
        #                     "num_freqs_pos":10, #3
        #                     "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                     "APE":False,
        #                 }, 
        #                 "gradualImageLayers":False,
        #                 "use_maskP":False,
        #             },
        #         },
        #         # 1.3 流体模块
        #         "NUM_fluid":0, # 0.00019 -> 0.00016、0.00015
        #         "configFluids":{ #参数数量
        #             "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #             "hidden_layers": 2, 
        #             "use_residual": False, # 似乎还有负面作用
        #             "posEnc":{ # 有显著作用
        #                 "num_freqs_pos":10, #3
        #                 "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                 "APE":False,
        #             }, 
        #             "gradualImageLayers":False,
        #             "use_maskP":False,
        #         }, # 现在的首要问题是无损失地拟合出来视频
        #         # 2.损失函数
        #         "useSmooth":False, #不进行平滑约束
        #         "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "lossType":2,
        #         "lossParam":{
        #             "rm":None,#"R,S",
        #             "ra":"R,F",
        #             # "blank":"F",
        #             }, 
        #         "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_021_01", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A21-01-best.rigid",
        #     "input_mode": "A21-01-best.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },

        # { #减少层数-刚体无局部运动(简化后最终指标有不显著地轻微提升)
        #     "decouple":{ # 解耦
        #         "tag":"A21-02",#只兼容了startDecouple1
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数
        #             # 纹理
        #             'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #             'hidden_features_map': 256,#64,#8,#2*4*512,#16*4*512,#128,#512, #128,
        #             # 'hidden_features_map': 512, #将隐含层特征维度变为1/8
        #             # 整体运动
        #             'hidden_layers_global':2,
        #             'hidden_features_global':128,
        #             # 局部运动（刚体层哪来的局部运动）
        #             'hidden_layers_local':2,
        #             'hidden_features_local':128,
        #         }, 
        #         "configRigids":{ # 整个刚体层模块的参数
        #             "useLocal":False, #不使用局部运动
        #             "loss_recon_all_type":"MSE",#{"myLog" 学习能力不如MSE, "MSE", "atten_d"} #我猜测均方误差更关注背景、注意力损失更关注血管
        #         }, 
        #         "openLocalDeform":False, #True,
        #         "stillness":    False, #True,#False,#不取消运动约束
        #         "stillnessFristLayer":False, #True,#False,#并无意义，要和stillness保持一致
        #         # 1.2 软体模块
        #         "NUM_soft":1,#1,
        #         "configSofts":{ # 软体
        #             "layer":{
        #                 # 纹理
        #                 'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #                 'hidden_features_map': 512, #将隐含层特征维度变为1/8
        #                 # 整体运动
        #                 'hidden_layers_global':2,
        #                 'hidden_features_global':128,
        #                 # 局部运动
        #                 'hidden_layers_local':2,
        #                 'hidden_features_local':128, # Mask遮挡
        #             },
        #             "useSoftMask" : False, #无法生成有意义的MASK
        #             "layerMask":{ #无效
        #                 "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #                 "hidden_layers": 2, 
        #                 "use_residual": False, # 似乎还有负面作用
        #                 "posEnc":{ # 有显著作用
        #                     "num_freqs_pos":10, #3
        #                     "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                     "APE":False,
        #                 }, 
        #                 "gradualImageLayers":False,
        #                 "use_maskP":False,
        #             },
        #         },
        #         # 1.3 流体模块
        #         "NUM_fluid":0, # 0.00019 -> 0.00016、0.00015
        #         "configFluids":{ #参数数量
        #             "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #             "hidden_layers": 2, 
        #             "use_residual": False, # 似乎还有负面作用
        #             "posEnc":{ # 有显著作用
        #                 "num_freqs_pos":10, #3
        #                 "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                 "APE":False,
        #             }, 
        #             "gradualImageLayers":False,
        #             "use_maskP":False,
        #         }, # 现在的首要问题是无损失地拟合出来视频
        #         # 2.损失函数
        #         "useSmooth":False, #不进行平滑约束
        #         "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "lossType":2,
        #         "lossParam":{
        #             "rm":None,#"R,S",
        #             "ra":"R,F",
        #             # "blank":"F",
        #             }, 
        #         "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_021_02", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A21-02.rigid",
        #     "input_mode": "A21-02.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },


        # { 
        #     "decouple":{ # 解耦
        #         "tag":"A21-03(20-01)",#企图复现20-01的指标，刚体层静止
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数
        #             # 纹理
        #             'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #             'hidden_features_map': 256,#64,#8,#2*4*512,#16*4*512,#128,#512, #128,
        #             # 'hidden_features_map': 512, #将隐含层特征维度变为1/8
        #             # 整体运动
        #             'hidden_layers_global':2,
        #             'hidden_features_global':128,
        #             # 局部运动（刚体层哪来的局部运动）
        #             'hidden_layers_local':2,
        #             'hidden_features_local':128,
        #         }, 
        #         "configRigids":{ # 整个刚体层模块的参数
        #             "useLocal":False, #不使用局部运动
        #             "loss_recon_all_type":"MSE",#{"myLog" 学习能力不如MSE, "MSE", "atten_d"} #我猜测均方误差更关注背景、注意力损失更关注血管
        #         }, 
        #         "openLocalDeform":False, #True,
        #         "stillness":    True,#False, #True,#False,#不取消运动约束
        #         "stillnessFristLayer":True,#:False, #True,#False,#并无意义，要和stillness保持一致
        #         # 1.2 软体模块
        #         "NUM_soft":1,#1,
        #         "configSofts":{ # 软体
        #             "layer":{
        #                 # 纹理
        #                 'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #                 'hidden_features_map': 512, #将隐含层特征维度变为1/8
        #                 # 整体运动
        #                 'hidden_layers_global':2,
        #                 'hidden_features_global':128,
        #                 # 局部运动
        #                 'hidden_layers_local':2,
        #                 'hidden_features_local':128, # Mask遮挡
        #             },
        #             "useSoftMask" : False, #无法生成有意义的MASK
        #             "layerMask":{ #无效
        #                 "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #                 "hidden_layers": 2, 
        #                 "use_residual": False, # 似乎还有负面作用
        #                 "posEnc":{ # 有显著作用
        #                     "num_freqs_pos":10, #3
        #                     "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                     "APE":False,
        #                 }, 
        #                 "gradualImageLayers":False,
        #                 "use_maskP":False,
        #             },
        #         },
        #         # 1.3 流体模块
        #         "NUM_fluid":0, # 0.00019 -> 0.00016、0.00015
        #         "configFluids":{ #参数数量
        #             "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #             "hidden_layers": 2, 
        #             "use_residual": False, # 似乎还有负面作用
        #             "posEnc":{ # 有显著作用
        #                 "num_freqs_pos":10, #3
        #                 "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                 "APE":False,
        #             }, 
        #             "gradualImageLayers":False,
        #             "use_maskP":False,
        #         }, # 现在的首要问题是无损失地拟合出来视频
        #         # 2.损失函数
        #         "useSmooth":False, #不进行平滑约束
        #         "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "lossType":2,
        #         "lossParam":{
        #             "rm":None,#"R,S",
        #             "ra":"R,F",
        #             # "blank":"F",
        #             }, 
        #         "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "A21-03(20-01)", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A21-03(20-01).rigid",
        #     "input_mode": "A21-03(20-01).rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },

        # { #网络层只使用一个sin激活函数(静态刚体层),效果较差
        #     "decouple":{ # 解耦
        #         "tag":"A21-04",#只兼容了startDecouple1
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数
        #             # 纹理
        #             'hidden_layers_map':0,#2,#1,#2,#4,#32,#4,
        #             'hidden_features_map': 256,#64,#8,#2*4*512,#16*4*512,#128,#512, #128,
        #             # 'hidden_features_map': 512, #将隐含层特征维度变为1/8
        #             # 整体运动
        #             'hidden_layers_global':0,#2,
        #             'hidden_features_global':128,
        #             # 局部运动（在新的代码中useLocal=False时此处的局部运动不会启用）
        #             'hidden_layers_local':0,#2, 
        #             'hidden_features_local':128,
        #         }, 
        #         "configRigids":{ # 整个刚体层模块的参数
        #             "useLocal":False, #不使用局部运动
        #             "loss_recon_all_type":"MSE",#{"myLog" 学习能力不如MSE, "MSE", "atten_d"} #我猜测均方误差更关注背景、注意力损失更关注血管
        #         }, 
        #         "openLocalDeform":False, #True,
        #         "stillness":    True,#False, #True,#False,#不取消运动约束
        #         "stillnessFristLayer":True,#False, #True,#False,#并无意义，要和stillness保持一致
        #         # 1.2 软体模块
        #         "NUM_soft":1,#1,
        #         "configSofts":{ # 软体
        #             "layer":{
        #                 # 纹理
        #                 'hidden_layers_map':0,#2,#1,#2,#4,#32,#4,
        #                 'hidden_features_map': 512, #将隐含层特征维度变为1/8
        #                 # 整体运动
        #                 'hidden_layers_global':0,#2,
        #                 'hidden_features_global':128,
        #                 # 局部运动
        #                 'hidden_layers_local':0,#2,
        #                 'hidden_features_local':128, # Mask遮挡
        #             },
        #             "useSoftMask" : False, #无法生成有意义的MASK
        #             "layerMask":{ #无效
        #                 "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #                 "hidden_layers": 0,#2, 
        #                 "use_residual": False, # 似乎还有负面作用
        #                 "posEnc":{ # 有显著作用
        #                     "num_freqs_pos":10, #3
        #                     "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                     "APE":False,
        #                 }, 
        #                 "gradualImageLayers":False,
        #                 "use_maskP":False,
        #             },
        #         },
        #         # 1.3 流体模块
        #         "NUM_fluid":0, # 0.00019 -> 0.00016、0.00015
        #         "configFluids":{ #参数数量
        #             "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #             "hidden_layers": 0,#2, 
        #             "use_residual": False, # 似乎还有负面作用
        #             "posEnc":{ # 有显著作用
        #                 "num_freqs_pos":10, #3
        #                 "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                 "APE":False,
        #             }, 
        #             "gradualImageLayers":False,
        #             "use_maskP":False,
        #         }, # 现在的首要问题是无损失地拟合出来视频
        #         # 2.损失函数
        #         "useSmooth":False, #不进行平滑约束
        #         "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "lossType":2,
        #         "lossParam":{
        #             "rm":None,#"R,S",
        #             "ra":"R,F",
        #             # "blank":"F",
        #             }, 
        #         "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_021_04", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A21-04.rigid",
        #     "input_mode": "A21-04.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },

        # { #提升刚体层维度,效果仍然很差
        #     "decouple":{ # 特征维度平方                
        #         "tag":"A21-05",#只兼容了startDecouple1
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
        #         # 1 模型本身
        #         "adaptiveFrameNumMode":0,
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数
        #             # 纹理
        #             'hidden_layers_map': 2, #2,#1,#2,#4,#32,#4,
        #             # 'hidden_features_map': 32*256, #64,#8,#2*4*512,#16*4*512,#128,#512, #128,
        #             'hidden_features_map': 128, ###-.123456789.-### 
        #             # 'hidden_features_map': 512, #将隐含层特征维度变为1/8
        #             # 整体运动（应该没被使用）
        #             'hidden_layers_global':2, #2,
        #             'hidden_features_global':128,
        #             # 局部运动（在新的代码中useLocal=False时此处的局部运动不会启用）
        #             'hidden_layers_local':2, #2, 
        #             'hidden_features_local':128,
        #         }, 
        #         "configRigids":{ # 整个刚体层模块的参数
        #             "useLocal":False, #不使用局部运动
        #             "loss_recon_all_type":"MSE",#{"myLog" 学习能力不如MSE, "MSE", "atten_d"} #我猜测均方误差更关注背景、注意力损失更关注血管
        #         }, 
        #         "openLocalDeform":False, #True,
        #         "stillness":    True,#False, #True,#False,#不取消运动约束
        #         "stillnessFristLayer":True,#False, #True,#False,#并无意义，要和stillness保持一致
        #         # 1.2 软体模块
        #         "NUM_soft":1,#1,
        #         "configSofts":{ # 软体
        #             "layer":{
        #                 # 纹理
        #                 'hidden_layers_map':2,#2,#1,#2,#4,#32,#4,
        #                 'hidden_features_map': 512, #将隐含层特征维度变为1/8
        #                 # 整体运动 #本身有整体运动吗？
        #                 'hidden_layers_global':2,#2,
        #                 'hidden_features_global':128,
        #                 # 局部运动
        #                 'hidden_layers_local':2,#2,
        #                 'hidden_features_local':128, # Mask遮挡
        #             },
        #             "useSoftMask" : False, #无法生成有意义的MASK
        #             "layerMask":{ #无效
        #                 "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #                 "hidden_layers": 0,#2, 
        #                 "use_residual": False, # 似乎还有负面作用
        #                 "posEnc":{ # 有显著作用
        #                     "num_freqs_pos":10, #3
        #                     "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                     "APE":False,
        #                 }, 
        #                 "gradualImageLayers":False,
        #                 "use_maskP":False,
        #             },
        #         },
        #         # 1.3 流体模块
        #         "NUM_fluid":0, # 0.00019 -> 0.00016、0.00015
        #         "configFluids":{ #参数数量
        #             "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #             "hidden_layers": 0,#2, 
        #             "use_residual": False, # 似乎还有负面作用
        #             "posEnc":{ # 有显著作用
        #                 "num_freqs_pos":10, #3
        #                 "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                 "APE":False,
        #             }, 
        #             "gradualImageLayers":False,
        #             "use_maskP":False,
        #         }, # 现在的首要问题是无损失地拟合出来视频
        #         # 2.损失函数
        #         "useSmooth":False, #不进行平滑约束
        #         "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "lossType":2,
        #         "lossParam":{
        #             "rm":None,#"R,S",
        #             "ra":"R,F",
        #             # "blank":"F",
        #             }, 
        #         "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_021_05", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A21-05.rigid",
        #     "input_mode": "A21-05.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },

        # { 
        #     "decouple":{ # 解耦
        #         "tag":"A21-06",#企图复现20-01的指标，刚体层静止
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数
        #             # 纹理
        #             'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #             # 'hidden_layers_map':1, # dice=0.767
        #             'hidden_features_map': 8*512,#256,#64,#8,#2*4*512,#16*4*512,#128,#512, #128,
        #             # 'hidden_features_map': 512, #将隐含层特征维度变为1/8
        #             # 整体运动
        #             'hidden_layers_global':2,
        #             'hidden_features_global':128,
        #             # 局部运动（刚体层哪来的局部运动）
        #             'hidden_layers_local':2,
        #             'hidden_features_local':128,
        #         }, 
        #         "configRigids":{ # 整个刚体层模块的参数
        #             "useLocal":False, #不使用局部运动
        #             "loss_recon_all_type":"MSE",#{"myLog" 学习能力不如MSE, "MSE", "atten_d"} #我猜测均方误差更关注背景、注意力损失更关注血管
        #         }, 
        #         "openLocalDeform":False, #True,
        #         "stillness":    True,#False, #True,#False,#不取消运动约束
        #         "stillnessFristLayer":True,#:False, #True,#False,#并无意义，要和stillness保持一致
        #         # 1.2 软体模块
        #         "NUM_soft":1,#1,
        #         "configSofts":{ # 软体
        #             ############################  没有重构约束  ############################
        #             # "layer":{ # loss=0.0002, 17min, dice=0.767·0.763 最佳
        #             #     # 纹理
        #             #     'hidden_layers_map':1,#1,#2,#4,#32,#4,
        #             #     'hidden_features_map': 512, #将隐含层特征维度变为1/8
        #             #     # 整体运动
        #             #     'hidden_layers_global':1,
        #             #     'hidden_features_global':8*128,
        #             #     # 局部运动
        #             #     'hidden_layers_local':1,
        #             #     'hidden_features_local':8*128, # Mask遮挡
        #             # },
        #             # "layer":{ #0.748
        #             #     # 纹理
        #             #     'hidden_layers_map':0,#1,#2,#4,#32,#4,
        #             #     'hidden_features_map': 512, #将隐含层特征维度变为1/8
        #             #     # 整体运动
        #             #     'hidden_layers_global':0,
        #             #     'hidden_features_global':8*128,
        #             #     # 局部运动
        #             #     'hidden_layers_local':0,
        #             #     'hidden_features_local':8*128, # Mask遮挡
        #             # },
        #             ############################  有重构约束  ############################
        #             "layer":{# loss=0.0003, 17min, dice=0.771 最佳
        #                 # 纹理
        #                 'hidden_layers_map':0,#1,#2,#4,#32,#4,
        #                 'hidden_features_map': 8*512, #将隐含层特征维度变为1/8
        #                 # 整体运动
        #                 'hidden_layers_global':1,
        #                 'hidden_features_global':8*128,
        #                 # 局部运动
        #                 'hidden_layers_local':1,
        #                 'hidden_features_local':8*128, # Mask遮挡
        #             },
        #             "useSoftMask" : False, #无法生成有意义的MASK
        #             "layerMask":{ #无效
        #                 "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #                 "hidden_layers": 2, 
        #                 "use_residual": False, # 似乎还有负面作用
        #                 "posEnc":{ # 有显著作用
        #                     "num_freqs_pos":10, #3
        #                     "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                     "APE":False,
        #                 }, 
        #                 "gradualImageLayers":False,
        #                 "use_maskP":False,
        #             },
        #         },
        #         # 1.3 流体模块
        #         "NUM_fluid":0, # 0.00019 -> 0.00016、0.00015
        #         "configFluids":{ #参数数量
        #             "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #             "hidden_layers": 2, 
        #             "use_residual": False, # 似乎还有负面作用
        #             "posEnc":{ # 有显著作用
        #                 "num_freqs_pos":10, #3
        #                 "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                 "APE":False,
        #             }, 
        #             "gradualImageLayers":False,
        #             "use_maskP":False,
        #         }, # 现在的首要问题是无损失地拟合出来视频
        #         # 2.损失函数
        #         "useSmooth":False, #不进行平滑约束
        #         "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "lossType":2,
        #         "lossParam":{
        #             "rm":"F",#"R,S",
        #             "ra":"R,S",
        #             # "blank":"F",
        #             }, 
        #         "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "A21-06", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A21-06.rigid",
        #     "input_mode": "A21-06.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },


        # { #测试一下纹理网络能否将层数改为1 #0.767
        #     "decouple":{ # 解耦
        #         "tag":"A21-07",#企图复现20-01的指标，刚体层静止
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数
        #             # 纹理
        #             'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #             'hidden_features_map': 8*512,#256,#64,#8,#2*4*512,#16*4*512,#128,#512, #128,
        #             # 'hidden_features_map': 512, #将隐含层特征维度变为1/8
        #             # 整体运动
        #             'hidden_layers_global':0,
        #             'hidden_features_global':128,
        #             # 局部运动（刚体层哪来的局部运动）
        #             'hidden_layers_local':0,
        #             'hidden_features_local':128,
        #         }, 
        #         "configRigids":{ # 整个刚体层模块的参数
        #             "useLocal":False, #不使用局部运动
        #             "loss_recon_all_type":"MSE",#{"myLog" 学习能力不如MSE, "MSE", "atten_d"} #我猜测均方误差更关注背景、注意力损失更关注血管
        #         }, 
        #         "openLocalDeform":False, #True,
        #         "stillness":    True,#False, #True,#False,#不取消运动约束
        #         "stillnessFristLayer":True,#:False, #True,#False,#并无意义，要和stillness保持一致
        #         # 1.2 软体模块
        #         "NUM_soft":1,#1,
        #         "configSofts":{ # 软体
        #             "layer":{
        #                 # 纹理
        #                 'hidden_layers_map':0,#1,#2,#4,#32,#4,
        #                 'hidden_features_map': 8*512, #将隐含层特征维度变为1/8
        #                 # 整体运动
        #                 'hidden_layers_global':1,
        #                 'hidden_features_global':8*128,
        #                 # 局部运动
        #                 'hidden_layers_local':1,
        #                 'hidden_features_local':8*128, # Mask遮挡
        #             },
        #             "useSoftMask" : False, #无法生成有意义的MASK
        #             "layerMask":{ #无效
        #                 "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #                 "hidden_layers": 2, 
        #                 "use_residual": False, # 似乎还有负面作用
        #                 "posEnc":{ # 有显著作用
        #                     "num_freqs_pos":10, #3
        #                     "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                     "APE":False,
        #                 }, 
        #                 "gradualImageLayers":False,
        #                 "use_maskP":False,
        #             },
        #         },
        #         # 1.3 流体模块
        #         "NUM_fluid":0, # 0.00019 -> 0.00016、0.00015
        #         "configFluids":{ #参数数量
        #             "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #             "hidden_layers": 2, 
        #             "use_residual": False, # 似乎还有负面作用
        #             "posEnc":{ # 有显著作用
        #                 "num_freqs_pos":10, #3
        #                 "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                 "APE":False,
        #             }, 
        #             "gradualImageLayers":False,
        #             "use_maskP":False,
        #         }, # 现在的首要问题是无损失地拟合出来视频
        #         # 2.损失函数
        #         "useSmooth":False, #不进行平滑约束
        #         "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "lossType":2,
        #         "lossParam":{
        #             "rm":None,#"F",#"R,S",
        #             "ra":"R,S",
        #             # "blank":"F",
        #             }, 
        #         "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "A21-07", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A21-07.rigid",
        #     "input_mode": "A21-07.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },

        # { #测试一下纹理网络能否将层数改为1 #0.763
        #     "decouple":{ # 解耦
        #         "tag":"A21-08",#企图复现20-01的指标，刚体层静止
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数
        #             # 纹理
        #             'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #             'hidden_features_map': 8*512,#256,#64,#8,#2*4*512,#16*4*512,#128,#512, #128,
        #             # 'hidden_features_map': 512, #将隐含层特征维度变为1/8
        #             # 整体运动
        #             'hidden_layers_global':0,
        #             'hidden_features_global':128,
        #             # 局部运动（刚体层哪来的局部运动）
        #             'hidden_layers_local':0,
        #             'hidden_features_local':128,
        #         }, 
        #         "configRigids":{ # 整个刚体层模块的参数
        #             "useLocal":False, #不使用局部运动
        #             "loss_recon_all_type":"MSE",#{"myLog" 学习能力不如MSE, "MSE", "atten_d"} #我猜测均方误差更关注背景、注意力损失更关注血管
        #         }, 
        #         "openLocalDeform":False, #True,
        #         "stillness":    True,#False, #True,#False,#不取消运动约束
        #         "stillnessFristLayer":True,#:False, #True,#False,#并无意义，要和stillness保持一致
        #         # 1.2 软体模块
        #         "NUM_soft":1,#1,
        #         "configSofts":{ # 软体
        #             "layer":{
        #                 # 纹理
        #                 'hidden_layers_map':0, # 1, # 2, # 4, # 32, # 4,
        #                 'hidden_features_map': 8*512, # 将隐含层特征维度变为1/8
        #                 # 整体运动
        #                 'hidden_layers_global':0, 
        #                 'hidden_features_global':8*128, 
        #                 # 局部运动
        #                 'hidden_layers_local':0,
        #                 'hidden_features_local':8*128, # Mask遮挡
        #             },
        #             "useSoftMask" : False, #无法生成有意义的MASK
        #             "layerMask":{ #无效
        #                 "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #                 "hidden_layers": 2, 
        #                 "use_residual": False, # 似乎还有负面作用
        #                 "posEnc":{ # 有显著作用
        #                     "num_freqs_pos":10, #3
        #                     "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                     "APE":False,
        #                 }, 
        #                 "gradualImageLayers":False,
        #                 "use_maskP":False,
        #             },
        #         },
        #         # 1.3 流体模块
        #         "NUM_fluid":0, # 0.00019 -> 0.00016、0.00015
        #         "configFluids":{ #参数数量
        #             "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #             "hidden_layers": 2, 
        #             "use_residual": False, # 似乎还有负面作用
        #             "posEnc":{ # 有显著作用
        #                 "num_freqs_pos":10, #3
        #                 "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                 "APE":False,
        #             }, 
        #             "gradualImageLayers":False,
        #             "use_maskP":False,
        #         }, # 现在的首要问题是无损失地拟合出来视频
        #         # 2.损失函数
        #         "useSmooth":False, #不进行平滑约束
        #         "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "lossType":2,
        #         "lossParam":{
        #             "rm":None,#"F",#"R,S",
        #             "ra":"R,S",
        #             # "blank":"F",
        #             }, 
        #         "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "A21-08", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A21-08.rigid",
        #     "input_mode": "A21-08.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },



        { #测试一下纹理网络能否将层数改为1 #0.763
            "decouple":{ # 解耦
                "tag":"A21-09",#企图复现20-01的指标，刚体层静止
                "de-rigid":"1_sim",#去噪框架
                "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
                # 1 模型本身
                # 1.1 刚体模块
                "NUM_rigid":1,#只有一个运动的刚体
                "configRigid":{ #单个刚体层的参数
                    # 纹理
                    'hidden_layers_map':2,#1,#2,#4,#32,#4,
                    'hidden_features_map': 8*512,#256,#64,#8,#2*4*512,#16*4*512,#128,#512, #128,
                    # 'hidden_features_map': 512, #将隐含层特征维度变为1/8
                    # 整体运动
                    'hidden_layers_global':2,
                    'hidden_features_global':128,
                    # 局部运动（刚体层哪来的局部运动）
                    'hidden_layers_local':2,
                    'hidden_features_local':128,
                }, 
                "configRigids":{ # 整个刚体层模块的参数
                    "useLocal":False, #不使用局部运动
                    "loss_recon_all_type":"MSE",#{"myLog" 学习能力不如MSE, "MSE", "atten_d"} #我猜测均方误差更关注背景、注意力损失更关注血管
                }, 
                "openLocalDeform":False, #True,
                "stillness":    True,#False, #True,#False,#不取消运动约束
                "stillnessFristLayer":True,#:False, #True,#False,#并无意义，要和stillness保持一致
                # 1.2 软体模块
                "NUM_soft":1,#1,
                "configSofts":{ # 软体
                    "layer":{
                        # 纹理
                        'hidden_layers_map':2, # 1, # 2, # 4, # 32, # 4,
                        'hidden_features_map': 8*512, # 将隐含层特征维度变为1/8
                        # 整体运动
                        'hidden_layers_global':2, 
                        'hidden_features_global':8*128, 
                        # 局部运动
                        'hidden_layers_local':2,
                        'hidden_features_local':8*128, # Mask遮挡
                    },
                    "useSoftMask" : False, #无法生成有意义的MASK
                    "layerMask":{ #无效
                        "hidden_features": 64,#8,#256,#3*256,#7*256, 
                        "hidden_layers": 2, 
                        "use_residual": False, # 似乎还有负面作用
                        "posEnc":{ # 有显著作用
                            "num_freqs_pos":10, #3
                            "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
                            "APE":False,
                        }, 
                        "gradualImageLayers":False,
                        "use_maskP":False,
                    },
                },
                # 1.3 流体模块
                "NUM_fluid":0, # 0.00019 -> 0.00016、0.00015
                "configFluids":{ #参数数量
                    "hidden_features": 64,#8,#256,#3*256,#7*256, 
                    "hidden_layers": 2, 
                    "use_residual": False, # 似乎还有负面作用
                    "posEnc":{ # 有显著作用
                        "num_freqs_pos":10, #3
                        "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
                        "APE":False,
                    }, 
                    "gradualImageLayers":False,
                    "use_maskP":False,
                }, # 现在的首要问题是无损失地拟合出来视频
                # 2.损失函数
                "useSmooth":False, #不进行平滑约束
                "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
                "interval":0.1,#将计算平滑损失的步长由1改为0.5
                "lossType":2,
                "lossParam":{
                    "rm":None,#"F",#"R,S",
                    "ra":"R,S",
                    # "blank":"F",
                    }, 
                "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处
                "useMask":True, #只有lossType==1的时候才有效
                ########################
                "de-soft":None,
            },
            "name": "A21-09", #提高模型的拟合能力
            "precomputed": False,
            "noise_label":"A21-09.rigid",
            "input_mode": "A21-09.rigid.non1",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": True,#False,
            "mergeMask": False,
        },




        #################################################################################################


        # { #测试一下纹理网络能否将层数改为1 #0.767
        #     "decouple":{ # 解耦
        #         "tag":"A21-test",#企图复现20-01的指标，刚体层静止
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数
        #             # # 纹理 # loss = 0.00005
        #             # 'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #             # 'hidden_features_map': 8*512,#256,#64,#8,#2*4*512,#16*4*512,#128,#512, #128,
        #             # 纹理 # loss = 0.00005
        #             'hidden_layers_map':4,#1,#2,#4,#32,#4, #1层与2层的区别巨大
        #             'hidden_features_map': 8*512,#256,#64,#8,#2*4*512,#16*4*512,#128,#512, #128,
        #             # 'hidden_features_map': 512, #将隐含层特征维度变为1/8
        #             # 整体运动
        #             'hidden_layers_global':0,
        #             'hidden_features_global':128,
        #             # 局部运动（刚体层哪来的局部运动）
        #             'hidden_layers_local':0,
        #             'hidden_features_local':128,
        #         }, 
        #         "configRigids":{ # 整个刚体层模块的参数
        #             "useLocal":False, #不使用局部运动
        #             "loss_recon_all_type":"MSE",#{"myLog" 学习能力不如MSE, "MSE", "atten_d"} #我猜测均方误差更关注背景、注意力损失更关注血管
        #         }, 
        #         "openLocalDeform":False, #True,
        #         "stillness":    True,#False, #True,#False,#不取消运动约束
        #         "stillnessFristLayer":True,#:False, #True,#False,#并无意义，要和stillness保持一致
        #         # 1.2 软体模块
        #         "NUM_soft":0,#1,
        #         "configSofts":{ # 软体
        #             "layer":{
        #                 # 纹理
        #                 'hidden_layers_map':0, # 1, # 2, # 4, # 32, # 4,
        #                 'hidden_features_map': 8*512, # 将隐含层特征维度变为1/8
        #                 # 整体运动
        #                 'hidden_layers_global':0, 
        #                 'hidden_features_global':8*128, 
        #                 # 局部运动
        #                 'hidden_layers_local':0,
        #                 'hidden_features_local':8*128, # Mask遮挡
        #             },
        #             "useSoftMask" : False, #无法生成有意义的MASK
        #             "layerMask":{ #无效
        #                 "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #                 "hidden_layers": 2, 
        #                 "use_residual": False, # 似乎还有负面作用
        #                 "posEnc":{ # 有显著作用
        #                     "num_freqs_pos":10, #3
        #                     "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                     "APE":False,
        #                 }, 
        #                 "gradualImageLayers":False,
        #                 "use_maskP":False,
        #             },
        #         },
        #         # 1.3 流体模块
        #         "NUM_fluid":0, # 0.00019 -> 0.00016、0.00015
        #         "configFluids":{ #参数数量
        #             "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #             "hidden_layers": 2, 
        #             "use_residual": False, # 似乎还有负面作用
        #             "posEnc":{ # 有显著作用
        #                 "num_freqs_pos":10, #3
        #                 "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                 "APE":False,
        #             }, 
        #             "gradualImageLayers":False,
        #             "use_maskP":False,
        #         }, # 现在的首要问题是无损失地拟合出来视频
        #         # 2.损失函数
        #         "useSmooth":False, #不进行平滑约束
        #         "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "lossType":2,
        #         "lossParam":{
        #             "rm":None,#"F",#"R,S",
        #             "ra":"R",
        #             # "blank":"F",
        #             }, 
        #         "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "A21-test", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A21-test.rigid",
        #     "input_mode": "A21-test.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },

    ]#不要重复做实验，要相信之前的结果
    #可以相信的东西：静态刚体的纹理、无局部刚体的运动
    #刚体的局部运动
    #软体使用刚体的全局运动

print("重构损失约束刚体、软体。")




