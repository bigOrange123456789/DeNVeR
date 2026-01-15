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

        # { #刚体软体动态特征MASK
        #     "decouple":{ # 解耦
        #         "tag":"A21-11",#企图复现20-01的指标，刚体层静止
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数 #有整体运动、但是没有局部运动
        #             # 纹理
        #             'hidden_layers_map':2,#1,#2,#4,#32,#4,
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
        #         "stillness":    False,#True,#False, #True,#False,#不取消运动约束
        #         "stillnessFristLayer":False,#True,#:False, #True,#False,#并无意义，要和stillness保持一致
        #         "use_dynamicFeatureMask":True,
        #         # 1.2 软体模块
        #         "NUM_soft":1,#1,
        #         "configSofts":{ # 软体
        #             "layer":{
        #                 # 纹理
        #                 'hidden_layers_map':2, # 1, # 2, # 4, # 32, # 4,
        #                 'hidden_features_map': 8*512, # 将隐含层特征维度变为1/8
        #                 # 整体运动
        #                 'hidden_layers_global':2, 
        #                 'hidden_features_global':8*128, 
        #                 # 局部运动
        #                 'hidden_layers_local':2,
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
        #         "NUM_fluid":1, # 0.00019 -> 0.00016、0.00015
        #         "configFluids":{ #参数数量
        #             "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #             "use_featureMask":True,
        #             "fm_total_steps":800, #use_featureMask=true的时候启用
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
        #         "weight_concise":0.00001,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "lossType":2,
        #         "lossParam":{#这里写错了
        #             "rm":"F",#"R,S",
        #             "ra":"R,S",
        #             # "blank":"F",
        #             }, 
        #         "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "A21-11", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A21-11.rigid",
        #     "input_mode": "A21-11.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },
        
        ####################### xca_dataset_sim4(固定视频长度) ####################### 
        ##########################  DeNVeR.22   ##########################  
        # { #刚体软体动态特征MASK #这次实验损失函数设置错误，流体拟合了背景(反映了之前的软体层无法拟合软体)
        #     "decouple":{ # 解耦
        #         "tag":"A22-01",#企图复现20-01的指标，刚体层静止
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数 #有整体运动、但是没有局部运动
        #             # 纹理
        #             'hidden_layers_map':2,#1,#2,#4,#32,#4,
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
        #         "stillness":    False,#True,#False, #True,#False,#不取消运动约束
        #         "stillnessFristLayer":False,#True,#:False, #True,#False,#并无意义，要和stillness保持一致
        #         "use_dynamicFeatureMask":True,
        #         # 1.2 软体模块
        #         "NUM_soft":1,#1,
        #         "configSofts":{ # 软体
        #             "layer":{
        #                 # 纹理
        #                 'hidden_layers_map':2, # 1, # 2, # 4, # 32, # 4,
        #                 'hidden_features_map': 8*512, # 将隐含层特征维度变为1/8
        #                 # 整体运动
        #                 'hidden_layers_global':2, 
        #                 'hidden_features_global':8*128, 
        #                 # 局部运动
        #                 'hidden_layers_local':2,
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
        #         "NUM_fluid":1, # 0.00019 -> 0.00016、0.00015
        #         "configFluids":{ #参数数量
        #             "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #             "use_featureMask":True,
        #             "fm_total_steps":800, #use_featureMask=true的时候启用
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
        #         "weight_concise":0.00001,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "lossType":2,
        #         "lossParam":{#这里写错了
        #             "rm":"F",#"R,S",
        #             "ra":"R,S",
        #             # "blank":"F",
        #             }, 
        #         "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "A22-01", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A22-01.rigid",
        #     "input_mode": "A22-01.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },

        # { #软体、刚体、流体 分别使用不同的损失函数 
        #     #效果很差、软体没啥能力、流体拟合不佳、导致刚体需要拟合流体 =》结论:1.需要提升软体能力 2.提高流体质量?
        #     "decouple":{ # 解耦
        #         "tag":"A22-02",#企图复现20-01的指标，刚体层静止
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数 #有整体运动、但是没有局部运动
        #             # 纹理
        #             'hidden_layers_map':2,#1,#2,#4,#32,#4,
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
        #         "stillness":    False,#True,#False, #True,#False,#不取消运动约束
        #         "stillnessFristLayer":False,#True,#:False, #True,#False,#并无意义，要和stillness保持一致
        #         "use_dynamicFeatureMask":True,
        #         # 1.2 软体模块
        #         "NUM_soft":1,#1,
        #         "configSofts":{ # 软体
        #             "layer":{
        #                 # 纹理
        #                 'hidden_layers_map':2, # 1, # 2, # 4, # 32, # 4,
        #                 'hidden_features_map': 8*512, # 将隐含层特征维度变为1/8
        #                 # 整体运动
        #                 'hidden_layers_global':2, 
        #                 'hidden_features_global':8*128, 
        #                 # 局部运动
        #                 'hidden_layers_local':2,
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
        #         "NUM_fluid":1, # 0.00019 -> 0.00016、0.00015
        #         "configFluids":{ #参数数量
        #             "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #             "use_featureMask":True,
        #             "fm_total_steps":800, #use_featureMask=true的时候启用
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
        #         "weight_concise":0.00001,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "lossType":2,
        #         "lossParam":{#这里写错了
        #             # "ra":"R,S",
        #             # "rm":"F",#"R,S",为啥这里拟合出了背景？
        #             #########################
        #             "ra":"R",
        #             "rm":"S",#拟合背景,
        #             "rv":"F",#拟合前景
        #         }, 
        #         "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "A22-02", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A22-02.rigid",
        #     "input_mode": "A22-02.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # }, 


        # { # 软体纹理层需要输入时间 #效果仍然非常差
        #     "decouple":{ # 解耦
        #         "tag":"A22-03",#企图复现20-01的指标，刚体层静止
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数 #有整体运动、但是没有局部运动
        #             # 纹理
        #             'hidden_layers_map':2,#1,#2,#4,#32,#4,
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
        #         "stillness":    False,#True,#False, #True,#False,#不取消运动约束
        #         "stillnessFristLayer":False,#True,#:False, #True,#False,#并无意义，要和stillness保持一致
        #         "use_dynamicFeatureMask":True,
        #         # 1.2 软体模块 #之前软体层在吃干饭,没有发挥什么作用
        #         "NUM_soft":1,#1,
        #         "configSofts":{ # 软体
        #             "layer":{
        #                 # 纹理
        #                 'hidden_layers_map':2, # 1, # 2, # 4, # 32, # 4,
        #                 'hidden_features_map': 8*512, # 将隐含层特征维度变为1/8
        #                 # 整体运动
        #                 'hidden_layers_global':2, 
        #                 'hidden_features_global':8*128, 
        #                 # 局部运动
        #                 'hidden_layers_local':2,
        #                 'hidden_features_local':8*128, # Mask遮挡
        #                 #动态纹理
        #                 "dynamicTex":True,
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
        #         "NUM_fluid":1, # 0.00019 -> 0.00016、0.00015
        #         "configFluids":{ #参数数量
        #             "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #             "use_featureMask":True,
        #             "fm_total_steps":800, #use_featureMask=true的时候启用
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
        #         "weight_concise":0.00001, #简化损失函数的权重似乎过低，应当适当提高
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "lossType":2,
        #         "lossParam":{#这里写错了
        #             # "ra":"R,S",
        #             # "rm":"F",#"R,S",为啥这里拟合出了背景？
        #             #########################
        #             "ra":"R",
        #             "rm":"S",#拟合背景,
        #             "rv":"F",#拟合前景
        #             "rv_":0,
        #         }, 
        #         "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "A22-03", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A22-03.rigid",
        #     "input_mode": "A22-03.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },

        # { # 降低复杂度、无软体 流体表示血管、静止刚体表示背景 #软体层纹理和所有运动应该有relu
        #   # 略有提升，但总体结果依然很差
        #     "decouple":{ # 解耦
        #         "tag":"A22-04",#企图复现20-01的指标，刚体层静止
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数 #有整体运动、但是没有局部运动
        #             # 纹理
        #             'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #             'hidden_features_map': 2*512,#256,#64,#8,#2*4*512,#16*4*512,#128,#512, #128,
        #             # 'hidden_features_map': 512, #将隐含层特征维度变为1/8
        #             # 整体运动
        #             'hidden_layers_global':2,
        #             'hidden_features_global':32,
        #             # 局部运动（刚体层哪来的局部运动）
        #             'hidden_layers_local':2,
        #             'hidden_features_local':32,
        #         }, 
        #         "configRigids":{ # 整个刚体层模块的参数
        #             "useLocal":False, #不使用局部运动
        #             "loss_recon_all_type":"MSE",#{"myLog" 学习能力不如MSE, "MSE", "atten_d"} #我猜测均方误差更关注背景、注意力损失更关注血管
        #         }, 
        #         "openLocalDeform":False, #True,
        #         "stillness":    True,#True,#False, #True,#False,#不取消运动约束
        #         "stillnessFristLayer":  True,#True,#:False, #True,#False,#并无意义，要和stillness保持一致
        #         "use_dynamicFeatureMask":True,
        #         # 1.2 软体模块 
        #         "NUM_soft":0,#1,
        #         "configSofts":{ # 软体
        #             "layer":{
        #                 # 纹理
        #                 'hidden_layers_map':2, # 1, # 2, # 4, # 32, # 4,
        #                 'hidden_features_map': 2*512, # 将隐含层特征维度变为1/8
        #                 # 整体运动
        #                 'hidden_layers_global':2, 
        #                 'hidden_features_global':2*128, 
        #                 # 局部运动
        #                 'hidden_layers_local':2,
        #                 'hidden_features_local':2*128, # Mask遮挡
        #                 #动态纹理
        #                 "dynamicTex":True,
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
        #         "NUM_fluid":1, # 0.00019 -> 0.00016、0.00015
        #         "configFluids":{ #参数数量
        #             "hidden_features": 16,#8,#256,#3*256,#7*256, 
        #             "use_featureMask":True,
        #             "fm_total_steps":800, #use_featureMask=true的时候启用
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
        #         "weight_concise":0.00001, #简化损失函数的权重似乎过低，应当适当提高
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "lossType":2,
        #         "lossParam":{#这里写错了
        #             "ra":"R",
        #             "rm":"F",
        #             "rv":None,
        #             #########################
        #             # "ra":"R",
        #             # "rm":"S",#拟合背景,
        #             # "rv":"F",#拟合前景
        #             # "rv_":0,
        #         }, 
        #         "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "A22-04", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A22-04.rigid",
        #     "input_mode": "A22-04.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },

        { #与最佳的情况22-11采用相同的参数, 只是视频长度不同 #现在框架中的缺点：刚体的运动、软体模块
          #效果仍然较差 => 视频分解效果不佳的原因应该是视频长度更短了
            "decouple":{ # 解耦
                "tag":"A22-05",#企图复现20-01的指标，刚体层静止
                "de-rigid":"1_sim",#去噪框架
                "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
                # 1 模型本身
                # 1.1 刚体模块
                "NUM_rigid":1,#只有一个运动的刚体
                "configRigid":{ #单个刚体层的参数 #有整体运动、但是没有局部运动
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
                "stillness":    False,#True,#False, #True,#False,#不取消运动约束
                "stillnessFristLayer":False,#True,#:False, #True,#False,#并无意义，要和stillness保持一致
                "use_dynamicFeatureMask":True,
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
                        #动态纹理
                        "dynamicTex":False,
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
                "NUM_fluid":1, # 0.00019 -> 0.00016、0.00015
                "configFluids":{ #参数数量
                    "hidden_features": 64,#8,#256,#3*256,#7*256, 
                    "use_featureMask":True,
                    "fm_total_steps":800, #use_featureMask=true的时候启用
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
                "weight_concise":0.00001,
                "interval":0.1,#将计算平滑损失的步长由1改为0.5
                "lossType":2,
                "lossParam":{
                    "rm":"F",#"R,S",
                    "ra":"R,S",
                    "rv":None,
                    # "blank":"F",
                    }, 
                "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处
                "useMask":True, #只有lossType==1的时候才有效
                ########################
                "de-soft":None,
            },
            "name": "A22-05", #提高模型的拟合能力
            "precomputed": False,
            "noise_label":"A22-05.rigid",
            "input_mode": "A22-05.rigid.non1",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": True,#False,
            "mergeMask": False,
        },



    ]#不要重复做实验，要相信之前的结果

    
    '''
    需要验证的小点:
        刚体重构损失约束 MSE=>mylog
        流体层拟合血管的时候 背景也添加微弱的约束
        提高concise损失的权重
    还没有实现的功能：
        像素语义特征
    '''

print("重构损失约束刚体、软体。")




