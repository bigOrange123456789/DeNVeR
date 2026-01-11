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
        
        # {#刚体层+软体层(基于刚体) #只有整体运动 # 1Rm+1S -> non_rigid
        #     "decouple":{#解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"A-e2e",#只兼容了startDecouple1
        #         "useSmooth":False,#不平滑
        #         "stillnessFristLayer":False,#并无意义，要和stillness保持一致
        #         "stillness":False,#不取消运动约束
        #         "NUM_soft":1,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_04_end2end",
        #     "precomputed": False,
        #     "input_mode": "A-e2e.rigid.main_non2",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        ####################### xca_dataset_sub4(用于分析刚体层解耦造成的运动伪影) ####################### 
        # 如何让解耦过程变成正交的
        ##########################  DeNVeR.019(尝试解决两大难点)   ##########################  
        # {# 完美拟合视频（提高模型对视频的拟合程度）
        #     "decouple":{ # 解耦
        #         "tag":"A19-config",#只兼容了startDecouple1
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":4000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
                
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":0,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数
        #                 # 能够完全重构图片 (epoch=2000: loss=0.00002341)
        #                 # "loss_recon_all_type":"MSE",
        #                 # 'hidden_layers_map':2,
        #                 # 'hidden_features_map': 2*4*512,

        #                 # 不能完全重构图片 (epoch=2000: loss=0.025)
        #                 # "loss_recon_all_type":"myLog", #是损失函数的问题
        #                 # 'hidden_layers_map':2,
        #                 # 'hidden_features_map': 2*4*512,
        #             # 纹理
        #             'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #             'hidden_features_map': 2*4*512,#2*4*512,#16*4*512,#128,#512, #128,
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
        #         "stillness":True,#False,#不取消运动约束
        #         "stillnessFristLayer":True,#False,#并无意义，要和stillness保持一致
        #         # 1.2 软体模块
        #         "NUM_soft":0,#1,
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
        #                 'hidden_features_local':128,
        #             }
        #         },
        #         # 1.3 流体模块
        #         "NUM_fluid":4, # 0.00019 -> 0.00016、0.00015
        #         "configFluids":{ #参数数量
                    
        #             "hidden_features": 7*256, 
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
        #         "lossParam":{"rm":None,"ra":"F"},#最佳效果
        #         # "lossParam":{"rm":None,"ra":"f01"},
        #         "useMask":True, #只有lossType==1的时候才有效

        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_019_02_updateConfig", #提高模型的拟合能力
        #     "precomputed": False,
        #     "input_mode": "A19-config.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        ##########################  DeNVeR.20(提升分割效果)   ##########################  
        # {# 20-01: 刚体和软体、不遮挡重构; 刚体去噪 (效果较好)
        #     "decouple":{ # 解耦
        #         "tag":"A20-base",#只兼容了startDecouple1
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
                
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数
        #                 # 能够完全重构图片 (epoch=2000: loss=0.00002341)
        #                 # "loss_recon_all_type":"MSE",
        #                 # 'hidden_layers_map':2,
        #                 # 'hidden_features_map': 2*4*512,

        #                 # 不能完全重构图片 (epoch=2000: loss=0.025)
        #                 # "loss_recon_all_type":"myLog", #是损失函数的问题
        #                 # 'hidden_layers_map':2,
        #                 # 'hidden_features_map': 2*4*512,
        #             # 纹理
        #             'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #             'hidden_features_map': 2*4*512,#2*4*512,#16*4*512,#128,#512, #128,
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
        #         "stillness":True,#False,#不取消运动约束
        #         "stillnessFristLayer":True,#False,#并无意义，要和stillness保持一致
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
        #                 'hidden_features_local':128,
        #             }
        #         },
        #         # 1.3 流体模块
        #         "NUM_fluid":0, # 0.00019 -> 0.00016、0.00015
        #         "configFluids":{ #参数数量
                    
        #             "hidden_features": 7*256, 
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
        #         "lossParam":{"rm":None,"ra":"R,S"},#最佳效果
        #         # "lossParam":{"rm":None,"ra":"f01"},
        #         "useMask":True, #只有lossType==1的时候才有效

        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_020_01_baseline", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A20-base.rigid",
        #     "input_mode": "A20-base.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,
        #     "mergeMask": False,
        # },

        # {# 20-02: {"rm":"F","ra":"R"}; recon_non (效果较差)
        #     "decouple":{ # 解耦
        #         "tag":"A20-02-soft",#只兼容了startDecouple1
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
                
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数
        #             # 纹理
        #             'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #             'hidden_features_map': 2*4*512,#2*4*512,#16*4*512,#128,#512, #128,
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
        #         "stillness":True,#False,#不取消运动约束
        #         "stillnessFristLayer":True,#False,#并无意义，要和stillness保持一致
        #         # 1.2 软体模块
        #         "NUM_soft":0,#1,
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
        #                 'hidden_features_local':128,
        #             }
        #         },
        #         # 1.3 流体模块
        #         "NUM_fluid":1, # 0.00019 -> 0.00016、0.00015
        #         "configFluids":{ #参数数量
                    
        #             "hidden_features": 7*256, 
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
        #         "lossParam":{"rm":"F","ra":"R"},
        #         # "lossParam":{"rm":None,"ra":"f01"},
        #         "useMask":True, #只有lossType==1的时候才有效

        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_020_02_newSoft", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A20-02-soft.recon",
        #     "input_mode": "A20-02-soft.recon_non",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,
        #     "mergeMask": False,
        # },

        # {# 20-03: {"rm":"F","ra":"R"}; rigid.non1 (效果较差)
        #     "name": "_020_03", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A20-02-soft.rigid",
        #     "input_mode": "A20-02-soft.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,
        #     "mergeMask": False,
        # },

        # {# 20-04: {"ra":"R,F"}; recon_non (错误)
        #     "decouple":{ # 解耦
        #         "tag":"A20-04",#只兼容了startDecouple1
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
                
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数
        #             # 纹理
        #             'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #             'hidden_features_map': 2*4*512,#2*4*512,#16*4*512,#128,#512, #128,
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
        #         "stillness":True,#False,#不取消运动约束
        #         "stillnessFristLayer":True,#False,#并无意义，要和stillness保持一致
        #         # 1.2 软体模块
        #         "NUM_soft":0,#1,
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
        #                 'hidden_features_local':128,
        #             }
        #         },
        #         # 1.3 流体模块
        #         "NUM_fluid":1, # 0.00019 -> 0.00016、0.00015
        #         "configFluids":{ #参数数量
                    
        #             "hidden_features": 7*256, 
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
        #         # "lossParam":{"rm":None,"ra":"R,S"}, #01
        #         # "lossParam":{"rm":"F","ra":"R"}, #02
        #         "lossParam":{"rm":None,"ra":"R,F"}, #04
        #         "useMask":True, #只有lossType==1的时候才有效

        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_020_04", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A20-04.recon",
        #     "input_mode": "A20-04.recon_non",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,
        #     "mergeMask": False,
        # },
        
        # {# 20-05: {"ra":"R,F"}; rigid.non1 
        #     "name": "_020_05", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A20-04.rigid",
        #     "input_mode": "A20-04.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,
        #     "mergeMask": False,
        # },

        # {# 20-06: new_soft_move 更细腻的局部运动 # follow 20-01 {ra:R+S; rigid.non1}
        #     "decouple":{ # 解耦
        #         "tag":"A20-06",#只兼容了startDecouple1
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
                
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数
        #             # 纹理
        #             'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #             'hidden_features_map': 2*4*512,#2*4*512,#16*4*512,#128,#512, #128,
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
        #         "stillness":True,#False,#不取消运动约束
        #         "stillnessFristLayer":True,#False,#并无意义，要和stillness保持一致
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
        #                 'hidden_layers_local':2,#4,
        #                 'hidden_features_local':7*256,#128,
        #             }
        #         },
        #         # 1.3 流体模块
        #         "NUM_fluid":0, # 0.00019 -> 0.00016、0.00015
        #         "configFluids":{ #参数数量
                    
        #             "hidden_features": 7*256, 
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
        #         # "lossParam":{"rm":None,"ra":"R,S"}, #01
        #         # "lossParam":{"rm":"F","ra":"R"}, #02
        #         "lossParam":{"rm":None,"ra":"R,S"}, #04
        #         "useMask":True, #只有lossType==1的时候才有效

        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_020_06", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A20-06.recon",
        #     "input_mode": "A20-06.recon_non",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,
        #     "mergeMask": False,
        # },


        # {# 20-07: new_soft_move 更细腻的局部运动 # follow 20-01 {ra:R+S; rigid.non1}
        #     "name": "_020_07", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A20-06.rigid",
        #     "input_mode": "A20-06.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,
        #     "mergeMask": False,
        # },


        # { # 没有任何意义的测试 # 后续想要: 寻找一种通过软体完全拟合视频的方法、现在的问题是软体的拟合能力不足 
        #     "decouple":{ # 解耦
        #         "tag":"A20-08",#只兼容了startDecouple1
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
                
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数
        #             # 纹理
        #             'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #             'hidden_features_map': 2*4*512,#2*4*512,#16*4*512,#128,#512, #128,
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
        #         "stillness":True,#False,#不取消运动约束
        #         "stillnessFristLayer":True,#False,#并无意义，要和stillness保持一致
        #         # 1.2 软体模块
        #         "NUM_soft":0,#1,
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
        #                 'hidden_features_local':128,
        #             }
        #         },
        #         # 1.3 流体模块
        #         "NUM_fluid":0, # 0.00019 -> 0.00016、0.00015
        #         "configFluids":{ #参数数量
                    
        #             "hidden_features": 7*256, 
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
        #         # "lossParam":{"rm":None,"ra":"R,S"}, #01
        #         # "lossParam":{"rm":"F","ra":"R"}, #02
        #         # "lossParam":{"rm":None,"ra":"R,F"}, #04
        #         "lossParam":{
        #             "rm":None,
        #             "ra":"R,F",
        #             # "blank":"F",
        #             }, 
        #         "useMask":True, #只有lossType==1的时候才有效

        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_020_08", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A20-08.rigid",
        #     "input_mode": "A20-08.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,
        #     "mergeMask": False,
        # },

        { #用于测试获取所有效果
            "decouple":{ # 解耦
                "tag":"A20-09",#只兼容了startDecouple1
                "de-rigid":"1_sim",#去噪框架
                "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
                
                # 1 模型本身
                # 1.1 刚体模块
                "NUM_rigid":1,#只有一个运动的刚体
                "configRigid":{ #单个刚体层的参数
                    # 纹理
                    'hidden_layers_map':2,#1,#2,#4,#32,#4,
                    'hidden_features_map': 2*4*512,#2*4*512,#16*4*512,#128,#512, #128,
                    # 'hidden_features_map': 512, #将隐含层特征维度变为1/8
                    # 整体运动
                    'hidden_layers_global':4,
                    'hidden_features_global':128,
                    # 局部运动
                    'hidden_layers_local':4,
                    'hidden_features_local':128,
                }, 
                "configRigids":{ # 整个刚体层模块的参数
                    "loss_recon_all_type":"MSE",#{"myLog" 学习能力不如MSE, "MSE", "atten_d"} #我猜测均方误差更关注背景、注意力损失更关注血管
                }, 
                "openLocalDeform":False,#True,
                "stillness":True,#False,#不取消运动约束
                "stillnessFristLayer":True,#False,#并无意义，要和stillness保持一致
                # 1.2 软体模块
                "NUM_soft":0,#1,
                "configSofts":{ # 软体
                    "layer":{
                        # 纹理
                        'hidden_layers_map':2,#1,#2,#4,#32,#4,
                        'hidden_features_map': 512, #将隐含层特征维度变为1/8
                        # 整体运动
                        'hidden_layers_global':4,
                        'hidden_features_global':128,
                        # 局部运动
                        'hidden_layers_local':4,
                        'hidden_features_local':128,
                    }
                },
                # 1.3 流体模块
                "NUM_fluid":0, # 0.00019 -> 0.00016、0.00015
                "configFluids":{ #参数数量
                    "hidden_features": 7*256, 
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
                # "lossParam":{"rm":None,"ra":"R,S"}, #01
                # "lossParam":{"rm":"F","ra":"R"}, #02
                # "lossParam":{"rm":None,"ra":"R,F"}, #04
                "lossParam":{
                    "rm":None,
                    "ra":"R",#"R,F",
                    # "blank":"F",
                    }, 
                "useMask":True, #只有lossType==1的时候才有效

                ########################
                "de-soft":None,
            },
            "name": "_020_09", #提高模型的拟合能力
            "precomputed": False,
            "noise_label":"A20-09.rigid",
            "input_mode": "A20-09.rigid.non1",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": False,
            "mergeMask": False,
        },
        
    ]#不要重复做实验，要相信之前的结果
    #可以相信的东西：静态刚体的纹理、无局部刚体的运动
    #刚体的局部运动
    #软体使用刚体的全局运动



