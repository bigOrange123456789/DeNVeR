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

        # { #用于测试获取所有效果
        #     "decouple":{ # 解耦
        #         "tag":"A20-09",#只兼容了startDecouple1
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":31,#2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
                
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数
        #             # 纹理
        #             'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #             'hidden_features_map': 256,#2*4*512,#2*4*512,#16*4*512,#128,#512, #128,
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
        #         # "lossParam":{"rm":None,"ra":"R,S"}, # 01
        #         # "lossParam":{"rm":"F","ra":"R"},    # 02
        #         # "lossParam":{"rm":None,"ra":"R,F"}, # 04
        #         "lossParam":{
        #             "rm":None,
        #             "ra":"R",#"R,F",
        #             # "blank":"F",
        #             }, 
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_020_09", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A20-09.rigid",
        #     "input_mode": "A20-09.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,
        #     "mergeMask": False,
        # },

        # { #复现最佳的刚体解耦效果  #方法的迁移
        #     # "decouple":{ # 解耦
        #     #     "tag":"A20-10-best1",#只兼容了startDecouple1
        #     #     "de-rigid":"1_sim",#去噪框架
        #     #     "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
                
        #     #     # 1 模型本身
        #     #     # 1.1 刚体模块
        #     #     "NUM_rigid":1,#只有一个运动的刚体
        #     #     "configRigid":{ #单个刚体层的参数
        #     #         # 纹理
        #     #         'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #     #         'hidden_features_map': 256,#64,#8,#2*4*512,#16*4*512,#128,#512, #128,
        #     #         # 'hidden_features_map': 512, #将隐含层特征维度变为1/8
        #     #         # 整体运动
        #     #         'hidden_layers_global':4,
        #     #         'hidden_features_global':128,
        #     #         # 局部运动
        #     #         'hidden_layers_local':4,
        #     #         'hidden_features_local':128,
        #     #     }, 
        #     #     "configRigids":{ # 整个刚体层模块的参数
        #     #         "loss_recon_all_type":"MSE",#{"myLog" 学习能力不如MSE, "MSE", "atten_d"} #我猜测均方误差更关注背景、注意力损失更关注血管
        #     #     }, 
        #     #     "openLocalDeform":False,#True,
        #     #     "stillness":    False,#True,#False,#不取消运动约束
        #     #     "stillnessFristLayer":False,#True,#False,#并无意义，要和stillness保持一致
        #     #     # 1.2 软体模块
        #     #     "NUM_soft":1,#1,
        #     #     "configSofts":{ # 软体
        #     #         "layer":{
        #     #             # 纹理
        #     #             'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #     #             'hidden_features_map': 512, #将隐含层特征维度变为1/8
        #     #             # 整体运动
        #     #             'hidden_layers_global':4,
        #     #             'hidden_features_global':128,
        #     #             # 局部运动
        #     #             'hidden_layers_local':4,
        #     #             'hidden_features_local':128,
        #     #         }
        #     #     },
        #     #     # 1.3 流体模块
        #     #     "NUM_fluid":0, # 0.00019 -> 0.00016、0.00015
        #     #     "configFluids":{ #参数数量
                    
        #     #         "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #     #         "hidden_layers": 2, 
        #     #         "use_residual": False, # 似乎还有负面作用
        #     #         "posEnc":{ # 有显著作用
        #     #             "num_freqs_pos":10, #3
        #     #             "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #     #             "APE":False,
        #     #         }, 
        #     #         "gradualImageLayers":False,
        #     #         "use_maskP":False,

        #     #     }, # 现在的首要问题是无损失地拟合出来视频

        #     #     # 2.损失函数
        #     #     "useSmooth":False, #不进行平滑约束
        #     #     "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #     #     "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #     #     "lossType":2,
        #     #     # "lossParam":{"rm":None,"ra":"R,S"}, #01
        #     #     # "lossParam":{"rm":"F","ra":"R"}, #02
        #     #     # "lossParam":{"rm":None,"ra":"R,F"}, #04
        #     #     "lossParam":{
        #     #         "rm":None,
        #     #         "ra":"R,S",#"R,F",
        #     #         # "blank":"F",
        #     #         }, 
        #     #     "useMask":True, #只有lossType==1的时候才有效

        #     #     ########################
        #     #     "de-soft":None,
        #     # },
            
        #     "name": "_020_10", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A20-10-best1.rigid",
        #     "input_mode": "A20-10-best1.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,
        #     "mergeMask": False,
        # },


        # { #复现最佳软体解耦效果2  
        #     "decouple":{ # 解耦
        #         "tag":"A20-11-best2",#只兼容了startDecouple1
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
        #                 'hidden_features_local':128,
        #             }
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
        #         # "lossParam":{"rm":None,"ra":"R,S"}, #01
        #         # "lossParam":{"rm":"F","ra":"R"}, #02
        #         # "lossParam":{"rm":None,"ra":"R,F"}, #04
        #         "lossParam":{
        #             "rm":"R,S",
        #             "ra":None,#"R,F",
        #             # "blank":"F",
        #             }, 
        #         "useMask":True, #只有lossType==1的时候才有效

        #         ########################
        #         "de-soft":None,
        #     },
            
        #     "name": "_020_11", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A20-11-best2.recon",
        #     "input_mode": "A20-11-best2.recon_non",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,
        #     "mergeMask": False,
        # },


        # { #复现最佳的刚体解耦效果2  
        #     "decouple":{ # 解耦
        #         "tag":"A20-12-best3",#只兼容了startDecouple1
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
        #                 'hidden_features_local':128,
        #                 # Mask遮挡
        #             }
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
        #         # "lossParam":{"rm":None,"ra":"R,S"}, #01
        #         # "lossParam":{"rm":"F","ra":"R"}, #02
        #         # "lossParam":{"rm":None,"ra":"R,F"}, #04
        #         "lossParam":{
        #             "rm":"R,S",
        #             "ra":None,#"R,F",
        #             # "blank":"F",
        #             }, 
        #         "maskPath_pathIn":"A20-10-best1.rigid.non1",
        #         "useMask":True, #只有lossType==1的时候才有效

        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_020_12", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A20-12-best3.recon",
        #     "input_mode": "A20-12-best3.recon_non",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },

        # { #准备测试有MASK遮挡的效果 #3个软体层
        #     "decouple":{ # 解耦
        #         "tag":"A20-14-softMask",#只兼容了startDecouple1
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,  
                
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
        #                 'hidden_features_local':128,
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
        #             }
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
        #             "rm":"R,S",
        #             "ra":None,#"R,F",
        #             }, 
        #         "maskPath_pathIn": None, #"A20-10-best1.rigid.non1", # 使用tag+".orig_mask"作为遮挡约束
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_020_14", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A20-14-softMask.recon", #被去除的噪声信息
        #     "input_mode": "A20-14-softMask.recon_non",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },

        { #分析单张图片的刚体拟合效果
            "decouple":{ # 解耦
                "tag":"A20-14-softMask",#只兼容了startDecouple1
                "de-rigid":"1_sim",#去噪框架
                "epoch":2000,#6000,#4000,#2000,  
                
                # 1 模型本身
                # 1.1 刚体模块
                "NUM_rigid":1,#只有一个运动的刚体
                "configRigid":{ #单个刚体层的参数
                    # 纹理
                    'hidden_layers_map':2,#1,#2,#4,#32,#4,
                    'hidden_features_map': 256,#64,#8,#2*4*512,#16*4*512,#128,#512, #128,
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
                "stillness":    False,#True,#False,#不取消运动约束
                "stillnessFristLayer":False,#True,#False,#并无意义，要和stillness保持一致
                # 1.2 软体模块
                "NUM_soft":1,#1,
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
                    }
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
                    "rm":"R,S",
                    "ra":None,#"R,F",
                    }, 
                "maskPath_pathIn": None, #"A20-10-best1.rigid.non1", # 使用tag+".orig_mask"作为遮挡约束
                "useMask":True, #只有lossType==1的时候才有效
                ########################
                "de-soft":None,
            },
            "name": "_020_14", #提高模型的拟合能力
            "precomputed": False,
            "noise_label":"A20-14-softMask.recon", #被去除的噪声信息
            "input_mode": "A20-14-softMask.recon_non",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": True,#False,
            "mergeMask": False,
        },


    ]#不要重复做实验，要相信之前的结果
    #可以相信的东西：静态刚体的纹理、无局部刚体的运动
    #刚体的局部运动
    #软体使用刚体的全局运动



