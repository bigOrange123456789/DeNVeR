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
        # { # 软体没有捕获到足够信息
        #     "decouple":{ # 解耦
        #         "tag":"A23-19",
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
        #         "dynamicVesselMask":{#有较长的时间开销
        #             "startEpoch":1000,
        #             "intervalEpoch":3000,#300,
        #         },
        #         # "dynamicVesselMask":False,
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数 #有整体运动、但是没有局部运动
        #             # 纹理
        #             'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #             'hidden_features_map': 8*512,#256,#64,#8,#2*4*512,#16*4*512,#128,#512, #128,
        #             # 'hidden_features_map': 512, #将隐含层特征维度变为1/8
        #             # 整体运动
        #             'hidden_layers_global':1,
        #             'hidden_features_global':1,
        #             # 局部运动（刚体层哪来的局部运动）
        #             'hidden_layers_local':1,
        #             'hidden_features_local':1,
        #         }, 
        #         "configRigids":{ # 整个刚体层模块的参数
        #             "useLocal":False, #不使用局部运动
        #             # "loss_recon_all_type":"MSE",#{"myLog" 学习能力不如MSE, "MSE", "atten_d"} #我猜测均方误差更关注背景、注意力损失更关注血管
        #         }, 
        #         "openLocalDeform":False, #True,
        #         "stillness":    True,#False,#True,#False, #True,#False,#不取消运动约束
        #         "stillnessFristLayer":True,#False,#True,#:False, #True,#False,#并无意义，要和stillness保持一致
        #         "use_dynamicFeatureMask":False,#True,
        #         # 1.2 软体模块
        #         "NUM_soft":1,
        #         "configSofts":{ # 软体
        #             "useLocal":False, #True,
        #             "layer":{
        #                 # 纹理
        #                 "dynamicTex":True, #动态纹理
        #                 'hidden_layers_map':2*2, # 1, # 2, # 4, # 32, # 4,
        #                 'hidden_features_map': 64,#8*512, # 将隐含层特征维度变为1/8
        #                 # 整体运动
        #                 'hidden_layers_global':1,#2, 
        #                 'hidden_features_global':1,#8*128, 
        #                 # 局部运动
        #                 'hidden_layers_local':1,#2,
        #                 'hidden_features_local':1,#8*128, # Mask遮挡                        
        #                 #######################
        #                 "use_featureMask":True,
        #                 "fm_total_steps":800, #use_featureMask=true的时候启用
        #                 "use_residual": False, # 似乎还有负面作用
        #                 "posEnc":{ # 有显著作用
        #                     "num_freqs_pos":10, #3
        #                     "num_freqs_time":100, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                     # 位置编码的频率是2的n次方，当n过大的时候容易超出浮点数上限出现None。 # PE(x) = [x, sin(2⁰·π·x), cos(2⁰·π·x), sin(2¹·π·x), cos(2¹·π·x), ..., 
        #                     "APE":False, #没有启用渐进式位置编码、启用不是改为True
        #                 }, 
        #             },
        #             "useSoftMask" : False, #无法生成有意义的MASK
        #             "layerMask":{ #无效
        #                 "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #                 "hidden_layers": 2, 
        #                 "use_residual": False, # 似乎还有负面作用
        #                 "posEnc":{ 
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
        #             "hidden_layers": 2*2, 
        #             "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #             "dynamicTex":True,#动态纹理 #用于兼容layer2类接口
        #             "use_featureMask":True,
        #             "fm_total_steps":800, #use_featureMask=true的时候启用
        #             "use_residual": False, # 似乎还有负面作用
        #             "posEnc":{ # 有显著作用
        #                 "num_freqs_pos":10, #3
        #                 "num_freqs_time":100,#*2,#5, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                 "APE":False, #没有启用渐进式位置编码、启用不是改为True
        #             }, 
        #             "gradualImageLayers":False, #没啥用的功能
        #             # "use_maskP":False, #自动学习MASK遮挡图、无效功能
        #             "vesselMaskInference":True,#False,
        #         }, # 现在的首要问题是无损失地拟合出来视频
        #         # 2.损失函数
        #         "useSmooth":False, #不进行平滑约束
        #         "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #         "weight_concise":0.00001,
        #         "weight_component": 1,#分量约束（子衰减量小于总衰减量=>子衰减结果大于总衰减结果）
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "lossType":2,
        #         "lossParam":{ 
        #             "ra":"R", 
        #             "rm":"S", #背景 #很奇怪、软体层为啥能看到血管
        #             "rv":"F", #前景
        #             }, 
        #         "lossFunType":{ #无法只拟合血管 #"MSE", "myLog", "atten_d"
        #             "ra":"MSE",
        #             "rm":"MSE", #背景更清晰一些
        #             "rv":"myLog",#"MSE", #更模糊一些
        #             "rv_eps":0.5,#0.1,#该参数的效果还没有被测试 #训练不足
        #             "vesselMask_eps":1,#0.1,#0.25,
        #         }, 
        #         "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处 #是否使用预先计算好的MASK
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "A23-19", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A23-19.rigid",
        #     "input_mode": "A23-19.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },

        ####################### xca_dataset(原始数据集) ####################### 
        ##########################  DeNVeR.24   ##########################  
        ##########################  提高代码的兼容性和可读性,并且不考虑兼容之前的参数格式   ##########################  

        # { 
        #     "decouple":{ # 解耦
        #         "tag":"A24-01",
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,#2000,#2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
        #         "dynamicVesselMask":{#有较长的时间开销
        #             "startEpoch":1000*10,
        #             "intervalEpoch":3000,#300,
        #         },
        #         # "dynamicVesselMask":False,
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigids":{ # 整个刚体层模块的参数
        #             "layer":{
        #                 "use_residual":{
        #                     "R":False,
        #                     "S":False,
        #                     "T":False,
        #                 },
        #                 # 整体运动
        #                 "useGlobal":False,
        #                 'hidden_layers_global':1,
        #                 'hidden_features_global':1,
        #                 "globalMotionMode":2,#[6矩阵,4移动旋转放缩,3,2移动]
        #                 "use_rot":False, #"globalMotionMode"为3的时候才有效
        #                 "use_sca":False,
        #                 # 局部运动
        #                 "useLocal":False,
        #                 'hidden_layers_local':1,
        #                 'hidden_features_local':1,
        #                 # 纹理
        #                 "dynamicTex":False, #动态纹理
        #                 'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #                 'hidden_features_map': 8*512,#256,#64,#8,#2*4*512,#16*4*512,#128,#512, #128,
        #                 "posEnc":False,
        #                 "use_featureMask":False,
        #             },
        #         }, 
        #         "openLocalDeform":False, #True,
        #         "stillnessFristLayer":True,#False,#True,#:False, #True,#False,#并无意义，要和stillness保持一致
        #         "use_dynamicFeatureMask":False,#True,
        #         # 1.2 软体模块
        #         "NUM_soft":1,
        #         "configSofts":{ # 软体
        #             "layer":{
        #                 "use_residual":{
        #                     "R":False,
        #                     "S":False,
        #                     "T":False,
        #                 },
        #                 # 1.整体运动
        #                 "useGlobal":False, #True,
        #                 'hidden_layers_global':1,#2, 
        #                 'hidden_features_global':1,#8*128, 
        #                 # 2.局部运动
        #                 "useLocal":False, #True,
        #                 'hidden_layers_local':1,#2,
        #                 'hidden_features_local':1,#8*128, # Mask遮挡
        #                 # 3.纹理
        #                 "dynamicTex":True, #动态纹理
        #                 'hidden_layers_map':4, # 1, # 2, # 4, # 32, # 4,
        #                 'hidden_features_map': 64,#8*512, # 将隐含层特征维度变为1/8
        #                 "posEnc":{ # 有显著作用
        #                     "num_freqs_pos":10, #3
        #                     "num_freqs_time":100, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                     "APE":False, #没有启用渐进式位置编码、启用不是改为True
        #                 }, # 频率是2的n次方，过大容易超出浮点数上限出现None。 # sin(2¹·π·x)  
        #                 "use_featureMask":True, #渐进式遮挡向量
        #                 "fm_total_steps":800, #use_featureMask=true的时候启用
        #             },
        #             "useSoftMask" : False, #无法生成有意义的MASK
        #             "layerMask":{ #无效
        #                 "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #                 "hidden_layers": 2, 
        #                 "use_residual": False, # 似乎还有负面作用
        #                 "posEnc":{ 
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
        #             "layer":{
        #                 "use_residual":{
        #                     "R":False,
        #                     "S":False,
        #                     "T":False,
        #                 },
        #                 # 整体运动
        #                 "useGlobal":False,
        #                 # 局部运动
        #                 "useLocal":False,
        #                 # 纹理
        #                 "dynamicTex":True,#动态纹理 #用于兼容layer2类接口
        #                 "hidden_layers_map": 4, 
        #                 "hidden_features_map": 64,#8,#256,#3*256,#7*256, 
        #                 "posEnc":{ # 有显著作用
        #                     "num_freqs_pos":10, #3
        #                     "num_freqs_time":100,#*2,#5, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                     "APE":False, #没有启用渐进式位置编码、启用不是改为True
        #                 }, 
        #                 "use_featureMask":True,
        #                 "fm_total_steps":800, #use_featureMask=true的时候启用
        #             },                    
        #             #######################
        #             "vesselMaskInference":True,#False,
        #             "gradualImageLayers":False, #没啥用的功能
        #             # "use_maskP":False, #自动学习MASK遮挡图、无效功能
        #         }, # 现在的首要问题是无损失地拟合出来视频
        #         # 2.损失函数
        #         "useSmooth":False, #不进行平滑约束
        #         "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #         "weight_concise":0.00001,
        #         "weight_component": 1,#分量约束（子衰减量小于总衰减量=>子衰减结果大于总衰减结果）
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "lossType":2,
        #         "lossParam":{ 
        #             "ra":"R", 
        #             "rm":"S", #背景 #很奇怪、软体层为啥能看到血管
        #             "rv":"F", #前景
        #             }, 
        #         "lossFunType":{ #无法只拟合血管 #"MSE", "myLog", "atten_d"
        #             "ra":"MSE",
        #             "rm":"MSE", #背景更清晰一些
        #             "rv":"myLog",#"MSE", #更模糊一些
        #             "rv_eps":0.5,#0.1,#该参数的效果还没有被测试 #训练不足
        #             "vesselMask_eps":1,#0.1,#0.25,
        #         }, 
        #         "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处 #是否使用预先计算好的MASK
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "A24-01", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A24-01.rigid",
        #     "input_mode": "A24-01.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },

        # { #降低epoch #流体似乎更好了
        #     "decouple":{ # 解耦
        #         "tag":"A24-02",
        #         "de-rigid":"1_sim",#去噪框架
        #         "total_steps":1000,#"epoch":1000,#2000,#2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
        #         "epochs":0.625,#
        #         "batch_size_scale":1/8,
        #         "dynamicVesselMask":{#有较长的时间开销
        #             # "startEpoch":1000*10,
        #             # "intervalEpoch":3000,#300,
        #             "startStep":0.5*10,
        #             "intervalStep":1.5,
        #         },
        #         # "dynamicVesselMask":False,
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigids":{ # 整个刚体层模块的参数
        #             "layer":{
        #                 "use_residual":{
        #                     "R":False,
        #                     "S":False,
        #                     "T":False,
        #                 },
        #                 # 整体运动
        #                 "useGlobal":False,
        #                 'hidden_layers_global':1,
        #                 'hidden_features_global':1,
        #                 "globalMotionMode":2,#[6矩阵,4移动旋转放缩,3,2移动]
        #                 "use_rot":False, #"globalMotionMode"为3的时候才有效
        #                 "use_sca":False,
        #                 # 局部运动
        #                 "useLocal":False,
        #                 'hidden_layers_local':1,
        #                 'hidden_features_local':1,
        #                 # 纹理
        #                 "dynamicTex":False, #动态纹理
        #                 'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #                 'hidden_features_map': 8*512,#256,#64,#8,#2*4*512,#16*4*512,#128,#512, #128,
        #                 "posEnc":False,
        #                 "use_featureMask":False,
        #             },
        #         }, 
        #         "openLocalDeform":False, #True,
        #         "stillnessFristLayer":True,#False,#True,#:False, #True,#False,#并无意义，要和stillness保持一致
        #         "use_dynamicFeatureMask":False,#True,
        #         # 1.2 软体模块
        #         "NUM_soft":1,
        #         "configSofts":{ # 软体
        #             "layer":{
        #                 "use_residual":{
        #                     "R":False,
        #                     "S":False,
        #                     "T":False,
        #                 },
        #                 # 1.整体运动
        #                 "useGlobal":False, #True,
        #                 'hidden_layers_global':1,#2, 
        #                 'hidden_features_global':1,#8*128, 
        #                 # 2.局部运动
        #                 "useLocal":False, #True,
        #                 'hidden_layers_local':1,#2,
        #                 'hidden_features_local':1,#8*128, # Mask遮挡
        #                 # 3.纹理
        #                 "dynamicTex":True, #动态纹理
        #                 'hidden_layers_map':4, # 1, # 2, # 4, # 32, # 4,
        #                 'hidden_features_map': 64,#8*512, # 将隐含层特征维度变为1/8
        #                 "posEnc":{ # 有显著作用
        #                     "num_freqs_pos":10, #3
        #                     "num_freqs_time":100, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                     "APE":False, #没有启用渐进式位置编码、启用不是改为True
        #                 }, # 频率是2的n次方，过大容易超出浮点数上限出现None。 # sin(2¹·π·x)  
        #                 "use_featureMask":True, #渐进式遮挡向量
        #                 "fm_total_steps":800/2000, #use_featureMask=true的时候启用
        #             },
        #             "useSoftMask" : False, #无法生成有意义的MASK
        #             "layerMask":{ #无效
        #                 "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #                 "hidden_layers": 2, 
        #                 "use_residual": False, # 似乎还有负面作用
        #                 "posEnc":{ 
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
        #             "layer":{
        #                 "use_residual":{
        #                     "R":False,
        #                     "S":False,
        #                     "T":False,
        #                 },
        #                 # 整体运动
        #                 "useGlobal":False,
        #                 # 局部运动
        #                 "useLocal":False,
        #                 # 纹理
        #                 "dynamicTex":True,#动态纹理 #用于兼容layer2类接口
        #                 "hidden_layers_map": 4, 
        #                 "hidden_features_map": 64,#8,#256,#3*256,#7*256, 
        #                 "posEnc":{ # 有显著作用
        #                     "num_freqs_pos":10, #3
        #                     "num_freqs_time":100,#*2,#5, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                     "APE":False, #没有启用渐进式位置编码、启用不是改为True
        #                 }, 
        #                 "use_featureMask":True,
        #                 "fm_total_steps":800, #use_featureMask=true的时候启用
        #             },                    
        #             #######################
        #             "vesselMaskInference":True,#False,
        #             "gradualImageLayers":False, #没啥用的功能
        #             # "use_maskP":False, #自动学习MASK遮挡图、无效功能
        #         }, # 现在的首要问题是无损失地拟合出来视频
        #         # 2.损失函数
        #         "useSmooth":False, #不进行平滑约束
        #         "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #         "weight_concise":0.00001,
        #         "weight_component": 1,#分量约束（子衰减量小于总衰减量=>子衰减结果大于总衰减结果）
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "lossType":2,
        #         "lossParam":{ 
        #             "ra":"R", 
        #             "rm":"S", #背景 #很奇怪、软体层为啥能看到血管
        #             "rv":"F", #前景
        #             }, 
        #         "lossFunType":{ #无法只拟合血管 #"MSE", "myLog", "atten_d"
        #             "ra":"MSE",
        #             "rm":"MSE", #背景更清晰一些
        #             "rv":"myLog",#"MSE", #更模糊一些
        #             "rv_eps":0.5,#0.1,#该参数的效果还没有被测试 #训练不足
        #             "vesselMask_eps":1,#0.1,#0.25,
        #         }, 
        #         "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处 #是否使用预先计算好的MASK
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "A24-02", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A24-02.rigid",
        #     "input_mode": "A24-02.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },
        # { #修复刚体层的stage bug之后
        #     #现在的问题是血管区域的训练不足
        #     "decouple":{ # 解耦
        #         "tag":"A24-03",
        #         "de-rigid":"1_sim",#去噪框架
        #         #"total_steps":2000,#1000,#"epoch":1000,#2000,#2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
        #         "epochs":0.625,#
        #         "batch_size_scale":1/8,
        #         "dynamicVesselMask":{#有较长的时间开销
        #             # "startEpoch":1000*10,
        #             # "intervalEpoch":3000,#300,
        #             "startStep":0.5*10,
        #             "intervalStep":1.5,
        #         },
        #         # "dynamicVesselMask":False,
        #         # 1 模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigids":{ # 整个刚体层模块的参数
        #             "layer":{
        #                 "use_residual":{
        #                     "R":False,
        #                     "S":False,
        #                     "T":False,
        #                 },
        #                 # 整体运动
        #                 "useGlobal":False,
        #                 'hidden_layers_global':1,
        #                 'hidden_features_global':1,
        #                 "globalMotionMode":2,#[6矩阵,4移动旋转放缩,3,2移动]
        #                 "use_rot":False, #"globalMotionMode"为3的时候才有效
        #                 "use_sca":False,
        #                 # 局部运动
        #                 "useLocal":False,
        #                 'hidden_layers_local':1,
        #                 'hidden_features_local':1,
        #                 # 纹理
        #                 "dynamicTex":False, #动态纹理
        #                 'hidden_layers_map':2,#1,#2,#4,#32,#4,
        #                 'hidden_features_map': 8*512,#256,#64,#8,#2*4*512,#16*4*512,#128,#512, #128,
        #                 "posEnc":False,
        #                 "use_featureMask":False,
        #             },
        #         }, 
        #         "openLocalDeform":False, #True,
        #         "stillnessFristLayer":True,#False,#True,#:False, #True,#False,#并无意义，要和stillness保持一致
        #         "use_dynamicFeatureMask":False,#True,
        #         # 1.2 软体模块
        #         "NUM_soft":1,
        #         "configSofts":{ # 软体
        #             "layer":{
        #                 "use_residual":{
        #                     "R":False,
        #                     "S":False,
        #                     "T":False,
        #                 },
        #                 # 1.整体运动
        #                 "useGlobal":False, #True,
        #                 'hidden_layers_global':1,#2, 
        #                 'hidden_features_global':1,#8*128, 
        #                 # 2.局部运动
        #                 "useLocal":False, #True,
        #                 'hidden_layers_local':1,#2,
        #                 'hidden_features_local':1,#8*128, # Mask遮挡
        #                 # 3.纹理
        #                 "dynamicTex":True, #动态纹理
        #                 'hidden_layers_map':4, # 1, # 2, # 4, # 32, # 4,
        #                 'hidden_features_map': 64,#8*512, # 将隐含层特征维度变为1/8
        #                 "posEnc":{ # 有显著作用
        #                     "num_freqs_pos":10, #3
        #                     "num_freqs_time":100, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                     "APE":False, #没有启用渐进式位置编码、启用不是改为True
        #                 }, # 频率是2的n次方，过大容易超出浮点数上限出现None。 # sin(2¹·π·x)  
        #                 "use_featureMask":True, #渐进式遮挡向量
        #                 "fm_total_steps":800/2000, #use_featureMask=true的时候启用
        #             },
        #             "useSoftMask" : False, #无法生成有意义的MASK
        #             "layerMask":{ #无效
        #                 "hidden_features": 64,#8,#256,#3*256,#7*256, 
        #                 "hidden_layers": 2, 
        #                 "use_residual": False, # 似乎还有负面作用
        #                 "posEnc":{ 
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
        #             "layer":{
        #                 "use_residual":{
        #                     "R":False,
        #                     "S":False,
        #                     "T":False,
        #                 },
        #                 # 整体运动
        #                 "useGlobal":False,
        #                 # 局部运动
        #                 "useLocal":False,
        #                 # 纹理
        #                 "dynamicTex":True,#动态纹理 #用于兼容layer2类接口
        #                 "hidden_layers_map": 4, 
        #                 "hidden_features_map": 64,#8,#256,#3*256,#7*256, 
        #                 "posEnc":{ # 有显著作用
        #                     "num_freqs_pos":10, #3
        #                     "num_freqs_time":100,#*2,#5, #4, #1 #后面要通过这里测试时序编码能否提升效果
        #                     "APE":False, #没有启用渐进式位置编码、启用不是改为True
        #                 }, 
        #                 "use_featureMask":True,
        #                 "fm_total_steps":800/2000, #use_featureMask=true的时候启用
        #             },                    
        #             #######################
        #             "vesselMaskInference":True,#False,
        #             "gradualImageLayers":False, #没啥用的功能
        #             # "use_maskP":False, #自动学习MASK遮挡图、无效功能
        #         }, # 现在的首要问题是无损失地拟合出来视频
        #         # 2.损失函数
        #         "useSmooth":False, #不进行平滑约束
        #         "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #         "weight_concise":0.00001,
        #         "weight_component": 1,#分量约束（子衰减量小于总衰减量=>子衰减结果大于总衰减结果）
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "lossType":2,
        #         "lossParam":{ 
        #             "ra":"R", 
        #             "rm":"S", #背景 #很奇怪、软体层为啥能看到血管
        #             "rv":"F", #前景
        #             }, 
        #         "lossFunType":{ #无法只拟合血管 #"MSE", "myLog", "atten_d"
        #             "ra":"MSE",
        #             "rm":"MSE", #背景更清晰一些
        #             "rv":"myLog",#"MSE", #更模糊一些
        #             "rv_eps":0.5,#0.1,#该参数的效果还没有被测试 #训练不足
        #             "vesselMask_eps":1,#0.1,#0.25,
        #         }, 
        #         "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处 #是否使用预先计算好的MASK
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "A24-03", #提高模型的拟合能力
        #     "precomputed": False,
        #     "noise_label":"A24-03.rigid",
        #     "input_mode": "A24-03.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },

        { #修复刚体层的stage bug之后
            #现在的问题是血管区域的训练不足
            "decouple":{ # 解耦
                "tag":"A25-01",
                "de-rigid":"1_sim",#去噪框架
                #"total_steps":2000,#1000,#"epoch":1000,#2000,#2000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
                "epochs":0.625,#
                "batch_size_scale":1/8,
                "dynamicVesselMask":{#有较长的时间开销
                    # "startEpoch":1000*10,
                    # "intervalEpoch":3000,#300,
                    "startStep":0.5*10,
                    "intervalStep":1.5,
                },
                # "dynamicVesselMask":False,
                # 1 模型本身
                # 1.1 刚体模块
                "NUM_rigid":1,#只有一个运动的刚体
                "configRigids":{ # 整个刚体层模块的参数
                    "layer":{
                        "use_residual":{
                            "R":False,
                            "S":False,
                            "T":False,
                        },
                        # 整体运动
                        "useGlobal":False,
                        'hidden_layers_global':1,
                        'hidden_features_global':1,
                        "globalMotionMode":2,#[6矩阵,4移动旋转放缩,3,2移动]
                        "use_rot":False, #"globalMotionMode"为3的时候才有效
                        "use_sca":False,
                        # 局部运动
                        "useLocal":False,
                        'hidden_layers_local':1,
                        'hidden_features_local':1,
                        # 纹理
                        "dynamicTex":False, #动态纹理
                        'hidden_layers_map':2,#1,#2,#4,#32,#4,
                        'hidden_features_map': 8*512,#256,#64,#8,#2*4*512,#16*4*512,#128,#512, #128,
                        "posEnc":False,
                        "use_featureMask":False,
                    },
                }, 
                "openLocalDeform":False, #True,
                "stillnessFristLayer":True,#False,#True,#:False, #True,#False,#并无意义，要和stillness保持一致
                "use_dynamicFeatureMask":False,#True,
                # 1.2 软体模块
                "NUM_soft":1,
                "configSofts":{ # 软体
                    "layer":{
                        "use_residual":{
                            "R":False,
                            "S":False,
                            "T":False,
                        },
                        # 1.整体运动
                        "useGlobal":False, #True,
                        'hidden_layers_global':1,#2, 
                        'hidden_features_global':1,#8*128, 
                        # 2.局部运动
                        "useLocal":False, #True,
                        'hidden_layers_local':1,#2,
                        'hidden_features_local':1,#8*128, # Mask遮挡
                        # 3.纹理
                        "dynamicTex":True, #动态纹理
                        'hidden_layers_map':4, # 1, # 2, # 4, # 32, # 4,
                        'hidden_features_map': 64,#8*512, # 将隐含层特征维度变为1/8
                        "posEnc":{ # 有显著作用
                            "num_freqs_pos":10, #3
                            "num_freqs_time":100, #4, #1 #后面要通过这里测试时序编码能否提升效果
                            "APE":False, #没有启用渐进式位置编码、启用不是改为True
                        }, # 频率是2的n次方，过大容易超出浮点数上限出现None。 # sin(2¹·π·x)  
                        "use_featureMask":True, #渐进式遮挡向量
                        "fm_total_steps":800/2000, #use_featureMask=true的时候启用
                    },
                    "useSoftMask" : False, #无法生成有意义的MASK
                    "layerMask":{ #无效
                        "hidden_features": 64,#8,#256,#3*256,#7*256, 
                        "hidden_layers": 2, 
                        "use_residual": False, # 似乎还有负面作用
                        "posEnc":{ 
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
                    "layer":{
                        "use_residual":{
                            "R":False,
                            "S":False,
                            "T":False,
                        },
                        # 整体运动
                        "useGlobal":False,
                        # 局部运动
                        "useLocal":False,
                        # 纹理
                        "dynamicTex":True,#动态纹理 #用于兼容layer2类接口
                        "hidden_layers_map": 4, 
                        "hidden_features_map": 64,#8,#256,#3*256,#7*256, 
                        "posEnc":{ # 有显著作用
                            "num_freqs_pos":10, #3
                            "num_freqs_time":100,#*2,#5, #4, #1 #后面要通过这里测试时序编码能否提升效果
                            "APE":False, #没有启用渐进式位置编码、启用不是改为True
                        }, 
                        "use_featureMask":True,
                        "fm_total_steps":800/2000, #use_featureMask=true的时候启用
                    },                    
                    #######################
                    "vesselMaskInference":True,#False,
                    "gradualImageLayers":False, #没啥用的功能
                    # "use_maskP":False, #自动学习MASK遮挡图、无效功能
                }, # 现在的首要问题是无损失地拟合出来视频
                # 2.损失函数
                "useSmooth":False, #不进行平滑约束
                "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
                "weight_concise":0.00001,
                "weight_component": 1,#分量约束（子衰减量小于总衰减量=>子衰减结果大于总衰减结果）
                "interval":0.1,#将计算平滑损失的步长由1改为0.5
                "lossType":2,
                # "lossParam":{ 
                #     "ra":"R", 
                #     "rm":"S", #背景 #很奇怪、软体层为啥能看到血管
                #     "rv":"F", #前景
                #     }, 
                "lossParam":{ 
                    "ra":"R,S,F", 
                    "rm":None, 
                    "rv":None, 
                    }, 
                "lossParam_vessel":{ 
                    "ra":"F", 
                    "rm":None, 
                    "rv":None, 
                    }, 
                "lossFunType":{ #无法只拟合血管 #"MSE", "myLog", "atten_d"
                    "ra":"MSE",
                    "rm":"MSE", #背景更清晰一些
                    "rv":"myLog",#"MSE", #更模糊一些
                    "rv_eps":0.5,#0.1,#该参数的效果还没有被测试 #训练不足
                    "vesselMask_eps":1,#0.1,#0.25,
                }, 
                "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处 #是否使用预先计算好的MASK
                "useMask":True, #只有lossType==1的时候才有效
                ########################
                "de-soft":None,
            },
            "name": "A25-01", #提高模型的拟合能力
            "precomputed": False,
            "noise_label":"A25-01.rigid",
            "input_mode": "A25-01.rigid.non1",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": True,#False,
            "mergeMask": False,
        },

    ]#不要重复做实验，要相信之前的结果
    #分析一下batch

    
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





