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
        ##########################  DeNVeR.010  ##########################  
        # {
        #     "name": "1.masks", 
        #     "precomputed": True,
        #     "result_path_template": os.path.join("../DeNVeR.006/log_6/outputs", "{videoId}", "1.masks", "{frameId}"),
        #     "input_mode_for_display": "orig",
        #     "binarize": True
        # },
        # {
        #     "name": "2.2.planar", 
        #     "precomputed": True,
        #     "result_path_template": os.path.join("../DeNVeR.006/log_6/outputs", "{videoId}", "2.2.planar", "{frameId}"),
        #     "input_mode_for_display": "orig",
        #     "binarize": True
        # },
        # {
        #     "name": "3.parallel", 
        #     "precomputed": True,
        #     "result_path_template": os.path.join("../DeNVeR.006/log_6/outputs", "{videoId}", "3.parallel", "{frameId}"),
        #     "input_mode_for_display": "orig",
        #     "binarize": True
        # },
        # {
        #     "name": "4.deform", 
        #     "precomputed": True,#复制结果
        #     "result_path_template": os.path.join("../DeNVeR.006/log_6/outputs", "{videoId}", "4.deform", "{frameId}"),
        #     "input_mode_for_display": "orig",
        #     "binarize": True
        # },
        # {
        #     "name": "5.refine", 
        #     "precomputed": True,
        #     "result_path_template": os.path.join("../DeNVeR.006/log_6/outputs", "{videoId}", "5.refine", "{frameId}"),
        #     "input_mode_for_display": "orig",
        #     "binarize": True
        # },
        # {
        #     "name": "orig",
        #     "precomputed": False,
        #     "input_mode": "orig",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True
        # },
        ##########################  DeNVeR.011  ##########################  
        # {
        #     "name": "_011_continuity_01",
        #     "precomputed": False,
        #     "input_mode": "orig",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll":True,
        #     "mergeMask":True,
        # },#"_011_continuity_01-temp" : orig
        # {
        #     "name": "_011_continuity_02",
        #     "precomputed": False,
        #     "input_mode": "noRigid1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll":True,
        #     "mergeMask":True,
        # }, #"_011_continuity_02-temp" : noRigid1
        ##########################  DeNVeR.012  ##########################  
        # {
        #     "name": "_012_continuity_01",
        #     "precomputed": False,
        #     "input_mode": "fluid2",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll":False,
        # },
        # {
        #     "name": "_012_02_bigMaskFluid",
        #     "precomputed": False,
        #     "input_mode": "fluid3",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll":False,
        # },
        ####################### 改用长视频子集合测试(没有相机移动) ####################### 
        ##########################  DeNVeR.013  ##########################  
        # {
        #     "name": "_013_long01_noRigid1",
        #     "precomputed": False,
        #     "input_mode": "noRigid1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll":True,
        #     "mergeMask":True,
        # },
        # {
        #     "name": "_013_long02_bigMaskFluid",
        #     "precomputed": False,
        #     "input_mode": "fluid3",#bigMask
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll":True,#False,
        #     "mergeMask":False,
        # },
        # {
        #     "name": "_013_long03_smallMaskFluid",
        #     "precomputed": False,
        #     "input_mode": "fluid2",#smallMask
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll":True,#False,
        #     "mergeMask":False,
        # },
        # {
        #     "name": "_013_04_traditionalDSA",
        #     "precomputed": False,
        #     "input_mode": "tDSA",#smallMask
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll":True,
        #     "mergeMask":False,
        # },
        # {
        #     "name": "_013_05_orig",
        #     "precomputed": False,
        #     "input_mode": "orig",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll":True,
        #     "mergeMask":False,
        # },
        ##########################  测试整个数据集  ##########################  
        ##########################  DeNVeR.014  ##########################  
        # {
        #     "name": "noRigid1",
        #     "precomputed": False,
        #     "input_mode": "noRigid1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,
        #     "mergeMask": False,
        # },
        # {
        #     "name": "fluid2",
        #     "precomputed": False,
        #     "input_mode": "fluid2",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,
        #     "mergeMask": False,
        # },
        # 使用单视频进行优化，在无显著下降的情况下提高运行速度
        ####################### 长视频子集合 xca_dataset_sub1 ####################### 
        ##########################  DeNVeR.015  ##########################  
        # {#刚体去噪,训练4000批次
        #     # "decouple":{#解耦
        #     #     ########################
        #     #     "de-rigid":"1",
        #     #     "epoch":4000,          #只兼容了startDecouple1
        #     #     "tag":"A",#只兼容了startDecouple1
        #     #     ########################
        #     #     "de-soft":None,
        #     # },
        #     "name": "_015_01_noRigid1",
        #     "precomputed": False,
        #     "input_mode": "noRigid1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },
        # {#只训练2000批次
        #     # "decouple":{#解耦
        #     #     ########################
        #     #     "de-rigid":"1",
        #     #     "epoch":2000,          #只兼容了startDecouple1
        #     #     "tag":"A-01-epoch2000",#只兼容了startDecouple1
        #     #     ########################
        #     #     "de-soft":None,
        #     # },
        #     "name": "_015_02_noRigid1(b2000)",
        #     "precomputed": False,
        #     "input_mode": "A-01-epoch2000.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,
        #     "mergeMask": False,
        # },
        # {#只训练1000批次
        #     # "decouple":{#解耦
        #     #     ########################
        #     #     "de-rigid":"1",
        #     #     "epoch":1000,          #只兼容了startDecouple1
        #     #     "tag":"A-01-epoch1000",#只兼容了startDecouple1
        #     #     ########################
        #     #     "de-soft":None,
        #     # },
        #     "name": "_015_03_noRigid1(b1000)",
        #     "precomputed": False,
        #     "input_mode": "A-01-epoch1000.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },
        # {#只训练500批次
        #     # "decouple":{#解耦
        #     #     ########################
        #     #     "de-rigid":"1",
        #     #     "epoch":500,          #只兼容了startDecouple1
        #     #     "tag":"A-01-epoch500",#只兼容了startDecouple1
        #     #     ########################
        #     #     "de-soft":None,
        #     # },
        #     "name": "_015_04_noRigid1(b500)",
        #     "precomputed": False,
        #     "input_mode": "A-01-epoch500.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },
        ##########################  DeNVeR.016  ##########################  
        # {#进行刚体平滑测试
        #     "decouple":{#解耦
        #         ########################
        #         "de-rigid":"1",
        #         "epoch":1000,          #只兼容了startDecouple1
        #         "tag":"A-02-smooth",#只兼容了startDecouple1
        #         "useSmooth":True,
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_016_01_noRigid1(b1000)",
        #     "precomputed": False,
        #     "input_mode": "A-02-smooth.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,#True,#False,
        #     "mergeMask": False,
        # },
        # {#测试总刚体去噪效果是否更好: 基于主刚体层(old) vs 基于总刚体层(now)
        #     # "decouple":{#解耦
        #     #     ########################
        #     #     "de-rigid":"1",
        #     #     "epoch":1000,          #只兼容了startDecouple1
        #     #     "tag":"A-02-smooth",#只兼容了startDecouple1
        #     #     "useSmooth":True,
        #     #     ########################
        #     #     "de-soft":None,
        #     # },
        #     "name": "_016_01_noRigidAll1(b1000)",
        #     "precomputed": False,
        #     "input_mode": "A-02-smooth.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,#True,#False,
        #     "mergeMask": False,
        # },
        # {#测试不同训练批次的影响 #对相机运动的视频应该会有影响
        #     "decouple":{#解耦
        #         ########################
        #         "de-rigid":"1",
        #         "epoch":4000,          #只兼容了startDecouple1
        #         "tag":"A-02-smooth-e4000",#只兼容了startDecouple1
        #         "useSmooth":True,
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_016_02_noRigid1(b4000)",
        #     "precomputed": False,
        #     "input_mode": "A-02-smooth-e4000.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,#True,#False,
        #     "mergeMask": False,
        # },
        ##########################  DeNVeR.017(测试smooth损失对运动数据集的影响) 在相机运动的场景下测试  ##########################  
        # {#有平滑 #运行时间约40分钟
        #     "decouple":{#解耦
        #         ########################
        #         "de-rigid":"1",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"A-02-smooth-e2000",#只兼容了startDecouple1
        #         "useSmooth":True,
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_017_01_nr(b2000)smooth",
        #     "precomputed": False,
        #     "input_mode": "A-02-smooth-e2000.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,#True,#False,
        #     "mergeMask": False,
        # },
        # {#无平滑 #在相机运动的场景下测试
        #     "decouple":{#解耦
        #         ########################
        #         "de-rigid":"1",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"A-02-e2000",#只兼容了startDecouple1
        #         "useSmooth":False,
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_017_02_nr(b2000)",
        #     "precomputed": False,
        #     "input_mode": "A-02-e2000.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,#True,#False,
        #     "mergeMask": False,
        # },
        # {# 有局部位移+有平滑 
        #     "decouple":{#解耦
        #         ########################
        #         "de-rigid":"1",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"A-02-smooth-localDeform",#只兼容了startDecouple1
        #         "useSmooth":True,
        #         "openLocalDeform":True,
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_017_03_nr(smooth.localDeform)",
        #     "precomputed": False,
        #     "input_mode": "A-02-smooth-localDeform.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,#True,#False,
        #     "mergeMask": False,
        # },
        # {# 降低平滑损失的权重
        #     "decouple":{#解耦
        #         ########################
        #         "de-rigid":"1",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"A-02-smooth0.1",#只兼容了startDecouple1
        #         "useSmooth":True,
        #         "weight_smooth":0.1,
        #         "openLocalDeform":False,#True,
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_017_04_nr(smooth0.1)",
        #     "precomputed": False,
        #     "input_mode": "A-02-smooth0.1.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,#True,#False,
        #     "mergeMask": False,
        # },
        # {# 将解耦方案变为段到段方案
        #     "decouple":{#解耦
        #         ########################
        #         "de-rigid":"2_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"B",#只兼容了startDecouple1
        #         "useSmooth":True,
        #         "weight_smooth":0.1,
        #         "openLocalDeform":False,#True,
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_017_05_rigid.non(end2end)",
        #     "precomputed": False,
        #     "input_mode": "B.rigid_non",#基于去刚的解耦效果
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,#True,#False,
        #     "mergeMask": False,
        # },
        # {# 将解耦方案变为端到端方案
        #     #复用前面的视频解耦结果（效果超级差、不知道是不是视频流体层的原因）
        #     "name": "_017_06_recon.non(end2end)",
        #     "precomputed": False,
        #     "input_mode": "B.recon_non",#基于流体层的解耦效果
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,#True,#False,
        #     "mergeMask": False,
        # },
        # {# 测试原视频不去噪的分割效果
        #     #复用前面的视频解耦结果（效果超级差、不知道是不是视频流体层的原因）
        #     "name": "_017_07_orig(sub2)",
        #     "precomputed": False,
        #     "input_mode": "orig",#基于流体层的解耦效果
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,#True,#False,
        #     "mergeMask": False,
        # },
        ####################### 长视频子集合 xca_dataset_sub3 ####################### 
        ##########################  DeNVeR.018(尝试解决两大难点)   ##########################  
        # {#刚体层完全静止 2Rt
        #     "decouple":{#解耦
        #         ########################
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"A-stillness2",#只兼容了startDecouple1
        #         "useSmooth":False,
        #         "stillness":True,#刚体层完全静止
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_01",
        #     "precomputed": False,
        #     "input_mode": "A-stillness2.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,#True,#False,
        #     "mergeMask": False,
        # },
        # {#刚体层完全静止并且只有一个刚体层(查全率下降) #1Rt
        #     "decouple":{#解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"A-stillness1",#只兼容了startDecouple1
        #         "useSmooth":False,
        #         "stillness":True,#刚体层完全静止
        #         # "NUM_soft":0,
        #         "NUM_rigid":1,
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_02_NumR1",
        #     "precomputed": False,
        #     "input_mode": "A-stillness1.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {#1层刚体层（有整体运动、无局部运动） #1Rm 
        #     "decouple":{#解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"A-still0-move1",#只兼容了startDecouple1
        #         "useSmooth":False,
        #         "stillnessFristLayer":False,
        #         "stillness":True,#刚体层完全静止
        #         # "NUM_soft":0,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_03_stillFrist",
        #     "precomputed": False,
        #     "input_mode": "A-still0-move1.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {#刚体层 有整体运动的单层 ） #1Rm 
        #     "decouple":{#解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"A-still0-move1",#只兼容了startDecouple1
        #         "useSmooth":False,
        #         "stillnessFristLayer":False,
        #         "stillness":False,#
        #         # "NUM_soft":0,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_03_stillFrist",
        #     "precomputed": False,
        #     "input_mode": "A-still0-move1.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
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
        # {#刚体层+软体层(基于重构) #只有整体运动 # 1Rm+1S -> non_recon
        #     "name": "_018_05_end2endRecon",
        #     "precomputed": False,
        #     "input_mode": "A-e2e.recon_non2",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {#使用MASK # M(1Rm+1S) -> non_rigid
        #     "decouple":{ # 解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"Am-e2e",#只兼容了startDecouple1
        #         "useSmooth":False,#不平滑
        #         "stillnessFristLayer":False,#并无意义，要和stillness保持一致
        #         "stillness":False,#不取消运动约束
        #         "NUM_soft":1,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "useMask":True,
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_06_end2end.m",
        #     "precomputed": False,
        #     "input_mode": "Am-e2e.rigid.main_non2",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {#刚体层+软体层(基于重构)+使用MASK # M(1Rm+1S) -> non_recon2
        #     "name": "_018_07_end2endRecon.m",
        #     "precomputed": False,
        #     "input_mode": "Am-e2e.recon_non2",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {#刚体层+软体层(基于重构)+使用MASK # M(1Rm+1S) -> non_recon2
        #     "name": "_018_08_end2endRecon1.m",
        #     "precomputed": False,
        #     "input_mode": "Am-e2e.recon_non",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {#双重构损失：总重构损失 + 刚体层重构损失
        #     "decouple":{#解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"Am-rlr",#只兼容了startDecouple1
        #         "useSmooth":False,#不平滑
        #         "stillnessFristLayer":False,#并无意义，要和stillness保持一致
        #         "stillness":False,#不取消运动约束
        #         "NUM_soft":1,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "useMask":True,
        #         "openReconLoss_rigid":True,
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_09_reconLossRigid",
        #     "precomputed": False,
        #     "input_mode": "Am-rlr.rigid.main_non2",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {
        #     "name": "_018_10_reconLossRigid",
        #     "precomputed": False,
        #     "input_mode": "Am-rlr.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {
        #     "name": "_018_11_reconLossRigid_rs",
        #     "precomputed": False,
        #     "input_mode": "Am-rlr.recon_non",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {
        #     "name": "_018_12_reconLossRigid_rs2",
        #     "precomputed": False,
        #     "input_mode": "Am-rlr.recon_non2",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {#使用MASK # M(1Rm+1S) -> non_rigid
        #     # '''
        #     #     1R+1S
        #     #     最好的工作:_018_06_end2end
        #     #         刚体重构损失: 优化刚体层。无MASK遮挡 loss=(S*R-O)
        #     #     上次的工作:_018_09 (效果较差)
        #     #         总重构损失: 优化刚+软层。加MASK遮挡 loss=M*(S*R-O)
        #     #         刚体重构损失: 优化刚体层。无MASK遮挡 loss=M*(R-O)
        #     #     更新的工作：
        #     # '''
        #     "decouple":{ # 解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"Am-loss2",#只兼容了startDecouple1
        #         "useSmooth":False,#不平滑
        #         "stillnessFristLayer":False,#并无意义，要和stillness保持一致
        #         "stillness":False,#不取消运动约束
        #         "NUM_soft":1,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "lossType":2,
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_13_loss2",#刚体重构损失考虑、但是不考虑
        #     "precomputed": False,
        #     "input_mode": "Am-loss2.rigid.main_non2",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {#使用MASK # M(1Rm+1S) -> non_rigid
        #     "name": "_018_14_loss2_reon1",
        #     "precomputed": False,
        #     "input_mode": "Am-loss2.recon_non",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {#使用平滑损失函数
        #     "decouple":{ # 解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"Am-loss2-smooth",#只兼容了startDecouple1
        #         "useSmooth":True,#False,#不平滑
        #         "weight_smooth":0.1,
        #         "stillnessFristLayer":False,#并无意义，要和stillness保持一致
        #         "stillness":False,#不取消运动约束
        #         "NUM_soft":1,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "lossType":2,
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_15_loss2_smooth",
        #     "precomputed": False,
        #     "input_mode": "Am-loss2-smooth.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {#使用基于加速度的平滑损失函数
        #     "decouple":{ # 解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"Am-loss2-smooth-a",#只兼容了startDecouple1
        #         "useSmooth":2,#True,#False,#不平滑
        #         "weight_smooth":0.1,
        #         "stillnessFristLayer":False,#并无意义，要和stillness保持一致
        #         "stillness":False,#不取消运动约束
        #         "NUM_soft":1,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "lossType":2,
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_16_loss2_smooth_a",
        #     "precomputed": False,
        #     "input_mode": "Am-loss2-smooth-a.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {#使用基于加速度的平滑损失函数 #复现成功、效果极佳
        #     # "decouple":{ # 解耦
        #     #     "de-rigid":"1_sim",
        #     #     "epoch":2000,          #只兼容了startDecouple1
        #     #     "tag":"Am-raRS",#只兼容了startDecouple1
        #     #     "useSmooth":False,#True,#False,#不平滑
        #     #     "weight_smooth":0.1,
        #     #     "stillnessFristLayer":False,#并无意义，要和stillness保持一致
        #     #     "stillness":False,#不取消运动约束
        #     #     "NUM_soft":1,
        #     #     "NUM_rigid":1,#只有一个运动的刚体
        #     #     "lossType":2,
        #     #     # "lossParam":{"rm":"S","ra":"R"},
        #     #     "lossParam":{"rm":None,"ra":"R,S"},#最佳效果
        #     #     "useMask":True, #只有lossType==1的时候才有效
        #     #     ########################
        #     #     "de-soft":None,
        #     # },
        #     # "name": "_018_17_bestMetric", #复现指标上的最佳效果
        #     # "precomputed": False,
        #     # "input_mode": "Am-raRS.rigid.main_non1",
        #     # "norm_method": norm_calculator.calculate_mean_variance,
        #     # "binarize": True,
        #     # "inferenceAll": True,#True,#False,
        #     # "mergeMask": False,
        # },
        # {#整体运动最好不要直接用矩阵,矩阵的运动分析比较复杂
        #     "decouple":{ # 解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"Am-smooth3",#只兼容了startDecouple1
        #         "useSmooth":3, #使用3号平滑损失函数=>全局运动趋向于固定
        #         "weight_smooth":0.1,
        #         "stillnessFristLayer":False,#并无意义，要和stillness保持一致
        #         "stillness":False,#不取消运动约束
        #         "NUM_soft":1,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "lossType":2,
        #         # "lossParam":{"rm":"S","ra":"R"},
        #         "lossParam":{"rm":None,"ra":"R,S"},#最佳效果
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_17_smooth3", #复现指标上的最佳效果
        #     "precomputed": False,
        #     "input_mode": "smooth3.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {# 使用割线斜率作为导数 #分割指标和之前没啥变化、(其它:刚体层的阴影明显增强)
        #     "decouple":{ # 解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"Am-smooth3",#只兼容了startDecouple1
        #         "useSmooth":3, #使用3号平滑损失函数=>全局运动趋向于固定
        #         "weight_smooth":0.1,
        #         "stillnessFristLayer":False,#并无意义，要和stillness保持一致
        #         "stillness":False,#不取消运动约束
        #         "NUM_soft":1,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "lossType":2,
        #         # "lossParam":{"rm":"S","ra":"R"},
        #         "lossParam":{"rm":None,"ra":"R,S"},#最佳效果
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_17_smooth3", #复现指标上的最佳效果
        #     "precomputed": False,
        #     "input_mode": "Am-smooth3.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {# 超高的一致性权重 
        #     "decouple":{ # 解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"Am-smooth3-2",#只兼容了startDecouple1
        #         "useSmooth":3, #使用3号平滑损失函数=>全局运动趋向于固定
        #         "weight_smooth":1000,#0.1,
        #         "interval":1,#将计算平滑损失的步长由1改为0.5
        #         "stillnessFristLayer":False,#并无意义，要和stillness保持一致
        #         "stillness":False,#不取消运动约束
        #         "NUM_soft":1,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "lossType":2,
        #         "lossParam":{"rm":None,"ra":"R,S"},#最佳效果
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_18_bigWeight", #复现指标上的最佳效果
        #     "precomputed": False,
        #     "input_mode": "Am-smooth3-2.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {# 用二阶导数而不是一阶导作为平滑损失 
        #     "decouple":{ # 解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"Am-smooth-4",#只兼容了startDecouple1
        #         "useSmooth":4, #使用4号平滑损失函数=>全局运动趋向于固定
        #         "weight_smooth":100,#0.1,
        #         "interval":1,#将计算平滑损失的步长由1改为0.5
        #         "stillnessFristLayer":False,#并无意义，要和stillness保持一致
        #         "stillness":False,#不取消运动约束
        #         "NUM_soft":1,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "lossType":2,
        #         "lossParam":{"rm":None,"ra":"R,S"},#最佳效果
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_19_acceleration", #复现指标上的最佳效果
        #     "precomputed": False,
        #     "input_mode": "Am-smooth-4.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {# 用二阶导数而不是一阶导作为平滑损失 #将割线的计算步长设置为0.5
        #     "decouple":{ # 解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"Am-smooth-4",#只兼容了startDecouple1
        #         "useSmooth":4, #使用4号平滑损失函数=>全局运动趋向于固定
        #         "weight_smooth":100,#0.1,
        #         "interval":1,#将计算平滑损失的步长由1改为0.5
        #         "stillnessFristLayer":False,#并无意义，要和stillness保持一致
        #         "stillness":False,#不取消运动约束
        #         "NUM_soft":1,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "lossType":2,
        #         "lossParam":{"rm":None,"ra":"R,S"},#最佳效果
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_19_acceleration", #复现指标上的最佳效果
        #     "precomputed": False,
        #     "input_mode": "Am-smooth-4.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {#将割线的计算步长设置为0.5
        #     "decouple":{ # 解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"Am-smooth-5",#只兼容了startDecouple1
        #         "useSmooth":4, #使用4号平滑损失函数=>全局运动趋向于固定
        #         "weight_smooth":40,#0.1,
        #         "interval":0.5,#将计算平滑损失的步长由1改为0.5
        #         "stillnessFristLayer":False,#并无意义，要和stillness保持一致
        #         "stillness":False,#不取消运动约束
        #         "NUM_soft":1,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "lossType":2,
        #         "lossParam":{"rm":None,"ra":"R,S"},#最佳效果
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_20_acceleration2", #复现指标上的最佳效果
        #     "precomputed": False,
        #     "input_mode": "Am-smooth-5.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {# 将割线的计算步长设置为0.1
        #     "decouple":{ # 解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"Am-smooth-6",#只兼容了startDecouple1
        #         "useSmooth":4, #使用4号平滑损失函数=>全局运动趋向于固定
        #         "weight_smooth":40,#0.1,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "stillnessFristLayer":False,#并无意义，要和stillness保持一致
        #         "stillness":False,#不取消运动约束
        #         "NUM_soft":1,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "lossType":2,
        #         "lossParam":{"rm":None,"ra":"R,S"},#最佳效果
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_21_acceleration3", #复现指标上的最佳效果
        #     "precomputed": False,
        #     "input_mode": "Am-smooth-6.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {# 使用5号平滑损失
        #     "decouple":{ # 解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"Am-smooth-7",#只兼容了startDecouple1
        #         "useSmooth":5, #使用5号平滑损失函数=>全局运动趋向于固定
        #         "weight_smooth":40,#0.1,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "stillnessFristLayer":False,#并无意义，要和stillness保持一致
        #         "stillness":False,#不取消运动约束
        #         "NUM_soft":1,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "lossType":2,
        #         "lossParam":{"rm":None,"ra":"R,S"},#最佳效果
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_22_smooth5", #复现指标上的最佳效果
        #     "precomputed": False,
        #     "input_mode": "Am-smooth-7.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {# 降低平滑损失函数 #有晃动问题
        #     "decouple":{ # 解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"Am-smooth-8",#只兼容了startDecouple1
        #         "useSmooth":5, #使用5号平滑损失函数=>全局运动趋向于固定
        #         "weight_smooth":0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "stillnessFristLayer":False,#并无意义，要和stillness保持一致
        #         "stillness":False,#不取消运动约束
        #         "NUM_soft":1,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "lossType":2,
        #         "lossParam":{"rm":None,"ra":"R,S"},#最佳效果
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_23_smooth", #复现指标上的最佳效果
        #     "precomputed": False,
        #     "input_mode": "Am-smooth-8.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {# 降低平滑损失函数 #没有必要添加整体运动的平滑约束，因为他整体上是正确的
        #     "decouple":{ # 解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"Am-smooth-9",#只兼容了startDecouple1
        #         "useSmooth":6, #使用6号平滑损失函数=>全局运动趋向于固定
        #         "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "stillnessFristLayer":False,#并无意义，要和stillness保持一致
        #         "stillness":False,#不取消运动约束
        #         "NUM_soft":1,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "lossType":2,
        #         "lossParam":{"rm":None,"ra":"R,S"},#最佳效果
        #         "useMask":True, #只有lossType==1的时候才有效
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_24_smooth", #复现指标上的最佳效果
        #     "precomputed": False,
        #     "input_mode": "Am-smooth-9.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {# 打开刚体的局部形变
        #     "decouple":{ # 解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"Am-10",#只兼容了startDecouple1
        #         "useSmooth":False, #不进行平滑约束
        #         "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "stillnessFristLayer":False,#并无意义，要和stillness保持一致
        #         "stillness":False,#不取消运动约束
        #         "NUM_soft":1,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "lossType":2,
        #         "lossParam":{"rm":None,"ra":"R,S"},#最佳效果
        #         "useMask":True, #只有lossType==1的时候才有效
        #         "openLocalDeform":True,
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_018_25_", #复现指标上的最佳效果
        #     "precomputed": False,
        #     "input_mode": "Am-10.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        #### .复现整体运动最合理的版本. ####
        ####################### xca_dataset_sub4(用于分析刚体层解耦造成的运动伪影) ####################### 
        # 如何让解耦过程变成正交的
        ##########################  DeNVeR.019(尝试解决两大难点)   ##########################  
        # {
        #     "decouple":{ # 解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"A19-best",#只兼容了startDecouple1
        #         "useSmooth":False, #不进行平滑约束
        #         "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "stillnessFristLayer":False,#并无意义，要和stillness保持一致
        #         "stillness":False,#不取消运动约束
        #         "NUM_soft":1,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "lossType":2,
        #         "lossParam":{"rm":None,"ra":"R,S"},#最佳效果
        #         "useMask":True, #只有lossType==1的时候才有效
        #         "openLocalDeform":False,#True,
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_019_01_bestMetric", #复现指标上的最佳效果
        #     "precomputed": False,
        #     "input_mode": "A19-best.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        # {# 打开刚体的局部形变
        #     "decouple":{ # 解耦
        #         "de-rigid":"1_sim",
        #         "epoch":2000,          #只兼容了startDecouple1
        #         "tag":"A19-best",#只兼容了startDecouple1
        #         "useSmooth":False, #不进行平滑约束
        #         "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "stillnessFristLayer":False,#并无意义，要和stillness保持一致
        #         "stillness":False,#不取消运动约束
        #         "NUM_soft":1,
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "lossType":2,
        #         "lossParam":{"rm":None,"ra":"R,S"},#最佳效果
        #         "useMask":True, #只有lossType==1的时候才有效
        #         "openLocalDeform":False,#True,
        #         "configRigid":{
        #             'hidden_layers_map':4,
        #             'hidden_layers_global':4,
        #             'hidden_layers_local':4,
        #             'hidden_features_map':128,
        #             'hidden_features_global':128,
        #             'hidden_features_local':128,
        #         },
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_019_01_bestMetric", #复现指标上的最佳效果
        #     "precomputed": False,
        #     "input_mode": "A19-best.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#True,#False,
        #     "mergeMask": False,
        # },
        #### 一组功能验证测试 ####
        # {# 完美拟合纹理（提高模型对纹理的拟合程度）
        #     "decouple":{ # 解耦
        #         "tag":"A19-config",#只兼容了startDecouple1
        #         "de-rigid":"1_sim",#去噪框架
        #         "epoch":2000,          #只兼容了startDecouple1
                
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
        #             "loss_recon_all_type":"MSE",#{"myLog" 学习能力不如MSE, "MSE", "atten_d"}
        #         }, #
        #         "openLocalDeform":False,#True,
        #         "stillness":True,#False,#不取消运动约束
        #         "stillnessFristLayer":True,#False,#并无意义，要和stillness保持一致
        #         # 1.2 软体模块
        #         "NUM_soft":0,#1,
                

        #         # 2.损失函数
        #         "useSmooth":False, #不进行平滑约束
        #         "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "lossType":2,
        #         "lossParam":{"rm":None,"ra":"R,S"},#最佳效果
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
        {# 完美拟合视频（提高模型对视频的拟合程度）
            "decouple":{ # 解耦
                "tag":"A19-config",#只兼容了startDecouple1
                "de-rigid":"1_sim",#去噪框架
                "epoch":4000,#6000,#4000,#2000,          #只兼容了startDecouple1 #recon_all=0.00011
                
                # 1 模型本身
                # 1.1 刚体模块
                "NUM_rigid":0,#只有一个运动的刚体
                "configRigid":{ #单个刚体层的参数
                        # 能够完全重构图片 (epoch=2000: loss=0.00002341)
                        # "loss_recon_all_type":"MSE",
                        # 'hidden_layers_map':2,
                        # 'hidden_features_map': 2*4*512,

                        # 不能完全重构图片 (epoch=2000: loss=0.025)
                        # "loss_recon_all_type":"myLog", #是损失函数的问题
                        # 'hidden_layers_map':2,
                        # 'hidden_features_map': 2*4*512,
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
                "NUM_fluid":4, # 0.00019 -> 0.00016、0.00015
                "configFluids":{ #参数数量

                    # "hidden_features":512,
                    # "hidden_layers":4,

                    # "hidden_features":2*4*512,
                    # "hidden_layers":4,
                    
                    # "hidden_features":2*4*512,
                    # "hidden_layers":4*4,
                    # 93分钟，#层数太多会无法拟合

                    # "hidden_features":15*512, 
                    # "hidden_layers":2,
                    # 显存溢出

                    # "hidden_features":12*512, 
                    # "hidden_layers":2,
                    # 27 min # 整体拟合效果还不错，无法拟合血管

                    # "hidden_features":2*512, 
                    # "hidden_layers":12,
                    # "use_residual":True,
                    # 梯度消失 # 不能说明Res结构用处不大，因为参数数量减少了 

                    # "hidden_features":2*512, 
                    # "hidden_layers":2*6*6, #72层
                    # "use_residual":True, 
                    # 梯度爆炸，输出数据变为Nan


                    # "hidden_features":2*512, 
                    # "hidden_layers":12,
                    # "use_residual":True,
                    # "posEnc":{
                    #     "num_freqs_pos":10, #3
                    #     "num_freqs_time":4, #1
                    # },
                    # 5 min 梯度消失

                    # "hidden_features":2*512, 
                    # "hidden_layers":11,
                    # "use_residual":True/False, #没啥用 #带宽不够、信息传不过来
                    # "posEnc":False,
                    # 5 min 梯度消失

                    # "hidden_features":2*512, 
                    # "hidden_layers":2,
                    # "use_residual":False,
                    # "posEnc":False,
                    # 1 min 正确、模糊、血管拟合不好

                    # "hidden_features":2*512, 
                    # "hidden_layers":6, 
                    # "use_residual":False,
                    # "posEnc":False,
                    # # 3 min 正确、更清晰一些、血管拟合不好


                    # "hidden_features":2*512, 
                    # "hidden_layers":10,#9,#6,#2 
                    # "use_residual":False,
                    # "posEnc":False,
                    # 3~5 min 正确、更清晰一些、血管拟合不好

                    # "hidden_features":2*512, 
                    # "hidden_layers":11, 
                    # "use_residual":False, 
                    # "posEnc":{ #能够缓解梯度消失
                    #     "num_freqs_pos":10, #3
                    #     "num_freqs_time":4, #1
                    # },
                    # 5 min 没有梯度消失、但是效果很不好

                    # "hidden_features": 3*512,  #能够缓解梯度消失
                    # "hidden_layers":11, 
                    # "use_residual":False, 
                    # "posEnc":False,
                    # 能够正常拟合

                    # "hidden_features": 512, 
                    # "hidden_layers":16, 
                    # "use_residual":False, 
                    # "posEnc":False,
                    # 3 min, 能够正确拟合，没有梯度消失

                    # "hidden_features": 256, 
                    # "hidden_layers":64, 
                    # "use_residual":False, 
                    # "posEnc":False,
                    # 5 min, 不能正确拟合，出现了梯度消失

                    # "hidden_features": 256, 
                    # "hidden_layers": 64, 
                    # "use_residual": True, #残差机制避免了梯度下降、但是没能提高重构质量
                    # "posEnc": False,
                    # 5 min, 不能正确拟合，没有梯度消失

                    # "hidden_features": 2*256, 
                    # "hidden_layers": 16, 
                    # "use_residual": False, 
                    # "posEnc": False,
                    # recon_all=0.00029728 #能拟合，但拟合的不好

                    # "hidden_features": 2*256, 
                    # "hidden_layers": 16, 
                    # "use_residual": False, 
                    # "posEnc":{ #使用后对损失函数影响不大
                    #     "num_freqs_pos":10, #3
                    #     "num_freqs_time":4, #1
                    # },
                    # 3 min, recon_all=0.00027176

                    # "hidden_features": 4*256, #增加后反而无法拟合了
                    # "hidden_layers": 16, 
                    # "use_residual": False, 
                    # "posEnc":False, 
                    # recon_all=0.00259008 无法拟合


                    # "hidden_features": 15*256, 
                    # "hidden_layers": 12、2, 
                    # "use_residual": False, # 似乎还有负面作用
                    # "posEnc":{ # 无显著作用
                    #     "num_freqs_pos":10, #3
                    #     "num_freqs_time":4, #1
                    # },
                    # recon_all=0.0026 无法拟合,梯度消失


                    # "hidden_features": 7*256, 
                    # "hidden_layers": 2, 
                    # "use_residual": False, # 似乎还有负面作用
                    # "posEnc":{ # 无显著作用
                    #     "num_freqs_pos":10, #3
                    #     "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
                    # },
                    # 3 min, recon_all=0.00018120


                    # "hidden_features": 7*256, 
                    # "hidden_layers": 2, 
                    # "use_residual": False, # 似乎还有负面作用
                    # "posEnc":{ # 无显著作用
                    #     "num_freqs_pos":4, #3
                    #     "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
                    # },
                    # recon_all=0.00022805 #略有下降


                    # "hidden_features": 7*256, 
                    # "hidden_layers": 2, 
                    # "use_residual": False, # 似乎还有负面作用
                    # "posEnc":{ # 无显著作用
                    #     "num_freqs_pos":10, #3
                    #     "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
                    #     "APE":False, #自适应位置编码
                    # }, #recon_all=0.00019


                    # "hidden_features": 7*256, 
                    # "hidden_layers": 2, 
                    # "use_residual": False, # 似乎还有负面作用
                    # "posEnc":{ # 无显著作用
                    #     "num_freqs_pos":10, #3
                    #     "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
                    #     "APE":{#自适应位置编码
                    #         "total_steps":4000,
                    #         "warmup_steps":3000,
                    #     },
                    # }, 
                    # recon_all=0.00018008, APE没有显著作用


                    # "hidden_features": 7*256, 
                    # "hidden_layers": 2, 
                    # "use_residual": False, # 似乎还有负面作用
                    # "posEnc":{ # 无显著作用
                    #     "num_freqs_pos":10, #3
                    #     "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
                    #     "APE":False,
                    # }, 
                    # "gradualImageLayers":{
                    #     "warmup_steps":4000,
                    # },
                    # 20 min, recon_all=0.00017424

                    ################################## "NUM_fluid":1->4 ######################################
                    
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
                    # 20 min, recon_all=0.00010287
                    # 20 min, recon_all=0.00009287、recon_all=0.00010064
                    # enpoch=6000, 30 min, recon_all=0.00013577 #后期浮动

                    # "hidden_features": 7*256, 
                    # "hidden_layers": 2, 
                    # "use_residual": False, # 似乎还有负面作用
                    # "posEnc":False, 
                    # "gradualImageLayers":False,
                    # "use_maskP":False,
                    # 20 min, recon_all=0.00029

                    # "hidden_features": 7*256, 
                    # "hidden_layers": 2, 
                    # "use_residual": False, # 似乎还有负面作用
                    # "posEnc":{ # 有显著作用
                    #     "num_freqs_pos":10, #3
                    #     "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
                    #     "APE":False,
                    # }, 
                    # "gradualImageLayers":{
                    #     "warmup_steps":4000,
                    # },
                    # "use_maskP":False, #应该不会太好
                    # 21 min, recon_all=0.0002

                }, # 现在的首要问题是无损失地拟合出来视频

                # 2.损失函数
                "useSmooth":False, #不进行平滑约束
                "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
                "interval":0.1,#将计算平滑损失的步长由1改为0.5
                "lossType":2,
                "lossParam":{"rm":None,"ra":"F"},#最佳效果
                # "lossParam":{"rm":None,"ra":"f01"},
                "useMask":True, #只有lossType==1的时候才有效

                ########################
                "de-soft":None,
            },
            "name": "_019_02_updateConfig", #提高模型的拟合能力
            "precomputed": False,
            "input_mode": "A19-config.rigid.non1",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": True,#True,#False,
            "mergeMask": False,
        },
        # {# 复现8号实验的解耦效果
        #     # 刚体平滑滤波模块 #在纹理空间上进行平滑
        #     "decouple":{ # 解耦
        #         "tag":"A19-config", # 只兼容了startDecouple1
        #         "de-rigid":"1_sim", # 去噪框架
        #         "epoch":2000,          # 只兼容了startDecouple1
                
        #         # 1.模型本身
        #         # 1.1 刚体模块
        #         "NUM_rigid":1,#只有一个运动的刚体
        #         "configRigid":{ #单个刚体层的参数
        #             # 纹理 { 能够拟合单张纹理的神经网络: (2, 2*4*512, MSE) }
        #             'hidden_layers_map':2, #1, #2, #4, #32, #4,
        #             # 'hidden_features_map': 2*4*512, #2*4*512,#16*4*512,#128,#512, #128,
        #             'hidden_features_map': 2*4*512, #
        #             # 整体运动 #能够模拟大致数据
        #             'hidden_layers_global':4,
        #             'hidden_features_global':128,
        #             # 局部运动
        #             'hidden_layers_local':4,
        #             'hidden_features_local':128,
        #         },  
        #         "openLocalDeform":False,#True,#False,#True,
        #         "stillnessFristLayer":False,#True,#False,#并无意义，要和stillness保持一致
        #         "stillness":False,#True,#False,#不取消运动约束
        #         # 1.2 软体模块
        #         "NUM_soft":1,
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

        #         # 2.损失函数
        #         # "lossParam":{"rm":None,"ra":"R,S"},#最佳效果
        #         # "lossParam":{"rm":"S","ra":"R"},
        #         "lossParam":{"rm":"R,S","ra":None},
        #         "configRigids":{ # 整个刚体层模块的参数
        #             # 为啥要把重构损失放到这里
        #             "loss_recon_all_type":"MSE",#{"myLog" 学习能力不如MSE, "MSE", "atten_d"}
        #         }, 
        #         "useSmooth":False, #不进行平滑约束
        #         "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
        #         "interval":0.1,#将计算平滑损失的步长由1改为0.5
        #         "lossType":2,
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
    ]#不要重复做实验，要相信之前的结果
    #可以相信的东西：静态刚体的纹理、无局部刚体的运动
    #刚体的局部运动
    #软体使用刚体的全局运动
    [#没有进行的测试
        # {#测试不同的平滑权重的影响（因为最终收敛到0了(在2000epoch之内)，所以我感觉权重影响不大）
        #     "decouple":{#解耦
        #         ########################
        #         "de-rigid":"1",
        #         "epoch":1000,          #只兼容了startDecouple1
        #         "tag":"A-03-smooth-e1000",#只兼容了startDecouple1
        #         "useSmooth":True,
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_016_02_noRigid1(b4000)",
        #     "precomputed": False,
        #     "input_mode": "A-02-smooth-e4000.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,#True,#False,
        #     "mergeMask": False,
        # },
    ]


