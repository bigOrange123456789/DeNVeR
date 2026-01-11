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
    configs = [
        
        {#刚体层完全静止 2Rt
            "decouple":{#解耦
                ########################
                "de-rigid":"1_sim",
                "epoch":2000,          #只兼容了startDecouple1
                "tag":"A-stillness2",#只兼容了startDecouple1
                "useSmooth":False,
                "stillness":True,#刚体层完全静止
                ########################
                "de-soft":None,
            },
            "name": "_018_01",
            "precomputed": False,
            "input_mode": "A-stillness2.rigid.main_non1",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": False,#True,#False,
            "mergeMask": False,
        },
        {#刚体层完全静止并且只有一个刚体层(查全率下降) #1Rt
            "decouple":{#解耦
                "de-rigid":"1_sim",
                "epoch":2000,          #只兼容了startDecouple1
                "tag":"A-stillness1",#只兼容了startDecouple1
                "useSmooth":False,
                "stillness":True,#刚体层完全静止
                # "NUM_soft":0,
                "NUM_rigid":1,
                ########################
                "de-soft":None,
            },
            "name": "_018_02_NumR1",
            "precomputed": False,
            "input_mode": "A-stillness1.rigid.main_non1",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": True,#True,#False,
            "mergeMask": False,
        },
        {#1层刚体层（有整体运动、无局部运动） #1Rm 
            "decouple":{#解耦
                "de-rigid":"1_sim",
                "epoch":2000,          #只兼容了startDecouple1
                "tag":"A-still0-move1",#只兼容了startDecouple1
                "useSmooth":False,
                "stillnessFristLayer":False,
                "stillness":True,#刚体层完全静止
                # "NUM_soft":0,
                "NUM_rigid":1,#只有一个运动的刚体
                ########################
                "de-soft":None,
            },
            "name": "_018_03_stillFrist",
            "precomputed": False,
            "input_mode": "A-still0-move1.rigid.main_non1",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": True,#True,#False,
            "mergeMask": False,
        },
        {#刚体层 有整体运动的单层 ） #1Rm 
            "decouple":{ #解耦
                "de-rigid":"1_sim",
                "epoch":2000,          #只兼容了startDecouple1
                "tag":"A-still0-move1",#只兼容了startDecouple1
                "useSmooth":False,
                "stillnessFristLayer":False,
                "stillness":False,#
                # "NUM_soft":0,
                "NUM_rigid":1,#只有一个运动的刚体
                ########################
                "de-soft":None,
            },
            "name": "_018_03_stillFrist",
            "precomputed": False,
            "input_mode": "A-still0-move1.rigid.main_non1",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": True,#True,#False,
            "mergeMask": False,
        },
        {#刚体层+软体层(基于刚体) #只有整体运动 # 1Rm+1S -> non_rigid
            "decouple":{#解耦
                "de-rigid":"1_sim",
                "epoch":2000,          #只兼容了startDecouple1
                "tag":"A-e2e",#只兼容了startDecouple1
                "useSmooth":False,#不平滑
                "stillnessFristLayer":False,#并无意义，要和stillness保持一致
                "stillness":False,#不取消运动约束
                "NUM_soft":1,
                "NUM_rigid":1,#只有一个运动的刚体
                ########################
                "de-soft":None,
            },
            "name": "_018_04_end2end",
            "precomputed": False,
            "input_mode": "A-e2e.rigid.main_non2",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": True,#True,#False,
            "mergeMask": False,
        },
        {#刚体层+软体层(基于重构) #只有整体运动 # 1Rm+1S -> non_recon
            "name": "_018_05_end2endRecon",
            "precomputed": False,
            "input_mode": "A-e2e.recon_non2",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": True,#True,#False,
            "mergeMask": False,
        },
        {#使用MASK # M(1Rm+1S) -> non_rigid
            "decouple":{ # 解耦
                "de-rigid":"1_sim",
                "epoch":2000,          #只兼容了startDecouple1
                "tag":"Am-e2e",#只兼容了startDecouple1
                "useSmooth":False,#不平滑
                "stillnessFristLayer":False,#并无意义，要和stillness保持一致
                "stillness":False,#不取消运动约束
                "NUM_soft":1,
                "NUM_rigid":1,#只有一个运动的刚体
                "useMask":True,
                ########################
                "de-soft":None,
            },
            "name": "_018_06_end2end.m",
            "precomputed": False,
            "input_mode": "Am-e2e.rigid.main_non2",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": True,#True,#False,
            "mergeMask": False,
        },
        {#刚体层+软体层(基于重构)+使用MASK # M(1Rm+1S) -> non_recon2
            "name": "_018_07_end2endRecon.m",
            "precomputed": False,
            "input_mode": "Am-e2e.recon_non2",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": True,#True,#False,
            "mergeMask": False,
        },
        {#刚体层+软体层(基于重构)+使用MASK # M(1Rm+1S) -> non_recon2
            "name": "_018_08_end2endRecon1.m",
            "precomputed": False,
            "input_mode": "Am-e2e.recon_non",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": True,#True,#False,
            "mergeMask": False,
        },
        {#双重构损失：总重构损失 + 刚体层重构损失
            "decouple":{#解耦
                "de-rigid":"1_sim",
                "epoch":2000,          #只兼容了startDecouple1
                "tag":"Am-rlr",#只兼容了startDecouple1
                "useSmooth":False,#不平滑
                "stillnessFristLayer":False,#并无意义，要和stillness保持一致
                "stillness":False,#不取消运动约束
                "NUM_soft":1,
                "NUM_rigid":1,#只有一个运动的刚体
                "useMask":True,
                "openReconLoss_rigid":True,
                ########################
                "de-soft":None,
            },
            "name": "_018_09_reconLossRigid",
            "precomputed": False,
            "input_mode": "Am-rlr.rigid.main_non2",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": True,#True,#False,
            "mergeMask": False,
        },
        {
            "name": "_018_10_reconLossRigid",
            "precomputed": False,
            "input_mode": "Am-rlr.rigid.main_non1",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": True,#True,#False,
            "mergeMask": False,
        },
        {
            "name": "_018_11_reconLossRigid_rs",
            "precomputed": False,
            "input_mode": "Am-rlr.recon_non",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": True,#True,#False,
            "mergeMask": False,
        },
        {
            "name": "_018_12_reconLossRigid_rs2",
            "precomputed": False,
            "input_mode": "Am-rlr.recon_non2",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": True,#True,#False,
            "mergeMask": False,
        },
    ]



