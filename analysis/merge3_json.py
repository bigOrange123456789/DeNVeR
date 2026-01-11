# import os
import cv2
import numpy as np
from pathlib import Path
# import argparse
# import sys

        
if True:

    root_paths_config = [

        # [0]
        # "outputs/xca_dataset_sub3_copy/images",#原视频
        "outputs/xca_dataset_sub3_copy/ground_truth",#真值
        "outputs/xca_dataset_sub3_copy/images",#原视频
        "outputs/xca_dataset_sub2_result/_017_07_orig(sub2)",    
        {
            "type": "metric", 
            "paths": [
                "outputs/metric/"+
                "_017_07_orig(sub2)"
                +"-CATH"+"_results.xlsx",
            ]
        },   
             

        # [1]
        {
            "type": "div", 
            "name": "method3-method4", 
            "paths": [
                "outputs/xca_dataset_sub3_copy/images",
                "outputs/xca_dataset_sub3_inputs/_018_01",
            ]
        },
        "outputs/xca_dataset_sub3_inputs/_018_01",
        "outputs/xca_dataset_sub3_result/_018_01",   
        {
            "type": "metric", 
            "paths": [
                "outputs/metric/"+
                "_018_01"
                +"-CATH"+"_results.xlsx",
            ]
        },  

        # [2]
        # {
        #     "type": "div", 
        #     "name": "method3-method4", 
        #     "paths": [
        #         "outputs/xca_dataset_sub3_copy/images",
        #         "outputs/xca_dataset_sub3_inputs/_018_02_NumR1",
        #     ]
        # },
        "outputs/xca_dataset_sub3_decouple/A-stillness.rigid.main",
        "outputs/xca_dataset_sub3_inputs/_018_02_NumR1",
        "outputs/xca_dataset_sub3_result/_018_02_NumR1",  
        {
            "type": "metric", 
            "paths": [
                "outputs/metric/"+
                "_018_02_NumR1"
                +"-CATH"+"_results.xlsx",
            ]
        },


        # [3]
        "outputs/xca_dataset_sub3_decouple/A-still0-move1.rigid.main", #解耦噪声
        "outputs/xca_dataset_sub3_inputs/_018_03_stillFrist",   #去噪输入
        "outputs/xca_dataset_sub3_result/_018_03_stillFrist",   #分割结果
        {
            "type": "metric", 
            "paths": [
                "outputs/metric/"+
                "_018_03_stillFrist"
                +"-CATH"+"_results.xlsx",
            ]
        },

    ]
    for i in [
        ["A-e2e.rigid.main","_018_04_end2end"], #4
        ["A-e2e.recon","_018_05_end2endRecon"], #5
        ["Am-e2e.rigid.main","_018_06_end2end.m"], #6
        ["Am-e2e.recon","_018_07_end2endRecon.m"], #7
        ["Am-e2e.recon","_018_08_end2endRecon1.m"],#8
        ["Am-rlr.rigid.main","_018_09_reconLossRigid"], #9
        ["Am-rlr.rigid.main","_018_10_reconLossRigid"],#10
        ["Am-rlr.recon","_018_11_reconLossRigid_rs"],#11
        ["Am-rlr.recon","_018_12_reconLossRigid_rs2"],#12
        ["Am-loss2.rigid.main","_018_13_loss2"],#13
        ["Am-loss2.recon","_018_14_loss2_reon1"],#14
        ["Am-loss2-smooth.rigid.main","_018_15_loss2_smooth"],#15
        ["Am-smooth-9.rigid.main","_018_24_smooth"],#24
    ]:
        root_paths_config.append(
            "outputs/xca_dataset_sub3_decouple/"+i[0]
        )
        root_paths_config.append(
            "outputs/xca_dataset_sub3_inputs/"+i[1]
        )
        root_paths_config.append(
            "outputs/xca_dataset_sub3_result/"+i[1] 
        )
        root_paths_config.append(
            {
            "type": "metric", 
            "paths": [
                "outputs/metric/"+i[1]
                +"-CATH"+"_results.xlsx",
            ]
        },
        )


    # layout = [
    #     [4*0+0,4*4+0,4*13+0,4*15+0],  
    #     [4*0+1,4*4+1,4*13+1,4*15+1],  
    #     [4*0+2,4*4+2,4*13+2,4*15+2],  
    #     [4*0+3,4*4+3,4*13+3,4*15+3],  
    # ]


    layout = [
        [4*0+0,4*8+0],  
        [4*0+1,4*8+1],  
        [4*0+2,4*8+2],  
        [4*0+3,4*8+3],  
    ]

    # layout = [
    #     [4*0+0], 
    #     [4*8+1], 
    #     # [4*4+2],  
    # ]


