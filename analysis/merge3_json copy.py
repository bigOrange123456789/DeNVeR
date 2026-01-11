# import os
import cv2
import numpy as np
from pathlib import Path
# import argparse
# import sys

        

if True:

    root_paths_config = [

        # [0]
        "outputs/xca_dataset_sub4_copy/images",#原视频
        "outputs/xca_dataset_sub4_copy/ground_truth",#真值
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
        "outputs/xca_dataset_sub4_decouple/A19-best.rigid",
        "outputs/xca_dataset_sub4_inputs/_019_01_bestMetric",
        "outputs/xca_dataset_sub4_result/_019_01_bestMetric",   
        {
            "type": "metric", 
            "paths": [
                "outputs/metric/"+
                "_019_01_bestMetric"
                +"-CATH"+"_results.xlsx",
            ]
        },  


    ]
    # for i in [
    #     ["A-e2e.rigid.main","_018_04_end2end"], #4
    #     ["A-e2e.recon","_018_05_end2endRecon"], #5
    #     ["Am-e2e.rigid.main","_018_06_end2end.m"], #6
    #     ["Am-e2e.recon","_018_07_end2endRecon.m"], #7
    #     ["Am-e2e.recon","_018_08_end2endRecon1.m"],#8
    # ]:
    #     root_paths_config.append(
    #         "outputs/xca_dataset_sub3_decouple/"+i[0]
    #     )
    #     root_paths_config.append(
    #         "outputs/xca_dataset_sub3_inputs/"+i[1]
    #     )
    #     root_paths_config.append(
    #         "outputs/xca_dataset_sub3_result/"+i[1] 
    #     )
    #     root_paths_config.append(
    #         {
    #         "type": "metric", 
    #         "paths": [
    #             "outputs/metric/"+i[1]
    #             +"-CATH"+"_results.xlsx",
    #         ]
    #     },
    #     )

    
    layout = [
        [4*13+0,4*14+0],  
        [4*13+1,4*14+1],  
        [4*13+2,4*14+2],  
        [4*13+3,4*14+3],  
    ]
    layout = [
        [4*0+0,4*4+0,4*13+0,4*15+0],  
        [4*0+1,4*4+1,4*13+1,4*15+1],  
        [4*0+2,4*4+2,4*13+2,4*15+2],  
        [4*0+3,4*4+3,4*13+3,4*15+3],  
    ]
    
    layout = [
        [0,1,2,3,4,15],
        [5,6,7,8,9,16],
        [10,11,12,13,14,16]
    ]
    layout = [
        [0,10,6,16]
    ]
    for i in range(len(layout)):
        for j in range(len(layout[i])):
            # print(i,j,"layout[i,j]:",layout[i][j])
            layout[i][j] = layout[i][j]*4 #解耦噪声
    
    layout=[
        [0,4],
        [1,5],
        [2,6],
        [3,7],
    ]


