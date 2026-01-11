# import os
import cv2
import numpy as np
from pathlib import Path
# import argparse
# import sys

        

if True:

    root_paths_config = [

        # [0] #直接分割效果
        # "outputs/xca_dataset_sub4_copy/images",#原视频
        "outputs/xca_dataset_sub4_copy/ground_truth",#真值
        "outputs/xca_dataset_sub4_copy/images",#原视频
        "outputs/xca_dataset_sub2_result/_017_07_orig(sub2)",    
        {
            "type": "metric", 
            "paths": [
                "outputs/metric/"+
                "_017_07_orig(sub2)"
                +"-CATH"+"_results.xlsx",
            ]
        },   
             

        # [1] #最佳效果
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
    for i in [
        # "_019_01_bestMetric",
        "_020_01_baseline", # 2
        "_020_02_newSoft",  # 3
        "_020_03", # 4
        "_020_04",
        "_020_05",
        "_020_06",
        "_020_07",
    ]:
        root_paths_config.append(
            "outputs/xca_dataset_sub4_decouple/"+i
        )
        root_paths_config.append(
            "outputs/xca_dataset_sub4_inputs/"+i
        )
        root_paths_config.append(
            "outputs/xca_dataset_sub4_result/"+i 
        )
        root_paths_config.append(
            {
            "type": "metric", 
            "paths": [
                "outputs/metric/"+i
                +"-CATH"+"_results.xlsx",
            ]
        },
        )

    
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
            layout[i][j] = layout[i][j]*4 #解耦噪声


    layout=[]
    for j in range(4):
        row=[]
        for i in range(int(len(root_paths_config)/4)):
            row.append(
                4*i+j
            )
        layout.append(row)

    
    # layout=[
    #     [0,4],
    #     [1,5],
    #     [2,6],
    #     [3,7],
    # ]


