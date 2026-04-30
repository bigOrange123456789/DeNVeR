import os
# from confs.json import config_data_base #现在精力充沛，等之后比较困的时候再整理代码，将训练json和测试json合并到一起
config_data0 = {
            "experiments" : [
                ####################### 改用sub4数据集 ####################### 
                ##########################  DeNVeR.019  ##########################                  
                # {# 原视频的分割效果
                #     "name": "_019_01_bestMetric",
                #     "color":"#70C00C",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/_019_01_bestMetric",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                ##########################  DeNVeR.020  ##########################  
                {# 
                    "name": "_020_01_baseline",
                    "color":"#80A00C",
                    "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub4_result/_020_01_baseline",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                # {# 
                #     "name": "_020_02_newSoft",
                #     "color":"#90C01C",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/_020_02_newSoft",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # {# 
                #     "name": "_020_03",
                #     "color":"#A0C02C",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/_020_03",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # {# 
                #     "name": "_020_04",
                #     "color":"#B0A02C",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/_020_04",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # {# 
                #     "name": "_020_05",
                #     "color":"#C0A00C",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/_020_05",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # {
                #     "name": "_020_06",
                #     "color":"#D0900C",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/_020_06",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # {
                #     "name": "_020_07",
                #     "color":"#E0803C",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/_020_07",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },

                # {
                #     "name": "_020_10",
                #     "color":"#F0702C",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/_020_10",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # {
                #     "name": "_020_11",
                #     "color":"#E0601C",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/_020_10",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },

                {
                    "name": "_021_01",
                    "color":"#E1602C",
                    "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub4_result/_021_01",
                    "block_cath":False,
                    "threshold": 0.5,
                },




            ],
            "usedVideoId":
                os.listdir("outputs/xca_dataset_sub4_copy/ground_truth"),
            "DataFiltering": 
                # "T",
                # "Move",
                None,
            "onlyVideoId":None,
            "onlyFrameId":None
        }

experiments=[]#{}
for e in config_data0["experiments"]:
    # if True:
    #     experiments.append(e)

    if True:
        e2=e.copy()
        e2["name"]=e["name"]+"-CATH"
        e2["block_cath"]=True
        experiments.append(e2)

    # if True:#不如0.8的效果
    #     e2=e.copy()
    #     e2["name"]=e["name"]+"-CATH"+"-t0.65"
    #     e2["block_cath"]=True
    #     e2["threshold"]=0.65
    #     experiments.append(e2)

    # if True:
    #     e2=e.copy()
    #     e2["name"]=e["name"]+"-CATH"+"-t0.8"
    #     e2["block_cath"]=True
    #     e2["threshold"]=0.8
    #     experiments.append(e2)
    
    if False:
        e2=e.copy()
        e2["name"]=e["name"]+"-noBinary"
        e2["threshold"]=None
        experiments.append(e2)

config_data={
    "experiments":experiments,
    "usedVideoId":config_data0["usedVideoId"],
    "DataFiltering":config_data0["DataFiltering"],
}
print("usedVideoId", config_data0["usedVideoId"])
