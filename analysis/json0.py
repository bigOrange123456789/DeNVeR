import os
# from confs.json import config_data_base #现在精力充沛，等之后比较困的时候再整理代码，将训练json和测试json合并到一起
config_data0 = {
            "experiments" : [
                ####################### 改用sub4数据集 ####################### 
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

                ##########################  DeNVeR.021  ##########################  
                {# 比020_01的指标低1个点
                    "name": "_021_01",
                    "color":"#E1602C",
                    "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub4_result/_021_01",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                {
                    "name": "_021_02",
                    "color":"#E2603C",
                    "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub4_result/_021_02",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                { #无显著变化
                    "name": "A21-03(20-01)",
                    "color":"#E3604C",
                    "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub4_result/A21-03(20-01)",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                { #无显著变化
                    "name": "_021_04",
                    "color":"#E3605C",
                    "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub4_result/_021_04",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                { 
                    "name": "_021_05",
                    "color":"#E4606C",
                    "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub4_result/_021_05",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                { 
                    "name": "A21-06",
                    "color":"#E5606D",
                    "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub4_result/A21-06",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                { 
                    "name": "A21-07",
                    "color":"#E6606E",
                    "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub4_result/A21-07",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                { 
                    "name": "A21-08",
                    "color":"#E7607E",
                    "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub4_result/A21-08",
                    "block_cath":False,
                    "threshold": 0.5,
                },




            ],
            "usedVideoId":
                [
                    # 'CVAI-1207LAO44_CRA29',
                    'CVAI-1253LAO0_CAU29',
                    # 'CVAI-2174LAO42_CRA18', 
                    # 'CVAI-2855LAO26_CRA31',
                ],
                # os.listdir("outputs/xca_dataset_sub4_copy/ground_truth"),
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
