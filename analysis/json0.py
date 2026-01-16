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

                ####################### xca_dataset(原视频长度) ####################### 
                ##########################  DeNVeR.23   ##########################  
                # { 
                #     "name": "A23-01",
                #     "color":"#29510E",
                #     "gt_path":"outputs/xca_dataset_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_result/A23-01",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # { 
                #     "name": "A23-02",
                #     "color":"#29612E",
                #     "gt_path":"outputs/xca_dataset_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_result/A23-02",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # { 
                #     "name": "A23-03",
                #     "color":"#29712E",
                #     "gt_path":"outputs/xca_dataset_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_result/A23-03",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # { 
                #     "name": "A23-04",
                #     "color":"#29713E",
                #     "gt_path":"outputs/xca_dataset_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_result/A23-04",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # { 
                #     "name": "A23-05",
                #     "color":"#29714E",
                #     "gt_path":"outputs/xca_dataset_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_result/A23-05",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },

                # { 
                #     "name": "A23-06",
                #     "color":"#29715E",
                #     "gt_path":"outputs/xca_dataset_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_result/A23-06",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },

                { 
                    "name": "A23-07",
                    "color":"#28716E",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_result/A23-07",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                { 
                    "name": "A23-08",
                    "color":"#28717E",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_result/A23-08",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                { 
                    "name": "A23-09",
                    "color":"#28718E",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_result/A23-09",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                { 
                    "name": "A23-10",
                    "color":"#28719E",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_result/A23-10",
                    "block_cath":False,
                    "threshold": 0.5,
                },

                { 
                    "name": "A23-11",
                    "color":"#2871AE",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_result/A23-11",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                { 
                    "name": "A23-12",
                    "color":"#2871BE",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_result/A23-12",
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
