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
                # {# 比020_01的指标低1个点
                #     "name": "_021_01",
                #     "color":"#E1602C",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/_021_01",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # {
                #     "name": "_021_02",
                #     "color":"#E2603C",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/_021_02",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # { #无显著变化
                #     "name": "A21-03(20-01)",
                #     "color":"#E3604C",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/A21-03(20-01)",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # { #无显著变化
                #     "name": "_021_04",
                #     "color":"#E3605C",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/_021_04",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # { 
                #     "name": "_021_05",
                #     "color":"#E4606C",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/_021_05",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # { 
                #     "name": "A21-06",
                #     "color":"#E5606D",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/A21-06",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # { 
                #     "name": "A21-07",
                #     "color":"#E6606E",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/A21-07",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # { 
                #     "name": "A21-08",
                #     "color":"#E7607E",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/A21-08",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # { 
                #     "name": "A21-09",
                #     "color":"#E7608E",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/A21-09",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # { 
                #     "name": "A21-10",
                #     "color":"#E8609E",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/A21-10",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # { 
                #     "name": "A21-11",
                #     "color":"#0960AE",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/A21-11",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # { 
                #     "name": "A21-11(1)",
                #     "color":"#0960AE",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/A21-11(1)",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # { 
                #     "name": "A21-11(2)",
                #     "color":"#0971AE",
                #     "gt_path":"outputs/xca_dataset_sub4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub4_result/A21-11(2)",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },

                ####################### xca_dataset_sim4(固定视频长度) ####################### 
                ##########################  DeNVeR.22   ##########################  
                # { 
                #     "name": "A22-03",
                #     "color":"#0911AE",
                #     "gt_path":"outputs/xca_dataset_sim4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sim4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sim4_result/A22-03",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # { 
                #     "name": "A22-04",
                #     "color":"#1911AE",
                #     "gt_path":"outputs/xca_dataset_sim4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sim4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sim4_result/A22-04",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # { 
                #     "name": "A22-05",
                #     "color":"#2911AE",
                #     "gt_path":"outputs/xca_dataset_sim4_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sim4_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sim4_result/A22-05",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },

                ####################### xca_dataset(原视频长度) ####################### 
                ##########################  DeNVeR.23   ##########################  
                { 
                    "name": "A23-01",
                    "color":"#29510E",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_result/A23-01",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                { 
                    "name": "A23-02",
                    "color":"#29612E",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_result/A23-02",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                { 
                    "name": "A23-03",
                    "color":"#29712E",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_result/A23-03",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                { 
                    "name": "A23-04",
                    "color":"#29713E",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_result/A23-04",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                { 
                    "name": "A23-05",
                    "color":"#29714E",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_result/A23-05",
                    "block_cath":False,
                    "threshold": 0.5,
                },

                { 
                    "name": "A23-06",
                    "color":"#29715E",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_result/A23-06",
                    "block_cath":False,
                    "threshold": 0.5,
                },

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
                    "name": "A23-11",
                    "color":"#2871AE",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_result/A23-11",
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
