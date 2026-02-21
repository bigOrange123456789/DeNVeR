import os
# from confs.json import config_data_base #现在精力充沛，等之后比较困的时候再整理代码，将训练json和测试json合并到一起
config_data0 = {
            "experiments" : [

                { 
                    "name": "1.masks",
                    "color":"#28716E",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_result/1.masks",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                { 
                    "name": "2.2.planar",
                    "color":"#28717E",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_result/2.2.planar",
                    "block_cath":False,
                    "threshold": 0.5,
                },       
                { 
                    "name": "3.parallel",
                    "color":"#28718E",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_result/3.parallel",
                    "block_cath":False,
                    "threshold": 0.5,
                },    
                { 
                    "name": "4.deform",
                    "color":"#28719E",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_result/4.deform",
                    "block_cath":False,
                    "threshold": 0.5,
                },    
                { 
                    "name": "5.refine",
                    "color":"#2871AE",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_result/5.refine",
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
