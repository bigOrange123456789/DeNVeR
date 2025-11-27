import os
config_data0 = {
            "experiments" : [
                ##########################  DeNVeR.011  ##########################  
                # {
                #     "name":"_011_continuity_01",
                #     "color":"#2E7D32",
                #     "gt_path":"outputs/xca_dataset_sim2_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sim2_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sim2_result/_011_continuity_01",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                {
                    "name":"_011_continuity_01(orig)-temp",#(orig)
                    "color":"#C2185B",
                    "gt_path":"outputs/xca_dataset_sim2_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sim2_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sim2_result/_011_continuity_01-temp",
                    "block_cath":False,
                    "threshold": 0.5,
                },

                ####################### 改用原数据集的测试效果 ####################### 
                ##########################  DeNVeR.014  ##########################  
                {#整个数据集的结果和局部数据的结果保持一致，这不合理。
                 #这太诡异了，整体指标和局部的指标相同，好像数据内的信息都是均匀分布一样。我不相信有这种巧合。
                 #好吧，我相信了，这似乎就是巧合
                    "name":"noRigid1(all-dataset)", 
                    "color":"#F0D38C",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_result/noRigid1",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                ####################### 长视频子集合 xca_dataset_sub1 ####################### 
                #########################  DeNVeR.015  ##########################  
                {#epoch:4000
                    "name": "_015_01_noRigid1(b4000)",
                    "color":"#F0D3AC",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub1_result/_015_01_noRigid1",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                {#epoch:2000
                    "name": "_015_02_noRigid1(b2000)",
                    "color":"#F0D3BC",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub1_result/_015_02_noRigid1(b2000)",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                {#测试刚体去噪
                    "name": "_015_03_noRigid1(b1000)",
                    "color":"#F7C3CC",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub1_result/_015_03_noRigid1(b1000)",
                    "block_cath":False,
                    "threshold": 0.5,
                },
            ],
            "usedVideoId":
                os.listdir("outputs/xca_dataset_sub1_copy/ground_truth")
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
    "usedVideoId":config_data0["usedVideoId"]
}
print("usedVideoId", config_data0["usedVideoId"])
