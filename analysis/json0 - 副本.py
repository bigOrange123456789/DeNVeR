import os
config_data0 = {
            "experiments" : [
                ######################### 短视频的测试结果 ######################### 
                ##########################  DeNVeR.010  ##########################  
                # {
                #     "name":"1.masks",
                #     "color":"#1E88E5",
                #     "gt_path":"outputs/xca_dataset_sim2_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sim2_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sim2_result/1.masks",
                #     "block_cath":False,
                #     # "binarize": True,
                #     "threshold": 0.5,
                # },
                # {
                #     "name":"2.2.planar",
                #     "color":"#38B2AC",
                #     "gt_path":"outputs/xca_dataset_sim2_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sim2_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sim2_result/2.2.planar",
                #     "block_cath":False,
                #     # "binarize": True,
                #     "threshold": 0.5,
                # },
                # {
                #     "name":"3.parallel",
                #     "color":"#ED8936",
                #     "gt_path":"outputs/xca_dataset_sim2_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sim2_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sim2_result/3.parallel",
                #     "block_cath":False,
                #     # "binarize": True,
                #     "threshold": 0.5,
                # },
                # {
                #     "name":"4.deform",
                #     "color":"#9C27B0",
                #     "gt_path":"outputs/xca_dataset_sim2_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sim2_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sim2_result/4.deform",
                #     "block_cath":False,
                #     # "binarize": True,
                #     "threshold": 0.5,
                # },
                # {
                #     "name":"5.refine",
                #     "color":"#E53E3E",
                #     "gt_path":"outputs/xca_dataset_sim2_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sim2_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sim2_result/5.refine",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # {
                #     "name":"orig", #与视频长短无关
                #     "color":"#00B8D4",
                #     "gt_path":"outputs/xca_dataset_sim2_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sim2_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sim2_result/orig",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },

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

                # {
                #     "name":"_011_continuity_02",
                #     "color":"g",
                #     "gt_path":"outputs/xca_dataset_sim2_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sim2_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sim2_result/_011_continuity_02",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                {#短视频上
                    "name":"_011_(noRigid1.short)_02-temp", #(noRigid1)去除静态噪声后的效果 
                    "color":"g",
                    "gt_path":"outputs/xca_dataset_sim2_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sim2_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sim2_result/_011_continuity_02-temp",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                ##########################  DeNVeR.012  ##########################  
                {
                    "name":"_012_(fluid2.short)_01", #基于流体分割的效果 
                    "color":"#FF7F8A",
                    "gt_path":"outputs/xca_dataset_sim2_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sim2_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sim2_result/_012_continuity_01",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                # {
                #     "name":"_012_02_bigMaskFluid", #基于流体分割的效果 
                #     "color":"#FFA36C",
                #     "gt_path":"outputs/xca_dataset_sim2_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sim2_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sim2_result/_012_02_bigMaskFluid",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },

                ####################### 改用长视频子集合测试(没有相机移动) ####################### 
                ##########################  DeNVeR.013  ##########################  
                {
                    "name":"noRigid1(sub)", #基于去刚效果的分割的效果 
                    "color":"#F0A36C",
                    "gt_path":"outputs/xca_dataset_sub1_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub1_result/_013_long01_noRigid1",
                    "block_cath":False,
                    "threshold": 0.5,
                }, 
                # {
                #     "name":"_013_long02_bigMaskFluid", #基于大遮挡流体效果的分割的效果 
                #     "color":"#F0B36C",
                #     "gt_path":"outputs/xca_dataset_sub1_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub1_result/_013_long02_bigMaskFluid",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                {
                    "name":"fluid2(sub)", #基于小遮挡流体效果的分割的效果 
                    "color":"#F0C36C",
                    "gt_path":"outputs/xca_dataset_sub1_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub1_result/_013_long03_smallMaskFluid",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                # {
                #     "name":"_013_04_traditionalDSA", #基于传统DSA去噪方法的分割的效果 
                #     "color":"#F0D36C",
                #     "gt_path":"outputs/xca_dataset_sub1_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub1_result/_013_04_traditionalDSA",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # {
                #     "name":"_013_05_orig", #使用没有去噪图像的分割的效果 
                #     "color":"#F0D37C",
                #     "gt_path":"outputs/xca_dataset_sub1_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub1_result/_013_05_orig",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
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
                {
                    "name":"fluid2(all-dataset)", 
                    "color":"#F0D39C",
                    "gt_path":"outputs/xca_dataset_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_result/fluid2",
                    "block_cath":False,
                    "threshold": 0.5,
                },
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
                    "gt_path":"outputs/xca_dataset_sub1_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub1_result/_015_02_noRigid1(b2000)",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                {#epoch:1000
                    "name": "_015_03_noRigid1(b1000)",
                    "color":"#F7C3CC",
                    "gt_path":"outputs/xca_dataset_sub1_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub1_result/_015_03_noRigid1(b1000)",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                {#epoch:500
                    "name": "_015_04_noRigid1(b500)",
                    "color":"#A7C3DC",
                    "gt_path":"outputs/xca_dataset_sub1_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub1_result/_015_04_noRigid1(b500)",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                #########################  DeNVeR.016  ##########################  
                {#epoch:1000
                    "name": "_016_01_noRigid1(b1000)[smooth]",
                    "color":"#97C3EC",
                    "gt_path":"outputs/xca_dataset_sub1_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub1_result/_016_01_noRigid1(b1000)",
                    "block_cath":False,
                    "threshold": 0.5,
                },
            ],
            "usedVideoId":
                None#os.listdir("outputs/xca_dataset_sub1_copy/ground_truth")
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
