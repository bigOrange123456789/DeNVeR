import os
# from confs.json import config_data_base #现在精力充沛，等之后比较困的时候再整理代码，将训练json和测试json合并到一起
config_data0 = {
            "experiments" : [
                ####################### 长视频子集合 xca_dataset_sub1 ####################### 
                # {
                #     "name":"_013_05_orig", #使用没有去噪图像的分割的效果 
                #     "color":"#F0D37C",
                #     "gt_path":"outputs/xca_dataset_sub1_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub1_result/_013_05_orig",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                #########################  DeNVeR.015  ##########################  
                # {#epoch:1000
                #     "name": "_015_03_noRigid1(b1000)",
                #     "color":"#F7C3CC",
                #     "gt_path":"outputs/xca_dataset_sub1_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub1_result/_015_03_noRigid1(b1000)",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                #########################  DeNVeR.016  ##########################  
                # {#smooth
                #     "name": "_016_01_noRigid1(b1000)[smooth]",
                #     "color":"#97C3EC",
                #     "gt_path":"outputs/xca_dataset_sub1_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub1_result/_016_01_noRigid1(b1000)",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # {#smooth 使用全部刚体
                #     "name": "_016_01_noAllRigid1(b1000)[smooth]",
                #     "color":"#97C3EC",
                #     "gt_path":"outputs/xca_dataset_sub1_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub1_result/_016_01_noRigidAll1(b1000)",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # {#smooth、epoch4000
                #     "name": "_016_02_noRigid1(b4000)[smooth]",
                #     "color":"#97B3FC",
                #     "gt_path":"outputs/xca_dataset_sub1_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub1_result/_016_02_noRigid1(b4000)",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                ####################### 改用sub2数据集 ####################### 
                ##########################  DeNVeR.017  ##########################  
                
                {# 有平滑
                    "name": "_017_01_nr(b2000)smooth",
                    "color":"#97B3FC",
                    "gt_path":"outputs/xca_dataset_sub2_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub2_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub2_result/_017_01_nr(b2000)smooth",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                # {# 无平滑
                #     "name": "_017_02_nr(b2000)",
                #     "color":"#97C3EC",
                #     "gt_path":"outputs/xca_dataset_sub2_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub2_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub2_result/_017_02_nr(b2000)",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                
                # {# 平滑+局部形变
                #     "name": "_017_03_nr(smooth.localDeform)",
                #     "color":"#87C3FC",
                #     "gt_path":"outputs/xca_dataset_sub2_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sub2_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sub2_result/_017_03_nr(smooth.localDeform)",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },

                {# 缩小平滑损失
                    "name": "_017_04_nr(smooth0.1)",
                    "color":"#87C90C",
                    "gt_path":"outputs/xca_dataset_sub2_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub2_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub2_result/_017_04_nr(smooth0.1)",
                    "block_cath":False,
                    "threshold": 0.5,
                },
            ],
            "usedVideoId":
                os.listdir("outputs/xca_dataset_sub2_copy/ground_truth"),
            "DataFiltering": 
                "T",
                # "Move",
                # None,
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
