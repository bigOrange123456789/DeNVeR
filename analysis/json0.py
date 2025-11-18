config_data0 = {
            "experiments" : [
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
                # {
                #     "name":"_011_continuity_01-temp",#(orig)
                #     "color":"#C2185B",
                #     "gt_path":"outputs/xca_dataset_sim2_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sim2_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sim2_result/_011_continuity_01-temp",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },

                # {
                #     "name":"_011_continuity_02",
                #     "color":"g",
                #     "gt_path":"outputs/xca_dataset_sim2_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sim2_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sim2_result/_011_continuity_02",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # {
                #     "name":"_011_(noRigid1)_02-temp", #(noRigid1)去除静态噪声后的效果 
                #     "color":"g",
                #     "gt_path":"outputs/xca_dataset_sim2_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sim2_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sim2_result/_011_continuity_02-temp",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                ##########################  DeNVeR.012  ##########################  
                # {
                #     "name":"_012_(fluid)_01", #基于流体分割的效果 
                #     "color":"#FF7F8A",
                #     "gt_path":"outputs/xca_dataset_sim2_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sim2_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sim2_result/_012_continuity_01",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },
                # {
                #     "name":"_012_02_bigMaskFluid", #基于流体分割的效果 
                #     "color":"#FFA36C",
                #     "gt_path":"outputs/xca_dataset_sim2_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sim2_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sim2_result/_012_02_bigMaskFluid",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },

                ####################### 改用长视频子集合测试 ####################### 
                ##########################  DeNVeR.013  ##########################  
                {
                    "name":"_013_long01_noRigid1", #基于去刚效果的分割的效果 
                    "color":"#F0A36C",
                    "gt_path":"outputs/xca_dataset_sub1_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub1_result/_013_long01_noRigid1",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                {
                    "name":"_013_long02_bigMaskFluid", #基于大遮挡流体效果的分割的效果 
                    "color":"#F0B36C",
                    "gt_path":"outputs/xca_dataset_sub1_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub1_result/_013_long02_bigMaskFluid",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                {
                    "name":"_013_long03_smallMaskFluid", #基于小遮挡流体效果的分割的效果 
                    "color":"#F0C36C",
                    "gt_path":"outputs/xca_dataset_sub1_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub1_result/_013_long03_smallMaskFluid",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                {
                    "name":"_013_04_traditionalDSA", #基于传统DSA去噪方法的分割的效果 
                    "color":"#F0D36C",
                    "gt_path":"outputs/xca_dataset_sub1_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub1_result/_013_04_traditionalDSA",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                {
                    "name":"_013_05_orig", #使用没有去噪图像的分割的效果 
                    "color":"#F0D37C",
                    "gt_path":"outputs/xca_dataset_sub1_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sub1_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sub1_result/_013_05_orig",
                    "block_cath":False,
                    "threshold": 0.5,
                },
            ]
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
    "experiments":experiments
}