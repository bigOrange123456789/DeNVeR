config_data0 = {
            "experiments" : [
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
                #     "name":"orig",
                #     "color":"#00B8D4",
                #     "gt_path":"outputs/xca_dataset_sim2_copy/ground_truth",
                #     "cath_path":"outputs/xca_dataset_sim2_copy/ground_truth_CATH",
                #     "pred_path":"outputs/xca_dataset_sim2_result/orig",
                #     "block_cath":False,
                #     "threshold": 0.5,
                # },

                {
                    "name":"_011_continuity_01",
                    "color":"#2E7D32",
                    "gt_path":"outputs/xca_dataset_sim2_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sim2_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sim2_result/_011_continuity_01",
                    "block_cath":False,
                    "threshold": 0.5,
                },
                {
                    "name":"_011_continuity_01-orig",
                    "color":"#C2185B",
                    "gt_path":"outputs/xca_dataset_sim2_copy/ground_truth",
                    "cath_path":"outputs/xca_dataset_sim2_copy/ground_truth_CATH",
                    "pred_path":"outputs/xca_dataset_sim2_result/_011_continuity_01-orig",
                    "block_cath":False,
                    "threshold": 0.5,
                },


            ]
        }
experiments=[]#{}
for e in config_data0["experiments"]:
    if False:
        experiments.append(e)

    if True:
        e2=e.copy()
        e2["name"]=e["name"]+"-CATH"
        e2["block_cath"]=True
        experiments.append(e2)

    if False:
        e2=e.copy()
        e2["name"]=e["name"]+"-CATH"+"-t0.65"
        e2["block_cath"]=True
        e2["threshold"]=0.65
        experiments.append(e2)
    
    if False:
        e2=e.copy()
        e2["name"]=e["name"]+"-noBinary"
        e2["threshold"]=None
        experiments.append(e2)



config_data={
    "experiments":experiments
}