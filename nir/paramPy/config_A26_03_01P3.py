'''
    内容：
    结果：
        指标没有任何变化
    分析：
        归一化代码应该不是我理解的那样
    实验设备: AutoDL_P、DeNVeR.26-3_new
    Running time: ?? hours 
'''
config_A26_03_01P3={ # follow: config_A26_03_01J2
            "name": "A26_03_01P3", #提高模型的拟合能力
            "precomputed": False,
            "noise_label":"A26_03_01P1.rigid",
            "input_mode": "A26_03_01P1.rigid.non1",
            "binarize": True,
            "inferenceAll": True,#False,
            "mergeMask": False,
            "normalization":"denoise"#"orig"#
        }