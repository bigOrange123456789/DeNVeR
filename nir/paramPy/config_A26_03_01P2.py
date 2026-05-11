'''
    内容：
        复现config_A26_03_01J2,应该达到myLastMethod
    结果：
    分析：
    实验设备: AutoDL_O、DeNVeR.26-3_new
    Running time: --- hours
'''
config_A26_03_01P2={ # follow: config_A26_03_01J2
            "name": "A26_03_01P2", #注意这里名称中使用的是下划线
            "precomputed": False,
            "noise_label":"A26_03_01P.rigid",
            "input_mode": "A26_03_01P.rigid.non1",
            "binarize": True,
            "inferenceAll": True,#False,
            "mergeMask": False,
            "normalization":"denoise"#"orig"#
        }