'''
    内容：
        指标的下降趋势不可阻挡，所以复现最佳方案进行测试
    目标：
        要和A26_03_01I2效果相同
    结果：
        ?
    分析一下：
        ?
    实验设备: 
        AutoDL_E、DeNVeR.26-3_new
    Running time: ?? hours 
'''
config_A26_03_01Q2={ # follow: config_A26_03_01I2  
            "decouple":{ # 解耦
                "tag":"A26-03-01Q2",
                "de-rigid":"1_sim",#去噪框架
                #"total_steps":2000,#1000,#"epoch":1000,#2000,#2000,#6000,#4000,#2000, #只兼容了startDecouple1 #recon_all=0.00011
                "epochs":0.625,#
                "batch_size_scale":1/8,#0.3,#0.35,#0.3,#0.5,#1/8,
                "dynamicVesselMask":{#有较长的时间开销
                    # "startStep":0.5*10, #False
                    # "intervalStep":1.5,
                    "startStep":0.5, #True
                    "intervalStep":0.2, #更新三次
                },
                # "dynamicVesselMask":False,
                "singleTrainVessel":False,#True, #是否单独增加在血管区域的训练次数
                "use_dynamicFeatureMask":True,#False,#True,
                "init_dynamicFeatureMask":1, #遮挡向量的的初始值为1
                "quickUpdate_dynamicFeatureMask":True,
                # 1 模型本身
                # 1.1 刚体模块
                "NUM_rigid":1,#只有一个运动的刚体
                "configRigids":{ # 整个刚体层模块的参数
                    "layer":{
                        "use_residual":{
                            "R":False,
                            "S":False,
                            "T":False,
                        },
                        # 整体运动
                        "useGlobal":True,#False,
                        'hidden_layers_global':2,#1,
                        'hidden_features_global':8*128,#1,
                        "globalMotionMode":2,#[6矩阵,4移动旋转放缩,3,2移动]
                        "use_rot":False, #"globalMotionMode"为3的时候才有效
                        "use_sca":False,
                        # 局部运动
                        "useLocal":False,
                        'hidden_layers_local':0,#1,
                        'hidden_features_local':0,#1,
                        # 纹理
                        "dynamicTex":False, #动态纹理
                        'hidden_layers_map':2,#1,#2,#4,#32,#4,
                        'hidden_features_map': 8*512,#256,#64,#8,#2*4*512,#16*4*512,#128,#512, #128,
                        "posEnc":False,
                        "use_featureMask":False,
                    },
                }, 
                "openLocalDeform":False, #True,
                "stillnessFristLayer":True,#False,#True,#:False, #True,#False,#并无意义，要和stillness保持一致
                # 1.2 软体模块
                "NUM_soft":1,
                "configSofts":{ # 软体
                    "layer":{
                        "use_residual":{
                            "R":False,
                            "S":False,
                            "L":False,
                            "T":False,
                        },
                        # 1.整体运动
                        "useGlobal":False,
                        'hidden_layers_global':0,#1,#2, 
                        'hidden_features_global':0,#1,#8*128, 
                        # 2.局部运动
                        "useLocal":True,#False, #True,
                        'hidden_layers_local':2,#1,#2,
                        'hidden_features_local':8*128,#1,#8*128, # Mask遮挡
                        # 3.纹理
                        "dynamicTex":True,#False, #动态纹理
                        'hidden_layers_map':4,#2,#4, # 1, # 2, # 4, # 32, # 4,
                        'hidden_features_map': 64,#4*512,#64,#8*512, # 将隐含层特征维度变为1/8
                        "posEnc":{ # 有显著作用
                            "num_freqs_pos":10, #3
                            "num_freqs_time":100, #4, #1 #后面要通过这里测试时序编码能否提升效果
                            "APE":False, #没有启用渐进式位置编码、启用不是改为True
                        }, # 频率是2的n次方，过大容易超出浮点数上限出现None。 # sin(2¹·π·x)  
                        "use_featureMask":False,#True, #渐进式遮挡向量
                        "fm_total_steps":800/2000, #use_featureMask=true的时候启用
                    },
                    "useSoftMask" : False, #无法生成有意义的MASK
                    "layerMask":{ #无效
                        "hidden_features": 64,#8,#256,#3*256,#7*256, 
                        "hidden_layers": 2, 
                        "use_residual": False, # 似乎还有负面作用
                        "posEnc":{ 
                            "num_freqs_pos":10, #3
                            "num_freqs_time":10, #4, #1 #后面要通过这里测试时序编码能否提升效果
                            "APE":False,
                        }, 
                        "gradualImageLayers":False,
                        "use_maskP":False,
                    },
                },
                # 1.3 流体模块
                "NUM_fluid":1, # 0.00019 -> 0.00016、0.00015
                "configFluids":{ #参数数量
                    "layer":{
                        "use_residual":{
                            "R":False,
                            "S":False,
                            "T":False,
                        },
                        # 整体运动
                        "useGlobal":False,
                        'hidden_layers_global':0, 
                        'hidden_features_global':0,
                        # 局部运动
                        "useLocal":False,
                        'hidden_layers_local':0,
                        'hidden_features_local':0,
                        # 纹理
                        "dynamicTex":True,#动态纹理 #用于兼容layer2类接口
                        "hidden_layers_map": 4, 
                        "hidden_features_map": 64,#8,#256,#3*256,#7*256, 
                        "posEnc":{ # 有显著作用
                            "num_freqs_pos":10, #3
                            "num_freqs_time":100,#*2,#5, #4, #1 #后面要通过这里测试时序编码能否提升效果
                            "APE":False, #没有启用渐进式位置编码、启用不是改为True
                        }, 
                        "use_featureMask":False,#True,
                        "fm_total_steps":800/2000, #use_featureMask=true的时候启用
                    },                    
                    #######################
                    "vesselMaskInference":True,#False,
                    "gradualImageLayers":False, #没啥用的功能
                    # "use_maskP":False, #自动学习MASK遮挡图、无效功能
                }, # 现在的首要问题是无损失地拟合出来视频
                # 2.损失函数
                "useSmooth":False, #不进行平滑约束
                "weight_smooth":0.1**7,#0.001,#0.1, #1,始终固定 #10,始终固定 #0.1,
                "weight_concise":0.25,#1,#5,#20,#1,#0.01,#0.00001,
                "weight_component": 1,#分量约束（子衰减量小于总衰减量=>子衰减结果大于总衰减结果）
                "interval":0.1,#将计算平滑损失的步长由1改为0.5
                "lossType":2,
                # "lossParam":{ 
                #     "ra":"R", 
                #     "rm":"S", #背景 #很奇怪、软体层为啥能看到血管
                #     "rv":"F", #前景
                #     }, 
                "lossParam":{ 
                    "ra":"R", 
                    "rm":"S", 
                    "rv":"F", 
                    }, 
                "lossParam_vessel":{ 
                    "ra":"F", 
                    "rm":None, 
                    "rv":None, 
                    }, 
                "lossFunType":{ #无法只拟合血管 #"MSE", "myLog", "atten_d"
                    "ra":"MSE",
                    "rm":"MSE", #背景更清晰一些
                    "rv":"myLog",#"MSE", #更模糊一些
                    "rv_eps":0.5,#0.1,#该参数的效果还没有被测试 #训练不足
                    "vesselMask_eps":1,#0.1,#0.25,
                }, 
                "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处 #是否使用预先计算好的MASK
                "useMask":True, #只有lossType==1的时候才有效
                ########################
                "de-soft":None,
                "saveTempImg":False,
            },
            "name": "A26-03-01Q2", #提高模型的拟合能力
            "precomputed": False,
            "noise_label":"A26-03-01Q2.rigid",
            "input_mode": "A26-03-01Q2.rigid.non1",
            # "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": True,#False,
            "mergeMask": False,
        }