'''
    内容：
        "ra":"MSE"
        "rm":"myLog"
        "rv":"myLog
    预测：
    结果：
        去噪结果看起来比较正常
    分析：
    实验设备: AutoDL_E、DeNVeR.26-3_new
    Running time: 约11 hours 
'''
config_A26_03_01O1={ # follow: config_A26_03_01O
            "decouple":{ # 解耦
                "tag":"A26-03-01O1",
                "de-rigid":"1_sim",#去噪框架
                #"total_steps":2000,#1000,#"epoch":1000,#2000,#2000,#6000,#4000,#2000, #只兼容了startDecouple1 #recon_all=0.00011
                "epochs":0.625,#
                "batch_size_scale":0.125,#1/8,#0.3,#0.35,#0.3,#0.5,#1/8,
                "dynamicVesselMask":{#有较长的时间开销
                    # "startStep":0.5*10, #False
                    # "intervalStep":1.5,
                    "startStep":0.5, #True
                    "intervalStep":0.2, #更新三次
                },
                # "dynamicVesselMask":False,
                "singleTrainVessel":False,#True, #是否单独增加在血管区域的训练次数
                "use_dynamicFeatureMask":True,#False,#True,
                "init_dynamicFeatureMask":{
                    "R":[0,1],#[运动，纹理]
                    "S":[1,1],
                    "F":[1,1],
                },#1, #遮挡向量的的初始值为1
                "quickUpdate_dynamicFeatureMask":False,#True,
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
                "lossParam":{ #这里有九种基础的组合, 全部算上有2^9种组合 
                    # "ra":"R,F", 
                    # "rm":"S", 
                    # "rv":None, 
                    "ra":"R", #"R,F", 
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
                    "rm":"myLog",#"MSE", #背景更清晰一些
                    "rv":"myLog",#"MSE",#"myLog",#"MSE", #更模糊一些 #myLog对于很暗的地方非常敏感
                    "rv_eps":0.5,#0,#0.1,#0.5,#0.1,#该参数的效果还没有被测试 #训练不足
                    "vesselMask_eps":1,#0.1,#0.25,
                }, 
                "UncertainLearning":{
                    "use":True,#False,#True,
                    "activationFunction":"sigmoid",#{None :不使用激活函数, "softplus": 软Relu ,"square" :平方, "sigmoid"}
                    "activationFunctionRadius":1, #只有当激活函数类型为sigmoid的时候才生效
                    "var_dias":0,#1,#默认为0
                    "weitht_all":2,#2,#默认为1
                    "weight_regular":{
                        "ra":1, #默认为1
                        "rm":1, #默认为1
                        "rv":1, #默认为1
                    },
                    "product_variance_type":"mul",#{"mul_err":最开始错误的版本，"mul","add"}
                },
                "maskPath_pathIn":None,#"A20-10-best1.rigid.non1", # 当"rm"==None的时候,没有用处 #是否使用预先计算好的MASK
                "useMask":True, #只有lossType==1的时候才有效
                ########################
                "de-soft":None,
                "saveTempImg":False,#True,
            },
            "name": "A26-03-01O1", #提高模型的拟合能力
            "precomputed": False,
            "noise_label":"A26-03-01O1.rigid",
            "input_mode": "A26-03-01O1.rigid.non1",
            "binarize": True,
            "inferenceAll": True,#False,
            "mergeMask": False,
        }
'''
                                    myLog                myLog            MSE
    Step[0300]: l=0.410082, recon_b=0.057028, recon_vess=0.062211, recon_all=0.163594,
    Step[0300]: l=0.368092, recon_b=0.071222, recon_vess=0.080404, recon_all=0.08907

recon_b_values = [
    0.050808, 0.054931, 0.055547, 0.050870, 0.060967, 0.057899, 0.051747,
    0.050878, 0.053766, 0.056933, 0.042089, 0.058778, 0.069543, 0.068365,
    0.051920, 0.052999, 0.043917, 0.051323, 0.049005, 0.051307, 0.049598,
    0.047299, 0.060759, 0.050832, 0.058286, 0.047279, 0.050525, 0.050230,
    0.048751, 0.046215, 0.051567, 0.046912, 0.048197, 0.048590, 0.047512,
    0.077183, 0.062050, 0.074694, 0.076883, 0.069747, 0.073817, 0.128388,
    0.101486, 0.100494, 0.094234, 0.091830, 0.080948, 0.100163, 0.110929,
    0.103284, 0.102785, 0.111534, 0.119064, 0.104711, 0.094853, 0.091810,
    0.109859, 0.118994, 0.107093, 0.084922, 0.102220, 0.086432, 0.067713,
    0.063898, 0.087333, 0.065861, 0.061413, 0.075777, 0.072321, 0.065067,
    0.078186, 0.083639, 0.078225, 0.080670, 0.077995, 0.087371, 0.070286,
    0.057028, 0.076726, 0.072670
]

'''