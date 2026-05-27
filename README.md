# Self-Supervised X-Ray Coronary Angiography Segmentation with Vessel-Aware Synthesis Learning (IEEE JBHI, 2026. 04)

[**📖 Paper**]((https://ieeexplore.ieee.org/document/11488688))

Shuang Liang<sup>1</sup> Zhicheng Liu<sup>1</sup> Guangyuan Liu<sup>1</sup> Tianliang Yao<sup>2</sup> Chunyi Yang<sup>1</sup> Peng Qi<sup>2,3✉</sup>


<sup>1</sup>School of Computer Science and Technology, Tongji University, Shanghai 200092, China; &emsp; 

<sup>2</sup>Department of Control Science and Engineering, College of Electronics and Information Engineering, and the Shanghai Institute of Intelligent Science and Technology, Tongji University, Shanghai 200092, China; &emsp; 

<sup>3</sup>State Key Laboratory of Cardiovascular Diseases and Medical Innovation Center, Shanghai East Hospital, School of Medicine, Tongji University 200092, Shanghai, China. &emsp; 


<sup>✉</sup> Corresponding Author. 

In this paper, we propose a novel self-supervised curvilinear object segmentation method that learns robust and distinctive features from fractals and unlabeled images.
## Usage



#### 1. Training scripts

```bash

python -m nir.inference 
```
<!-- python train_DA_contrast_liot_finalversion.py  -->

#### 2. Evaluation scripts


```bash
python -m analysis.ana
```
<!-- python test.py -->

#### 3. Trained models
Trained models can be downloaded from here. [[Xunlei Drive](https://pan.xunlei.com/s/VOrR3FfC61LLMqoTIomfFAPFA1?pwd=r9s8#)].   
<!-- Put the weights in the "logs/" directory.   -->

#### 4. Trained Data
Dataset can be down from here. [[Xunlei Drive](https://pan.xunlei.com/s/VOrR5RAQcjOPSwjwyLkGDVErA1?pwd=vb8g#)].   

#### 5. Historical codes
We have archived historical codes used for ablation experiments and other evaluation tests.
Historical codes can be down from here. [[Xunlei Drive](https://pan.xunlei.com/s/VOrQycW9Xuiq3U095r3LBFg3A1?pwd=k5w3#)] 

#### 6. Else
config_A26_03_01C #没软体
config_A26_03_01D #没软体和流体

config_A26_03_02G #去除自适应特征

config_A26_03_01H #自适应遮挡的初值为1
config_A26_03_01I #自适应遮挡的初值为1+快收敛(指标和G相似)
config_A26_03_01I2 #高迭代数    
  
config_A26_03_01Q #降低刚体运动的总复杂度   

## Future Work

-SSCVS will be continuously updated.

## Contact

Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- 2814255951@qq.com
- 2411497@tongji.edu.cn

## 📜Citation
If you find this work helpful for your project, please consider citing our paper.
```
@article{liang2026self,
  author={Liang, Shuang and Liu, Zhicheng and Liu, Guangyuan and Yao, Tianliang and Yang, Chunyi and Qi, Peng},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Self-Supervised X-Ray Coronary Angiography Segmentation with Vessel-Aware Synthesis Learning}, 
  year={2026},
  volume={},
  number={},
  pages={1-11}
}

```

This is our paper:
["Self-Supervised X-Ray Coronary Angiography Segmentation with Vessel-Aware Synthesis Learning", Accepted by JBHI 2026](https://ieeexplore.ieee.org/document/11488688)

This is our baseline work:
["FreeCOS: Self-Supervised Learning from Fractals and Unlabeled Images for Curvilinear Object Segmentation", Accepted by ICCV 2023](https://arxiv.org/abs/2307.07245)

