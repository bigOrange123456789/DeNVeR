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
????

#### 4. Trained Data
????

#### 5. Historical codes
????

#### 6. log
```bash
config_A26_03_01D    *72.7 #没软体和流体
config_A26_03_01C    *78.9 #没软体


config_A26_03_02G    *77.8 #去除自适应特征

config_A26_03_01H    *77.9 #自适应遮挡的初值为1
config_A26_03_01I    *78.7 #自适应遮挡的初值为1+快收敛(指标和G相似)

config_A26_03_01I2   *79.5 #高迭代数    
config_A26_03_01I2_2 *77.4 #减少血管MASK更新次数
config_A26_03_01I2_3 *78.9 #1I2的消融(快更新)
config_A26_03_01I2_4 *78.8 #1I2的消融(流体)
config_A26_03_01I2_5 *78.9 #1I2的消融(软体)
config_A26_03_01I2_6 ***** #1I2的消融(自适应模块)
config_A26_03_01I2_7 ***** #1I2的消融(sharp遮挡向量)
config_A26_03_01I2_8 ***** #1I2的消融(流体+软体)
config_A26_03_01I2_9 ***** #1I2的消融(分割先验模块) #初始分割先验+动态分割先验
config_A26_03_01I2_10 ***** #1I2的消融(静态先验) 
config_A26_03_01I2_11 ***** #1I2的消融(logMSE)
  
config_A26_03_01Q    *79.1 #降低刚体运动的总复杂度   
config_A26_03_01Q1   *78.8 #流体层只关注血管
config_A26_03_01Q2   *79.3 #复现 I2
config_A26_03_01Q3   *79.4 #流体网络宽度加倍
config_A26_03_01Q4   *79.4 #增加流体层数
config_A26_03_01Q5   *79.5 #关闭软体运动网络
config_A26_03_01Q6   *79.4 #流体训练不使用MASK
config_A26_03_01Q7   *78.3 #软体训练二值化遮挡
config_A26_03_01Q8   *78.1 #去除detach
config_A26_03_01Q9   *78.1 #myLog=>MSE
config_A26_03_01Q10  *79.4 #ra:"R,S"
config_A26_03_01Q11  *79.5 #关闭'动态MASK'
config_A26_03_01R    *78.8 #降低刚体运动的总复杂度(新)
config_A26_03_01R1   *78.2 #软体消融

config_A26_03_01R2   *79.1 #刚体运动复杂度加4倍
config_A26_03_01R3   *79.3 #关闭“快收敛”
config_A26_03_01R4   *79.3 #关闭“自适应调整”
config_A26_03_01R5   *78.7 #关闭软体
config_A26_03_01R6   *78.7 #关闭流体
config_A26_03_01R7   *79.0 #关闭刚体运动

config_A26_03_01R8   *79.4 #刚体运动复杂度加8倍
config_A26_03_01R9   *79.3 #关闭“快收敛”
config_A26_03_01R10  *79.3 #关闭“自适应调整”
config_A26_03_01R11  *78.6 #关闭软体
config_A26_03_01R12  *78.6 #关闭流体

config_A26_03_01R13  *78.3 #刚体运动复杂度加16倍
config_A26_03_01R14  *78.7 #1R13的消融(关闭“快收敛”)
config_A26_03_01R15  *79.0 #1R13的消融(关闭软体)
```

