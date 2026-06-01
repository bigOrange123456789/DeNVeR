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
config_A26_03_01Q11  ***** #关闭'动态MASK'
config_A26_03_01R    ***** #降低刚体运动的总复杂度(新)
config_A26_03_01R1   ***** #软体消融
```

