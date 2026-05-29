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
config_A26_03_01Q2   ***** #完全复现01I2
```

