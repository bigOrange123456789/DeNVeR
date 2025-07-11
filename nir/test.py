import numpy as np
import matplotlib.pyplot as plt
import math

# 定义函数
def custom_tanh(x):
    # return np.tan(1 - (2/math.pi) * x)
    # return np.tan((x-math.pi/2)*(2/math.pi) )
    return -np.tan( (math.pi/2) * (x +1) )


def binaryMask_error(x):#为了让MASK更接近二值图
    return 1-np.exp(-1/x)
def binaryMask(x):#为了让MASK更接近二值图
    k=2**64-2 #一个超参数
    k = 10 ** 10  # 一个无限大的超参数
    x = np.log(1+k*x)/np.log(1+k)
    x = np.log(1 + k * x) / np.log(1 + k)
    return x



def f(x):return binaryMask(x)

# 创建 x 值范围（避开正切函数的间断点）
x = np.linspace(-4, 5, 1000) #custom_tanh
x = np.linspace(1e-20, 1, 500) #binaryMask
# 计算 y 值
y = f(x)

# 设置间断点附近的阈值（正切函数在 π/2 + nπ 处有间断）
threshold = 10
# y = np.where(np.abs(y) > threshold, np.nan, y)

# 创建图形
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label=r'$y = \tan\left(1 - \frac{2}{\pi}x\right)$')

# 添加标题和标签
plt.title(r'Function Plot: $y = \tan\left(1 - \frac{2}{\pi}x\right)$', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# 标记关键点
critical_x = math.pi/2 * (1 - math.pi/2)/(2/math.pi)  # 计算第一个间断点
plt.axvline(critical_x, color='red', linestyle='--', alpha=0.5,
            label=f'Discontinuity at x ≈ {critical_x:.2f}')
plt.legend(loc='best')

# # 设置坐标轴范围
# plt.xlim(-4, 5)
# plt.ylim(-10, 10)

# 添加数学表达式
plt.text(0.5, 0.95, r'$\mathbf{y} = \tan\left(1 - \frac{2}{\pi}\mathbf{x}\right)$',
         transform=plt.gca().transAxes, fontsize=16, ha='center')

# 显示图像
plt.tight_layout()
plt.show()