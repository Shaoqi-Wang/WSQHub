import numpy as np
from scipy.stats import arcsine, norm, uniform
import matplotlib.pyplot as plt

# 生成随机序列数据
N = 1000

x = np.concatenate([np.zeros(200), np.ones(600,), np.zeros(200)])

# 将序列数据表示为列向量
x_col = x.reshape(-1, 1)

# 对列向量进行傅里叶变换，得到频率谱特征
X = np.fft.rfft(x_col, axis=0) / N
freqs = np.fft.rfftfreq(N, d=1.0 / N)  # 计算对应的频率值

# 可视化时域信号
plt.subplot(2, 1, 1)  # 子图1，显示时域信号
plt.plot(x)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Time-domain signal')

# 可视化频域信号
plt.subplot(2, 1, 2)  # 子图2，显示频域信号
plt.plot(freqs, np.abs(X))
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.title('Frequency-domain signal')

plt.show()
