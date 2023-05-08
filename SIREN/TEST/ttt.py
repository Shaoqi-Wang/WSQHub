import numpy as np
import matplotlib.pyplot as plt
import torch

# 设置随机数种子以获得可重复的结果
# np.random.seed(42)

# 创建一个基于 [-1, 1] 的均匀分布抽样
num_samples = 256
# samples = np.random.uniform(-1, 1, num_samples)
samples = torch.rand([256, ]) * 2 - 1
samples = samples.numpy()

# 计算概率密度（直方图）
hist, bin_edges = np.histogram(samples, bins=1, density=True)

# 画出概率密度图像
print(bin_edges)
# plt.bar(bin_edges[:-1], hist, width=(bin_edges[1] - bin_edges[0]), color='blue', alpha=0.7)
plt.bar(bin_edges[:-1], height=hist, width=bin_edges[1] - bin_edges[0], align='edge')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Probability Density of Uniformly Distributed Samples')
plt.show()
