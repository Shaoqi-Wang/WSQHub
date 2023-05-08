"""
此文件复现了论文 “Implicit Neural Representations with Periodic Activation Functions”
中关于arcsin分布与正态分布、均匀分布之间的近似关系

并检验了大数定理，w_l * X_l 是否服从标准正态分布，其中w_l和X_l按照论文要求初始化
"""

import torch
import torch.nn as nn
import numpy as np
from numpy import arcsin, sqrt, pi, arccos
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, arcsine, norm, uniform


def exp1():
    def arcsin_sqrt(x):
        return arcsin(sqrt(x))

    def arcsin_2x1(x):
        # return 0.5 * arcsin(2 * x - 1) + 0.25 * pi
        return 0.5 * arccos(1 - 2 * x)

    x = np.linspace(-1, 1, 200)

    standard_norm = norm(loc=0, scale=1)

    y_1 = standard_norm.cdf(3) - standard_norm.cdf(2 - 2 / pi * arcsin(x)) + \
          standard_norm.cdf(2 / pi * arcsin(x)) - standard_norm.cdf(-2 / pi * arcsin(x) - 2)

    y_2 = 1 / pi * arcsin(x) + 0.5

    plt.plot(y_1, label='arcsin_sqrt')
    plt.plot(y_2, label='arcsin_2x1')
    plt.legend()
    plt.grid()
    plt.show()


def exp2():
    # 设置arcsin分布的参数
    a, b = -1, 1
    loc, scale = 0, 1

    # 生成100个arcsin分布随机样本
    samples_arcsine = arcsine.rvs(loc=-1, scale=2, size=10000)
    # 生成100个arcsin分布随机样本
    samples_norm = norm.rvs(loc=0, scale=1, size=10000)
    # 生成100个arcsin分布随机样本
    samples_uniform = uniform.rvs(loc=-1, scale=2, size=10000)

    # 绘制直方图
    plt.hist(samples_arcsine, bins=100, density=True)
    plt.hist(samples_norm, bins=100, density=True)
    plt.hist(samples_uniform, bins=100, density=True)
    plt.xlabel('x')
    plt.ylabel('Frequency')
    plt.show()

    fan_in = 128
    fan_out = 128
    W = np.stack([uniform.rvs(loc=-sqrt(6 / fan_in), scale=2 * sqrt(6 / fan_in), size=fan_in)
                  for _ in range(fan_out)])
    h_input = arcsine.rvs(loc=-1, scale=2, size=fan_in)
    h_output = W @ h_input

    # 绘制直方图
    plt.hist(h_input, bins=10, density=True, label='h_input')
    plt.hist(W[0], bins=10, density=True, label='W[0]')
    plt.hist(h_output, bins=10, density=True, label='h_output')
    plt.xlabel('x')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid()
    plt.show()
