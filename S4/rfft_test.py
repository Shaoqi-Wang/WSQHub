import torch
from opt_einsum import contract
from einops import rearrange
import numpy as np
import torch.nn.functional as F
from functools import partial
import torchvision
import math
from torch.fft import rfft, irfft
import matplotlib.pyplot as plt


def sin(x):
    return torch.sin(200 * x)


x_slice = range(1000)
pi = torch.pi
x = torch.linspace(0, 2 * pi, 1000)

omega = torch.tensor(
    np.exp(-1j * 2 * np.pi * 200 / 1000),
    dtype=torch.cfloat
)
omega = omega ** torch.arange(0, 1000)

y = sin(x)

print(torch.sum(y_mf := y * omega).abs())

y_f = rfft(y,n=500)
y_np = y_f.abs().numpy()
y_rec = irfft(y_f)

plt.scatter(x_slice, y.numpy(), color='g')
plt.scatter(x_slice, y_rec.numpy(), color='r')
plt.show()
