import copy

import torch
import numpy as np
from opt_einsum import contract
from einops import rearrange, repeat, reduce
from torch import einsum
from torch.fft import rfft, irfft
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.nn import MSELoss
import matplotlib.pyplot as plt

plt.plot([[1, 5], [2, 3], [3, 4], [4, 5], [6, 7]], label=['123', '321'])
plt.show()
plt.legend()
