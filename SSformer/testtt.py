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

a = torch.randn(2, 2)
_c2r = torch.view_as_real
_r2c = torch.view_as_complex
_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
_resolve_conj = lambda x: x.conj().resolve_conj()

b = a.flatten()
c = b**2

print(a.shape)
print(b.shape)

print(b)
print(c)