import torch
from torch import nn
# from torchmeta.modules import (MetaModule, MetaSequential)
# from torchmeta.modules.utils import get_subdict
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.utils import weight_norm as w_norm
import re


def get_subdict(dictionary, key=None):
    if dictionary is None:
        return None
    if (key is None) or (key == ''):
        return dictionary
    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
    return OrderedDict((key_re.sub(r'\1', k), value) for (k, value)
                       in dictionary.items() if key_re.match(k) is not None)


class MetaModule(nn.Module):
    """
    Base class for PyTorch meta-learning modules. These modules accept an
    additional argument `params` in their `forward` method.

    Notes
    -----
    Objects inherited from `MetaModule` are fully compatible with PyTorch
    modules from `torch.nn.Module`. The argument `params` is a dictionary of
    tensors, with full support of the computation graph (for differentiation).
    """

    def meta_named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items()
            if isinstance(module, MetaModule) else [],
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param


class MetaSequential(nn.Sequential, MetaModule):
    __doc__ = nn.Sequential.__doc__

    def forward(self, input, params=None):
        for name, module in self._modules.items():
            if isinstance(module, MetaModule):
                input = module(input, params=get_subdict(params, name))
            elif isinstance(module, nn.Module):
                input = module(input)
            else:
                raise TypeError('The module must be either a torch module '
                                '(inheriting from `nn.Module`), or a `MetaModule`. '
                                'Got type: `{0}`'.format(type(module)))
        return input


class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params.get('weight', None)
        if weight is None:
            return nn.Linear.forward(self, input)

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input, **kwargs):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class TESTmodel(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 net_cfg,
                 kernel_cfg,
                 conv_cfg,
                 mask_cfg, **kwargs
                 ):
        super(TESTmodel, self).__init__()
        input_dim = 784
        hidden_dim = net_cfg.no_hidden
        output_dim = out_channels
        layers = net_cfg.no_blocks

        # nl, nl_weight_init, first_layer_init = Sine(), sine_init, first_layer_sine_init
        nl, nl_weight_init, first_layer_init = nn.ReLU(), init_weights_normal, None

        input_layer = [
            BatchLinear(input_dim, hidden_dim), nl
        ]
        hidden_layers = []
        for _ in range(layers):
            hidden_layers.extend([
                BatchLinear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nl,
            ])
        output_layer = BatchLinear(hidden_dim, out_channels)

        self.net = MetaSequential(*(input_layer + hidden_layers))
        self.out_layer = output_layer

        self.net.apply(nl_weight_init)
        self.out_layer.apply(nl_weight_init)
        if first_layer_init:
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        # B In S
        if params is None:
            params = OrderedDict(self.named_parameters())

        def layer_output_with_grad(layer, input, create_graph):
            output = layer(input, params=get_subdict(params, f'net.{i}'))

            return output, None
            # input_grad, = torch.autograd.grad(output, input, grad_outputs=torch.ones_like(output), create_graph=create_graph, retain_graph=True)
            # input.retain_grad()
            # output.retain_grad()
            # return output, input_grad

        input = coords.squeeze(1)
        grads_output = {}
        create_graph = True
        for i, layer in enumerate(self.net):
            if isinstance(layer, BatchLinear) or isinstance(layer, Sine):
                input, input_grad = layer_output_with_grad(layer, input, create_graph)
                grads_output[rf'{i}_grad'] = input_grad
                grads_output[rf'{i}_output'] = input
            else:
                input = layer(input)
        input = self.out_layer(input)
        return input


class EVALmodel_sequence(TESTmodel):
    def __init__(self,
                 in_channels,
                 out_channels,
                 net_cfg,
                 kernel_cfg,
                 conv_cfg,
                 mask_cfg, **kwargs):
        super(EVALmodel_sequence, self).__init__(in_channels,
                                                 out_channels,
                                                 net_cfg,
                                                 kernel_cfg,
                                                 conv_cfg,
                                                 mask_cfg, **kwargs)
        for name, value in self.net._modules.items():
            if isinstance(value, nn.Linear):
                # self.net._modules[name] = w_norm(self.net._modules[name], name='weight')
                # self.net._modules[name] = w_norm(self.net._modules[name], name='bias')
                pass


#
if __name__ == '__main__':
    dic_a = {
        'a.1': 1,
        'a.2': 2,
        'b': 3
    }
    b = dic_a.get('123')
    print(b)
