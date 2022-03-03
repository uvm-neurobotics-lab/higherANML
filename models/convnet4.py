"""
This file is borrowed with many thanks from [Few-Shot Meta-Baseline](https://github.com/yinboc/few-shot-meta-baseline),
by Yinbo Chen. It was copied on 2021-12-17. The license for this file can be found at ./few-shot-meta-baseline-LICENSE.
"""
from collections import OrderedDict
from functools import partial

import torch.nn as nn

from models.registry import register


NORM_MAPPING = {}
NORM_MAPPING["bn"] = nn.BatchNorm2d
NORM_MAPPING["batch"] = NORM_MAPPING["bn"]
NORM_MAPPING["batchnorm"] = NORM_MAPPING["bn"]
NORM_MAPPING["gn"] = partial(nn.GroupNorm, num_groups=1)
NORM_MAPPING["group"] = NORM_MAPPING["gn"]
NORM_MAPPING["groupnorm"] = NORM_MAPPING["gn"]
NORM_MAPPING["ln"] = partial(nn.LayerNorm, normalized_shape=[3, 84, 84])
NORM_MAPPING["layer"] = NORM_MAPPING["ln"]
NORM_MAPPING["layernorm"] = NORM_MAPPING["ln"]
NORM_MAPPING["in"] = partial(nn.InstanceNorm2d, affine=True)
NORM_MAPPING["instance"] = NORM_MAPPING["in"]
NORM_MAPPING["instancenorm"] = NORM_MAPPING["in"]


def conv_block(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, norm_type="in", pool_size=2):
    if isinstance(norm_type, str):
        norm_type = NORM_MAPPING[norm_type]
    ops = [("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))]
    if norm_type:
        ops.append(("norm", norm_type(out_channels)))
    ops.append(("relu", nn.ReLU()))
    if pool_size > 0:
        ops.append(("pool", nn.MaxPool2d(pool_size)))
    return nn.Sequential(OrderedDict(ops))


def init_model_weights(model):
    # Don't use PyTorch default initialization.
    # https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html#Solution
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


@register("convnet4")
class ConvNet4(nn.Module):

    def __init__(self, input_shape=None, x_dim=3, hid_dims=64, z_dim=64, num_blocks=4, kernel_size=(3, 3),
                 stride=(1, 1), padding=1, norm_type="in", pool_size=2):
        super().__init__()

        if num_blocks < 1:
            raise RuntimeError(f"num_blocks must be at least 1, but got {num_blocks}.")
        # Input shape overrides x_dim if present.
        if input_shape:
            x_dim = input_shape[0]
        # hid_dims can be a single value or num_blocks values.
        if num_blocks == 1:
            hid_dims = []
        elif not isinstance(hid_dims, (list, tuple)):
            hid_dims = [hid_dims] * (num_blocks - 1)

        layers = [x_dim] + (hid_dims if hid_dims else []) + [z_dim]
        make_block = partial(conv_block, kernel_size=kernel_size, stride=stride, padding=padding, norm_type=norm_type,
                             pool_size=pool_size)
        ops = [make_block(inn, out) for inn, out in zip(layers, layers[1:])]
        if len(ops) > 1:
            self.encoder = nn.Sequential(*ops)
        else:
            self.encoder = ops[0]

        init_model_weights(self)

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.shape[0], -1)

