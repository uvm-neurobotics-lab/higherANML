"""
This file is borrowed with many thanks from [Few-Shot Meta-Baseline](https://github.com/yinboc/few-shot-meta-baseline),
by Yinbo Chen. It was copied on 2021-12-17. The license for this file can be found at ./few-shot-meta-baseline-LICENSE.
"""
import logging
from collections import OrderedDict
from functools import partial
from itertools import count

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


def conv_block(in_channels, out_channels, num_conv=1, kernel_size=(3, 3), stride=(1, 1), padding=1, norm_type="in",
               pool_size=2):
    if isinstance(norm_type, str):
        norm_type = NORM_MAPPING[norm_type]

    # Like a ResNet, each block consists of some number of conv + norm + relu, followed by a final pooling.
    ops = []
    for i in range(num_conv):
        ops.append((f"conv{i}", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)))
        if norm_type:
            ops.append((f"norm{i}", norm_type(out_channels)))
        ops.append((f"relu{i}", nn.ReLU()))
        in_channels = out_channels

    if pool_size:
        ops.append(("pool", nn.MaxPool2d(pool_size)))

    return nn.Sequential(OrderedDict(ops))


def init_model_weights(model):
    # Don't use PyTorch default initialization.
    # https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html#Solution
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


@register("convnet")
class ConvNet(nn.Module):

    def __init__(self, input_shape=None, x_dim=3, num_filters=64, num_blocks=4, num_conv_per_block=1,
                 kernel_size=(3, 3), stride=(1, 1), padding=1, norm_type="in", pool_size=2):
        super().__init__()

        if num_blocks < 1:
            raise RuntimeError(f"num_blocks must be at least 1, but got {num_blocks}.")
        # Input shape overrides x_dim if present.
        if input_shape:
            x_dim = input_shape[0]

        def ensure_list(param):
            # If a single value, make it the same for all blocks.
            if not isinstance(param, (list, tuple)):
                return [param] * num_blocks
            # If two values, and the values are not sequences, assume this is a single 2D value for all blocks.
            elif len(param) == 2 and not isinstance(param[0], (list, tuple)):
                if num_blocks == 2:
                    logging.warning(f"Assuming your list of two parameters {param} is a single 2D parameter, rather"
                                    " than two separate settings for each of the two conv blocks. If you instead"
                                    " intended (1st block val, 2nd block val), you should provide explicit 2D settings"
                                    " like ((1st, 1st), (2nd, 2nd)).")
                return [param] * num_blocks
            # If multiple values, there must be one value for each block.
            elif len(param) == num_blocks:
                return param
            # Value list does not match number of blocks.
            else:
                raise RuntimeError(f"Must be either a single value or the same as the number of blocks ({num_blocks})."
                                   f" Instead got {len(param)} values.")

        # Each of the following settings can be either a single value or num_blocks values.
        num_conv_per_block = ensure_list(num_conv_per_block)
        num_filters = ensure_list(num_filters)
        kernel_size = ensure_list(kernel_size)
        stride = ensure_list(stride)
        padding = ensure_list(padding)
        norm_type = ensure_list(norm_type)
        pool_size = ensure_list(pool_size)
        # One more parameter for the "in_channels".
        in_channels = [x_dim] + num_filters[:-1]

        # Now use each setting to make a different block. Careful that the ordering here is correct!
        ops = [(f"block{idx}", conv_block(in_channels=i, out_channels=o, num_conv=nc, kernel_size=ke, stride=st,
                                          padding=pa, norm_type=no, pool_size=po))
               for idx, i, o, nc, ke, st, pa, no, po in zip(count(1), in_channels, num_filters, num_conv_per_block,
                                                            kernel_size, stride, padding, norm_type, pool_size)]
        if len(ops) > 1:
            self.encoder = nn.Sequential(OrderedDict(ops))
        else:
            self.encoder = ops[0][1]

        init_model_weights(self)

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.shape[0], -1)


@register("convnet4")
def convnet4(**kwargs):
    kwargs["num_blocks"] = 4
    return ConvNet(**kwargs), kwargs
