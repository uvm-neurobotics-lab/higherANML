"""
This file is borrowed with many thanks from [Few-Shot Meta-Baseline](https://github.com/yinboc/few-shot-meta-baseline),
by Yinbo Chen. It was copied on 2021-12-17. The license for this file can be found at ./few-shot-meta-baseline-LICENSE.
"""
import torch.nn as nn

from models.registry import register


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


@register("convnet4")
class ConvNet4(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        # TODO: Add proper initialization.
        # TODO: Make out_dim auto-detecting (based on input shape)?
        self.out_dim = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.shape[0], -1)

