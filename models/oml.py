"""
This is the OML model (Online-Aware Meta-Learning) from "Meta-Learning Representations for Continual Learning".
https://arxiv.org/abs/1905.12588

Exact architecture is taken from here:
https://github.com/khurramjaved96/mrcl/blob/1714cb56aa5b6001e3fd43f90d4c41df1b5831ff/model/modelfactory.py#L139-L167
"""
import torch.nn as nn

from models.registry import register
from utils import calculate_output_size_for_fc_layer


def _conv_block(in_channels, out_channels, stride=(1, 1)):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (3, 3), stride=stride, padding=0),
        # nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        # nn.MaxPool2d(2)
    )


@register("oml-encoder")
class RLN(nn.Module):

    def __init__(self, input_shape, planes=256):
        super().__init__()
        self.blocks = nn.Sequential(
            _conv_block(input_shape[0], planes, (2, 2)),
            _conv_block(planes, planes, (1, 1)),
            _conv_block(planes, planes, (2, 2)),
            _conv_block(planes, planes, (1, 1)),
            _conv_block(planes, planes, (2, 2)),
            _conv_block(planes, planes, (2, 2)),
            nn.Flatten(),
        )
        # Don't use PyTorch default initialization.
        # https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html#Solution
        for m in self.modules():
            # Also initializing linear layers using Kaiming Normal, following the original OML implementation.
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.blocks(x)


@register("oml")
class OML(nn.Module):

    def __init__(self, input_shape, planes=256, fc_layer_size=1024, num_classes=1000):
        super().__init__()
        self.input_shape = input_shape

        self.rln = RLN(input_shape, planes)

        # To create the correct size of linear layer, we need to first know the size of the conv output.
        feature_size = calculate_output_size_for_fc_layer(self.rln, input_shape)
        self.pn = nn.Sequential(
            nn.Linear(feature_size, fc_layer_size),
            nn.ReLU(),
            nn.Linear(fc_layer_size, num_classes),
        )

        # Don't use PyTorch default initialization.
        # https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html#Solution
        for m in self.modules():
            # Also initializing linear layers using Kaiming Normal, following the original OML implementation.
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.rln(x)
        return self.pn(x)
