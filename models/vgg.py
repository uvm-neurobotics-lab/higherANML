"""
VGG style models, imported from PyTorch.
"""
import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16, vgg16_bn

from models.registry import register


class VGGEncoder(nn.Module):

    def __init__(self, vgg):
        super().__init__()
        self.features = vgg.features
        self.avgpool = vgg.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


@register('vgg16')
def vgg16(pretrained=False, progress=True, dropout=0.0, **kwargs):
    """
    VGG-16 model from Torchvision.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K.
        progress (bool): If True, displays a progress bar of the download to stderr.
        dropout (float): The probability of DropOut on linear layers during training (0.0 means no DropOut).
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG`` base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_ for more details about this class.
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    return vgg16(weights=weights, progress=progress, dropout=dropout, **kwargs)


@register('vgg16_bn')
def vgg16_bn(pretrained=False, progress=True, dropout=0.0, **kwargs):
    """
    VGG-16-BN model from Torchvision.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K.
        progress (bool): If True, displays a progress bar of the download to stderr.
        dropout (float): The probability of DropOut on linear layers during training (0.0 means no DropOut).
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG`` base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_ for more details about this class.
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    return vgg16_bn(weights=weights, progress=progress, dropout=dropout, **kwargs)


@register('vgg16_encoder')
def vgg16_encoder(pretrained=False, progress=True, dropout=0.0, **kwargs):
    """
    VGG-16 model from Torchvision, but only the encoder part (before the linear classifier).

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K.
        progress (bool): If True, displays a progress bar of the download to stderr.
        dropout (float): The probability of DropOut on linear layers during training (0.0 means no DropOut).
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG`` base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_ for more details about this class.
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    vgg = vgg16(weights=weights, progress=progress, dropout=dropout, **kwargs)
    return VGGEncoder(vgg)


@register('vgg16_bn_encoder')
def vgg16_bn_encoder(pretrained=False, progress=True, dropout=0.0, **kwargs):
    """
    VGG-16-BN model from Torchvision, but only the encoder part (before the linear classifier).

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K.
        progress (bool): If True, displays a progress bar of the download to stderr.
        dropout (float): The probability of DropOut on linear layers during training (0.0 means no DropOut).
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG`` base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_ for more details about this class.
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    vgg = vgg16_bn(weights=weights, progress=progress, dropout=dropout, **kwargs)
    return VGGEncoder(vgg)
