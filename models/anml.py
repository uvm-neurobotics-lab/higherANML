"""
A Neuromodulated Meta-Learner (ANML)
"""

import torch
import torch.nn as nn

from models.registry import register
from utils import calculate_output_size_for_fc_layer, has_arg


def _conv_block(in_channels, out_channels, pooling=True):
    # NOTE: If the parameters of this conv+pooling change, we also need to change `recommended_number_of_convblocks()`.
    oplist = [
        nn.Conv2d(in_channels, out_channels, (3, 3), stride=(1, 1), padding=0),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(),
    ]
    if pooling:
        oplist.append(nn.MaxPool2d(2))
    return nn.Sequential(*oplist)


def recommended_number_of_convblocks(input_shape):
    """
    Uses a heuristic to estimate how many ConvBlocks we want in our net, based on the size of the images.
    Args:
        input_shape (tuple): The shape of the input images to be used.
    Returns:
        int: The recommended number of ConvBlocks.
    """
    def downsample(sz):
        # Simulate what one 3x3 convolution + 2-stride max pooling will do.
        return (sz - 2) // 2

    num_blocks = 0
    sz = input_shape[-1]
    while sz > 4:  # Let's shoot for a final feature map of 4x4 or smaller.
        sz = downsample(sz)
        num_blocks += 1
    return num_blocks


def _construct_convnet(in_channels, block_descriptions):
    assert len(block_descriptions) > 0
    blocks = []
    for channels, should_pool in block_descriptions:
        blocks.append(_conv_block(in_channels, channels, should_pool))
        in_channels = channels
    return nn.Sequential(*blocks)


def _construct_uniform_convnet(in_channels, hidden_channels, num_blocks, pool_at_end):
    assert num_blocks > 0
    block_descriptions = [[hidden_channels, True] for _ in range(num_blocks)]
    block_descriptions[-1][-1] = pool_at_end
    return _construct_convnet(in_channels, block_descriptions)


def init_model_weights(model):
    # Don't use PyTorch default initialization.
    # https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html#Solution
    for m in model.modules():
        # TODO: Should we include linear layers here? Seems to be most important for the conv, though.
        if isinstance(m, nn.Conv2d):
            # TODO: If bias is present, we may want to re-init that; not sure.
            nn.init.kaiming_normal_(m.weight)


class RLN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_conv_blocks=3, pool_at_end=False):
        super(RLN, self).__init__()
        self.encoder = _construct_uniform_convnet(in_channels, hidden_channels, num_conv_blocks, pool_at_end)

    def forward(self, x):
        x = self.encoder(x)
        return torch.flatten(x, start_dim=1)


class NM(nn.Module):
    def __init__(self, input_shape, hidden_channels, mask_size, num_conv_blocks=3, pool_at_end=True):
        super(NM, self).__init__()
        self.encoder = _construct_uniform_convnet(input_shape[0], hidden_channels, num_conv_blocks, pool_at_end)
        # To create the correct size of linear layer, we need to first know the size of the conv output.
        feature_size = calculate_output_size_for_fc_layer(self.forward_conv, input_shape)
        self.fc = nn.Linear(feature_size, mask_size)
        self.sigmoid = nn.Sigmoid()

    def forward_conv(self, x):
        x = self.encoder(x)
        return torch.flatten(x, start_dim=1)

    def forward_linear(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

    def forward(self, x):
        x = self.forward_conv(x)
        x = self.forward_linear(x)
        return x


class ANML(nn.Module):
    """
    A Neuromodulated Meta-Learner (ANML)
    """
    def __init__(self, input_shape, rln_chs, nm_chs, num_classes=1000, num_conv_blocks=3, pool_rln_output=False):
        super(ANML, self).__init__()
        self.input_shape = input_shape
        self.rln = RLN(input_shape[0], rln_chs, num_conv_blocks, pool_rln_output)
        feature_size = calculate_output_size_for_fc_layer(self.rln, input_shape)
        self.nm = NM(input_shape, nm_chs, feature_size, num_conv_blocks)
        self.fc = nn.Linear(feature_size, num_classes)
        init_model_weights(self)

    def forward(self, x):
        features = self.rln(x)
        nm_mask = self.nm(x)

        features = features * nm_mask

        out = self.fc(features)

        return out


class ANMLEncoder(nn.Module):
    """
    ANML, but without the linear classifier on top.
    """
    def __init__(self, input_shape, rln_chs, nm_chs, num_conv_blocks=3, pool_rln_output=False):
        super(ANMLEncoder, self).__init__()
        self.input_shape = input_shape
        self.rln = RLN(input_shape[0], rln_chs, num_conv_blocks, pool_rln_output)
        feature_size = calculate_output_size_for_fc_layer(self.rln, input_shape)
        self.nm = NM(input_shape, nm_chs, feature_size, num_conv_blocks)
        init_model_weights(self)

    def forward(self, x):
        features = self.rln(x)
        nm_mask = self.nm(x)

        out = features * nm_mask

        return out


class SANML(nn.Module):
    """
    ANML, without the neuromodulation (the entire NM module is removed).
    """
    def __init__(self, input_shape, rln_chs, num_classes=1000, num_conv_blocks=3, pool_rln_output=False):
        super(SANML, self).__init__()
        self.input_shape = input_shape
        self.rln = RLN(input_shape[0], rln_chs, num_conv_blocks, pool_rln_output)
        feature_size = calculate_output_size_for_fc_layer(self.rln, input_shape)
        self.fc = nn.Linear(feature_size, num_classes)
        init_model_weights(self)

    def forward(self, x):
        features = self.rln(x)
        return self.fc(features)


class NONML(nn.Module):
    """
    Non-modulated version of ANML.
    Instead of using an elementwise softmax masking the features from the NM path are concatenated.
    """
    def __init__(self, input_shape, rln_chs, nm_chs, num_classes=1000, num_conv_blocks=3, pool_rln_output=False):
        super(NONML, self).__init__()
        self.input_shape = input_shape
        self.rln = RLN(input_shape[0], rln_chs, num_conv_blocks, pool_rln_output)
        feature_size = calculate_output_size_for_fc_layer(self.rln, input_shape)
        self.nm = NM(input_shape, nm_chs, feature_size, num_conv_blocks)
        self.fc = nn.Linear(feature_size, num_classes)
        init_model_weights(self)

    def forward(self, x):
        features = self.rln(x)
        conditioning = self.nm(x)

        features = torch.cat([features, conditioning], dim=1)
        out = self.pln(features)

        return out


def create_anml_variant(ModelClass, input_shape, **kwargs):
    model_args = dict(kwargs)
    model_args["input_shape"] = input_shape

    # Auto-derive some arguments if they are not explicitly defined by the user.
    if "num_classes" not in model_args and has_arg(ModelClass, "num_classes"):
        # TODO: Auto-size this instead.
        # model_args["num_classes"] = max(sampler.num_train_classes(), sampler.num_test_classes())
        model_args["num_classes"] = 1000
    if "num_conv_blocks" not in model_args:
        model_args["num_conv_blocks"] = recommended_number_of_convblocks(input_shape)
    if "pool_rln_output" not in model_args:
        # For backward compatibility, we turn off final pooling if the images are <=30 px, as done in the original ANML.
        model_args["pool_rln_output"] = input_shape[-1] > 30

    # Finally, create the model.
    anml = ModelClass(**model_args)
    return anml, model_args


@register("anml")
def create_anml(input_shape, **kwargs):
    return create_anml_variant(ANML, input_shape, **kwargs)


@register("anml-encoder")
def create_anml(input_shape, **kwargs):
    return create_anml_variant(ANMLEncoder, input_shape, **kwargs)


@register("sanml")
def create_sanml(input_shape, **kwargs):
    return create_anml_variant(SANML, input_shape, **kwargs)


@register("nonml")
def create_nonml(input_shape, **kwargs):
    return create_anml_variant(NONML, input_shape, **kwargs)
