"""
A Neuromodulated Meta-Learner (ANML)
"""

import torch
import torch.nn as nn

from .models import register


def _linear_layer(in_dims, out_dims):
    # Sanity check: we can increase this limit if desired.
    MAX_LAYER_SIZE = int(1e8)
    if in_dims * out_dims > MAX_LAYER_SIZE:
        raise RuntimeError(f"You tried to create a layer with more than {MAX_LAYER_SIZE} parameters. Is this a"
                           " mistake? You should add more pooling or longer strides.")
    return nn.Linear(in_dims, out_dims)


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
        batch_shape = (2,) + tuple(input_shape)
        shape_after_conv = self.forward_conv(torch.zeros(batch_shape)).shape
        assert len(shape_after_conv) == 2, "Conv output should only be two dims."
        self.fc = _linear_layer(shape_after_conv[-1], mask_size)
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
    def __init__(self, input_shape, rln_chs, nm_chs, num_classes=1000, num_conv_blocks=3, pool_rln_output=False):
        super(ANML, self).__init__()
        self.input_shape = input_shape
        self.rln = RLN(input_shape[0], rln_chs, num_conv_blocks, pool_rln_output)
        # Automatically determine what the size of the final layer needs to be.
        # Simulate a batch by adding an extra dim at the beginning.
        batch_shape = (2,) + tuple(input_shape)
        shape_after_rln = self.rln(torch.zeros(batch_shape)).shape
        assert len(shape_after_rln) == 2, "RLN output should only be two dims."
        feature_size = shape_after_rln[-1]
        self.nm = NM(input_shape, nm_chs, feature_size, num_conv_blocks)
        self.fc = _linear_layer(feature_size, num_classes)

        # Don't use PyTorch default initialization.
        # https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html#Solution
        for m in self.modules():
            # TODO: Should we include linear layers here? Seems to be most important for the conv, though.
            if isinstance(m, nn.Conv2d):
                # TODO: If bias is present, we may want to re-init that; not sure.
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        features = self.rln(x)
        nm_mask = self.nm(x)

        features = features * nm_mask

        out = self.fc(features)

        return out


@register("anml")
def create_anml(input_shape, **kwargs):
    model_args = dict(kwargs)
    model_args["input_shape"] = input_shape

    # Auto-derive some arguments if they are not explicitly defined by the user.
    if "num_classes" not in model_args:
        # TODO: Auto-size this instead.
        # model_args["num_classes"] = max(sampler.num_train_classes(), sampler.num_test_classes())
        model_args["num_classes"] = 1000
    if "num_conv_blocks" not in model_args:
        model_args["num_conv_blocks"] = recommended_number_of_convblocks(input_shape)
    if "pool_rln_output" not in model_args:
        # For backward compatibility, we turn off final pooling if the images are <=30 px, as done in the original ANML.
        model_args["pool_rln_output"] = input_shape[-1] > 30

    # Finally, create the model.
    anml = ANML(**model_args)
    return anml, model_args
