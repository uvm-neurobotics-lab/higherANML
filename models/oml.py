"""
This is the OML model (Online-Aware Meta-Learning) from "Meta-Learning Representations for Continual Learning".
https://arxiv.org/abs/1905.12588

Exact architecture is taken from here:
https://github.com/khurramjaved96/mrcl/blob/1714cb56aa5b6001e3fd43f90d4c41df1b5831ff/model/modelfactory.py#L139-L167
"""
import torch
import torch.nn as nn

from .models import register


# TODO: Clean this up and unify with ANML code.
def _linear_layer(in_dims, out_dims):
    # Sanity check: we can increase this limit if desired.
    MAX_LAYER_SIZE = int(1e8)
    if in_dims * out_dims > MAX_LAYER_SIZE:
        raise RuntimeError(f"You tried to create a layer with more than {MAX_LAYER_SIZE} parameters. Is this a"
                           " mistake? You should add more pooling or longer strides.")
    return nn.Linear(in_dims, out_dims)


def _conv_block(in_channels, out_channels, stride=(1, 1)):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (3, 3), stride=stride, padding=0),
        # nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        # nn.MaxPool2d(2)
    )


class OML(nn.Module):

    def __init__(self, input_shape, num_classes=1000):
        super().__init__()
        self.rln = nn.Sequential(
            _conv_block(input_shape[0], 256, (2, 2)),
            _conv_block(256, 256, (1, 1)),
            _conv_block(256, 256, (2, 2)),
            _conv_block(256, 256, (1, 1)),
            _conv_block(256, 256, (2, 2)),
            _conv_block(256, 256, (2, 2)),
            nn.Flatten(),
        )
        # To create the correct size of linear layer, we need to first know the size of the conv output.
        batch_shape = (2,) + tuple(input_shape)
        shape_after_conv = self.rln(torch.zeros(batch_shape)).shape
        assert len(shape_after_conv) == 2, "Conv output should only be two dims."
        self.pn = nn.Sequential(
            _linear_layer(shape_after_conv[-1], 1024),
            nn.ReLU(),
            _linear_layer(1024, num_classes),
        )
        # Given a special name so we can identify this layer separately.
        self.output_layer = self.pn[-1]

    def forward(self, x):
        x = self.rln(x)
        return self.pn(x)


@register("oml")
def create_oml(input_shape, **kwargs):
    model_args = dict(kwargs)
    model_args["input_shape"] = input_shape

    # Auto-derive some arguments if they are not explicitly defined by the user.
    if "num_classes" not in model_args:
        # TODO: Auto-size this instead.
        # model_args["num_classes"] = max(sampler.num_train_classes(), sampler.num_test_classes())
        model_args["num_classes"] = 1000

    # Finally, create the model.
    anml = OML(**model_args)
    return anml, model_args
