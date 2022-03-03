"""
This file is borrowed with many thanks from [Few-Shot Meta-Baseline](https://github.com/yinboc/few-shot-meta-baseline),
by Yinbo Chen. It was copied on 2021-12-17. The license for this file can be found at ./few-shot-meta-baseline-LICENSE.
"""
import math

import torch
import torch.nn as nn

import utils
from models.registry import make, register


@register("classifier")
class Classifier(nn.Module):
    
    def __init__(self, input_shape, encoder, classifier, encoder_args=None, classifier_args=None):
        super().__init__()
        if encoder_args is None:
            encoder_args = {}
        self.encoder, _ = make(encoder, input_shape, **encoder_args)
        if classifier_args is None:
            classifier_args = {}
        output_shape = utils.calculate_output_shape(self.encoder, input_shape)
        self.classifier, _ = make(classifier, output_shape, **classifier_args)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


@register("linear-classifier")
class LinearClassifier(nn.Module):

    def __init__(self, input_shape, num_classes):
        super().__init__()
        if len(input_shape) != 1:
            raise RuntimeError(f"Cannot stack {type(self).__name__} on top of output of shape: {input_shape}.")
        self.linear = nn.Linear(input_shape[0], num_classes)

    def forward(self, x):
        return self.linear(x)


@register("mlp-classifier")
class MLPClassifier(nn.Module):

    def __init__(self, input_shape, num_classes, hidden_layers=None):
        super().__init__()
        if len(input_shape) != 1:
            raise RuntimeError(f"Cannot stack {type(self).__name__} on top of output of shape: {input_shape}.")

        if hidden_layers is None:
            hidden_layers = []
        elif not isinstance(hidden_layers, (list, tuple)):
            hidden_layers = [hidden_layers]
        layers = [input_shape[0]] + hidden_layers + [num_classes]
        ops = []
        for inn, out in zip(layers, layers[1:]):
            ops.append(nn.Linear(inn, out))
            ops.append(nn.ReLU())
        # Remove the last ReLU. It will be passed through a softmax instead (in the CE loss, not here).
        del ops[-1]
        if len(ops) > 1:
            self.mlp = nn.Sequential(*ops)
        else:
            self.mlp = ops[0]

    def forward(self, x):
        return self.mlp(x)


@register("metric-classifier")
class MetricClassifier(nn.Module):

    def __init__(self, input_shape, num_classes, metric="cos", temp=None):
        super().__init__()
        if len(input_shape) != 1:
            raise RuntimeError(f"Cannot stack {type(self).__name__} on top of output of shape: {input_shape}.")

        self.proto = nn.Parameter(torch.empty(num_classes, input_shape[0]))
        nn.init.kaiming_uniform_(self.proto, a=math.sqrt(5))
        if temp is None:
            if metric == "cos":
                temp = nn.Parameter(torch.tensor(10.))
            else:
                temp = 1.0
        self.metric = metric
        self.temp = temp
        self.outlayer = None

    def forward(self, x):
        return utils.compute_logits(x, self.proto, self.metric, self.temp)

