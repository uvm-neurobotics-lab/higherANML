"""
Models
"""

import logging
from pathlib import Path

import numpy as np
import torch

from .legacy import ANML as LegacyANML
from .registry import make, make_from_config, load
from . import anml
from . import classifier
from . import convnet
from . import meta_baseline
from . import oml
from . import resnet
from . import resnet12

import utils.storage as storage
from utils import collect_matching_named_params, collect_matching_params, ensure_config_param, get_matching_module


def load_model(model_path, sampler_input_shape, device=None):
    """
    Deserialize model from the given file onto the given device, with backward compatibility to older models and other
    error checking.

    Args:
        model_path (str or Path): The file from which to load.
        sampler_input_shape (tuple): The shape of data we expect to be feeding into the model (shape of a single input,
            not a batch).
        device (str or torch.device): The device to load to.

    Returns:
        any: The loaded model.
    """
    model_path = Path(model_path).resolve()
    if model_path.suffix == ".net":
        # Assume this was saved by the storage module, which pickles the entire model.
        model = storage.load(model_path, device=device)
    elif model_path.suffix == ".pt" or model_path.suffix == ".pth":
        # Assume the model was saved in the legacy format:
        #   - Only state_dict is stored.
        #   - Model shape is identified by the filename.
        sizes = [int(num) for num in model_path.name.split("_")[:-1]]
        if len(sizes) != 3:
            raise RuntimeError(f"Unsupported model shape: {sizes}")
        rln_chs, nm_chs, mask_size = sizes
        if mask_size != (rln_chs * 9):
            raise RuntimeError(f"Unsupported model shape: {sizes}")

        # Backward compatibility: Before we constructed the network based on `input_shape` and `num_classes`. At this
        # time, `num_classes` was always 1000 and we always used greyscale 28x28 images.
        input_shape = (1, 28, 28)
        out_classes = 1000
        model = LegacyANML(input_shape, rln_chs, nm_chs, out_classes)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        supported = (".net", ".pt", ".pth")
        raise RuntimeError(f"Unsupported model file type: {model_path}. Expected one of {supported}.")

    logging.debug(f"Model shape:\n{model}")

    # If possible, check if the images we are testing on match the dimensions of the images this model was built for.
    if hasattr(model, "input_shape") and tuple(model.input_shape) != tuple(sampler_input_shape):
        raise RuntimeError("The specified dataset image sizes do not match the size this model was trained for.\n"
                           f"Data size:  {sampler_input_shape}\n"
                           f"Model size: {model.input_shape}")
    return model


def kaiming_reinit(params, model):
    for n, p in collect_matching_named_params(model, params):
        # HACK: Here we will use the parameter naming to tell us how the params should be initialized. This may not be
        # appropriate for all types of layers! We are typically only expecting fully-connected Linear layers here.
        if n.endswith("weight"):
            torch.nn.init.kaiming_normal_(p)
        elif n.endswith("bias"):
            torch.nn.init.constant_(p, 0)
        else:
            raise RuntimeError(f"Cannot reinitialize this unknown parameter type: {n}")


def onehottify(labels, num_total_classes):
    onehots = torch.zeros((len(labels), num_total_classes))
    onehots[torch.arange(len(labels)), labels] = 1
    return onehots.cpu().detach().numpy()


def lstsq_reinit(params, model, init_support_set):
    # Sanity checking inputs.
    if isinstance(params, (list, tuple)):
        if len(params) == 0:
            params = None
        elif len(params) == 1:
            params = params[0]
        else:
            raise ValueError("To use least squares initialization, you must only supply a single linear layer to be"
                             f" reinitialized. Instead you gave: {params}.")
    if not isinstance(params, str):
        raise ValueError(f"Expected the name of a single module. Instead received: {params}")
    if not params:
        # Nothing to reinit.
        return

    if not init_support_set:
        raise ValueError("You must supply an init support set for least squares initialization.")
    images, labels = init_support_set

    # Get the linear layer to reinit.
    fc = get_matching_module(model, params)

    # Now reinit.
    ys = onehottify(labels, fc.weight.shape[0])  # num_total_classes = fc.out_features
    # TODO: Clean this up so we don't need to magically know the model contains an "encoder".
    feats = model.encoder(images)
    device = feats.device
    dtype = feats.dtype

    xs = feats.detach().cpu().numpy()
    bias = np.ones([xs.shape[0], 1])
    inp = np.concatenate([xs, bias], axis=1)  # add bias
    W, residuals, rank, s = np.linalg.lstsq(inp, ys, rcond=-1)
    W, b = W[:-1], W[-1]  # retrieve bias

    # insert W,b into Linear layer
    fc.weight.data = torch.from_numpy(W.T).type(dtype)
    fc.bias.data = torch.from_numpy(b).type(dtype)
    # TODO: Do we actually need this? Don't think so...
    # fc = fc.to(device).type(dtype)


def fine_tuning_setup(model, config, init_support_set=None):
    """
    Set up the given model for fine-tuning; creating an optimizer to optimize parameters selected by the given config,
    and reinitialize certain parameters as dictated by the config.

    Args:
        model (torch.nn.Module): The model to be fine-tuned.
        config (dict): The config with fine-tuning and optimization parameters.
        init_support_set (tuple): A pair of (image, label) tensors which may be used for any special initialization.

    Returns:
        torch.optim.Optimizer: The optimizer that can be used in a training loop.
    """
    ensure_config_param(config, "reinit_params", lambda obj: isinstance(obj, (str, list, tuple)))
    ensure_config_param(config, "opt_params", lambda obj: isinstance(obj, (str, list, tuple)))
    ensure_config_param(config, "lr", lambda val: val > 0)
    if "reinit_method" in config:
        ensure_config_param(config, "reinit_method", lambda val: val in ("kaiming", "lstsq"))

    # Set up which parameters we will be fine-tuning and/or learning from scratch.
    # First, reinitialize layers that we want to learn from scratch.
    if config.get("reinit_method") == "lstsq":
        lstsq_reinit(config["reinit_params"], model, init_support_set)
    else:
        # Default to kaiming normal.
        kaiming_reinit(config["reinit_params"], model)

    # Now, select which layers will recieve updates during optimization, by setting the requires_grad property.
    for p in model.parameters():  # disable all learning by default.
        p.requires_grad_(False)
    for p in collect_matching_params(model, config["opt_params"]):  # re-enable just for these params.
        p.requires_grad_(True)

    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    return opt
