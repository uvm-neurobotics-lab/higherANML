"""
Models
"""

import logging
from pathlib import Path

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
from utils import collect_matching_named_params, collect_matching_params


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


def fine_tuning_setup(model, config):
    """
    Set up the given model for fine-tuning; creating an optimizer to optimize parameters selected by the given config.

    Args:
        model (torch.nn.Module): The model to be fine-tuned.
        config (dict): The config with fine-tuning and optimization parameters.

    Returns:
        torch.optim.Optimizer: The optimizer that can be used in a training loop.
    """
    # Set up which parameters we will be fine-tuning and/or learning from scratch.
    # First, reinitialize layers that we want to learn from scratch.
    for n, p in collect_matching_named_params(model, config["reinit_params"]):
        # HACK: Here we will use the parameter naming to tell us how the params should be initialized. This may not be
        # appropriate for all types of layers! We are typically only expecting fully-connected Linear layers here.
        if n.endswith("weight"):
            torch.nn.init.kaiming_normal_(p)
        elif n.endswith("bias"):
            torch.nn.init.constant_(p, 0)
        else:
            raise RuntimeError(f"Cannot reinitialize this unknown parameter type: {n}")

    # Now, select which layers will recieve updates during optimization, by setting the requires_grad property.
    for p in model.parameters():  # disable all learning by default.
        p.requires_grad_(False)
    for p in collect_matching_params(model, config["opt_params"]):  # re-enable just for these params.
        p.requires_grad_(True)

    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    return opt
