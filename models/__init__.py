"""
Models
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

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


def collect_sparse_init_sample(encoder, dataset, device):
    """ Collect a single example from each class (the first one returned when iterating through the entire dataset). """
    images = []
    labels = []
    seen = set()
    for image, label in dataset:
        if label.item() not in seen:
            seen.add(label.item())
            images.append(image)
            labels.append(label)
    xs = encoder(torch.stack(images).to(device))
    ys = torch.stack(labels)
    return xs, ys


def collect_dense_init_sample(encoder, config, dataset, device):
    """ Collect as many samples as requested by the init_size parameter. """
    ensure_config_param(config, "init_size", lambda val: val > 0)
    ensure_config_param(config, "batch_size", lambda val: val > 0)

    xs = []
    ys = []
    num_remaining = config["init_size"]
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    while num_remaining > 0:
        for images, labels in loader:
            images = images.to(device)
            feats = encoder(images).detach().cpu()
            num_remaining -= len(feats)
            if num_remaining < 0:
                # Drop the unneeded items.
                feats = feats[:num_remaining]
                labels = labels[:num_remaining]
            xs.append(feats)
            ys.append(labels)
            if num_remaining <= 0:
                break

    xs = torch.cat(xs)
    ys = torch.cat(ys)
    return xs, ys


def maybe_collect_init_sample(model, config, dataset, device):
    """
    Collect samples from the given support set that may be used for initialization of parameters. If the "init_size"
    variable isn't present in the config, then simply take one sample per class according to
    `collect_sparse_init_sample`.

    The samples will be pre-processed into a single batch of feature encodings, taken from the second-to-last layer of
    the model. This is a suitable format to use for, e.g., `lstsq_reinit()`.

    This will return None if no samples are needed according to the config.

    Args:
        model (torch.nn.Module): The model to be partially reinitialized (used to encode the samples).
        config (dict): The config with initialization parameters.
        dataset (torch.utils.data.Dataset): The support set containing samples which are allowed to be used for
            initialization purposes.
        device (torch.device or str): The device on which to run inference.

    Returns:
        xs (torch.Tensor or None): The feature-encodings of all samples in a single batch.
        ys (torch.Tensor or None): The labels corresponding to the samples.
    """
    # Shortcut: We can skip this whole procedure if the samples won't be used.
    if not ("reinit_params" in config) or config.get("reinit_method") != "lstsq":
        return None, None

    # TODO: Fix this so we can extract the next-to-last layer, whatever that is.
    if config.get("init_size"):
        return collect_dense_init_sample(model.encoder, config, dataset, device)
    else:
        return collect_sparse_init_sample(model.encoder, dataset, device)


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
    feats, labels = init_support_set
    assert len(feats) == len(labels)

    # Get the linear layer to reinit.
    fc = get_matching_module(model, params)
    # Check that our samples match the linear layer size.
    if len(feats.shape) != 2 or feats.shape[1] != fc.in_features:
        raise RuntimeError(f"Expected the init sample set to be a batch of {fc.in_features}-length vectors.")

    # Now reinit.
    ys = onehottify(labels, fc.out_features)
    device = fc.weight.device
    dtype = fc.weight.dtype

    xs = feats.detach().cpu().numpy()
    bias = np.ones([xs.shape[0], 1])
    inp = np.concatenate([xs, bias], axis=1)  # add bias
    W, residuals, rank, s = np.linalg.lstsq(inp, ys, rcond=-1)
    W, b = W[:-1], W[-1]  # retrieve bias

    # insert W,b into Linear layer
    fc.weight.data = torch.from_numpy(W.T).to(device).type(dtype)
    fc.bias.data = torch.from_numpy(b).to(device).type(dtype)


def reinit_params(model, config, init_support_set=None):
    """
    Reinitialize certain parameters as dictated by the config.

    Args:
        model (torch.nn.Module): The model to be partially reinitialized.
        config (dict): The config with fine-tuning and optimization parameters.
        init_support_set (tuple): A pair of (feature, label) tensors which may be used for any special initialization.
    """
    if not ("reinit_params" in config):
        # Nothing to reinit.
        return
    ensure_config_param(config, "reinit_params", lambda obj: isinstance(obj, (str, list, tuple)))
    if "reinit_method" in config:
        ensure_config_param(config, "reinit_method", lambda val: val in ("kaiming", "lstsq"))

    if config.get("reinit_method") == "lstsq":
        lstsq_reinit(config["reinit_params"], model, init_support_set)
    else:
        # Default to kaiming normal.
        kaiming_reinit(config["reinit_params"], model)


def create_optimizer(model, config):
    """
    Create an optimizer and set up certain parameters for optimization, as specified by the given config.

    Args:
        model (torch.nn.Module): The model to be fine-tuned.
        config (dict): The config with fine-tuning and optimization parameters.

    Returns:
        torch.optim.Optimizer: The optimizer that can be used in a training loop.
    """
    ensure_config_param(config, "opt_params", lambda obj: isinstance(obj, (str, list, tuple)))
    ensure_config_param(config, "lr", lambda val: val > 0)

    # Select which layers will recieve updates during optimization, by setting the requires_grad property.
    for p in model.parameters():  # disable all learning by default.
        p.requires_grad_(False)
    for p in collect_matching_params(model, config["opt_params"]):  # re-enable just for these params.
        p.requires_grad_(True)

    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    return opt


def fine_tuning_setup(model, config, init_support_set=None):
    """
    Set up the given model for fine-tuning; creating an optimizer to optimize parameters selected by the given config,
    and reinitialize certain parameters as dictated by the config.

    Args:
        model (torch.nn.Module): The model to be fine-tuned.
        config (dict): The config with fine-tuning and optimization parameters.
        init_support_set (tuple): A pair of (feature, label) tensors which may be used for any special initialization.

    Returns:
        torch.optim.Optimizer: The optimizer that can be used in a training loop.
    """
    # Set up which parameters we will be fine-tuning and/or learning from scratch.
    # First, reinitialize layers that we want to learn from scratch.
    reinit_params(model, config, init_support_set)

    # Now, set up an optimizer for just the layers we want to be updated.
    return create_optimizer(model, config)
