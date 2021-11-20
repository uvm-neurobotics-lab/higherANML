"""
Utilities for frequently used command-line arguments and other main-script things.
"""

import argparse
import logging


def configure_logging(**kwargs):
    """
    You are advised to call this at the beginning of your main script, and use `logging.info()` throughout the program,
    instead of `print()`.

    Delegates to `logging.basicConfig()`. Accepts all the same arguments, but supplies some better defaults.
    """
    # Default options.
    options = {
        "level": logging.INFO,
        "format": "[%(asctime)s] [%(levelname)s] %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }
    options.update(kwargs)  # allow client's selections to override defaults
    logging.basicConfig(**options)


class HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """
    This class adds no new functionality, only is used to combine the existing functionality of two different
    formatters through multiple inheritance.
    """
    pass


def create_parser(desc):
    """
    A base parser with sensible default formatting.
    Args:
        desc (str): Description of the program.
    Returns:
        ArgumentParser: A new parser.
    """
    return argparse.ArgumentParser(description=desc, formatter_class=HelpFormatter)


def add_dataset_args(parser):
    """
    Add an argument for the user to specify a dataset.
    """
    parser.add_argument(
        "--dataset",
        choices=["omni", "miniimagenet"],
        type=str.lower,
        default="omni",
        help="The dataset to use."
    )
    return parser


def get_OML_dataset_sampler(parser, args):
    """
    Parses the dataset arguments, as given by `add_dataset_args()`. Also requires a `seed` argument.

    Args:
        parser (argparse.ArgumentParser): The argument parser.
        args (argparse.Namespace): The parsed args.
    Returns:
        ContinualMetaLearningSampler: A sampler for the user-specified dataset.
    """
    import datasets.mini_imagenet as imagenet
    import datasets.omniglot as omniglot

    if args.dataset == "omni":
        return omniglot.create_OML_sampler(root="../data/omni", seed=args.seed)
    elif args.dataset == "miniimagenet":
        return imagenet.create_OML_sampler(root="../data/mini-imagenet", seed=args.seed)
    else:
        parser.error(f"Unknown dataset: {args.dataset}")


class DeviceAction(argparse.Action):
    """
    A class for letting the user specify their PyTorch device.
    """
    # Import in this scope so clients can still use the other utilities in this module without Numpy/Torch.
    from torch import cuda

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, device, option_string=None):
        """
        For function definition, see: https://docs.python.org/3/library/argparse.html#action-classes

        Args:
            device: The Torch device string, in lowercase.
        """
        if device is None:
            device = "cuda" if self.cuda.is_available() else "cpu"
        elif device == "cuda" and not self.cuda.is_available():
            error_msg = "Torch says CUDA is not available. Remove it from your command to proceed on CPU."
            parser.error(error_msg)
        logging.info(f"Using device: {device}")
        setattr(namespace, self.dest, device)


def add_torch_args(parser, default_seed=None):
    """
    Adds arguments which may be useful for most programs that use PyTorch:
        - device
        - seed

    Args:
        parser (ArgumentParser): The parser to modify.
        default_seed (int or list[int] or None): Supply a custom seed if you want your program to be deterministic by
            default. Otherwise, defaults to true stochasticity.
    """
    parser.add_argument(
        "-d",
        "--device",
        choices=["cpu", "cuda"],
        type=str.lower,
        action=DeviceAction,
        help="Device to use for PyTorch.",
    )
    parser.add_argument("--seed", type=int, default=default_seed, help="Random seed.")
    return parser


def set_seed(seed):
    """
    Interprets the user's seed argument as given by `add_torch_args()` and seeds Python, NumPy, and PyTorch.
    """
    # Import in this scope so clients can still use the other utilities in this module without Torch.
    import numpy as np
    import random
    import torch

    if seed is None:
        logging.info(f"Using a non-deterministic random seed.")
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        logging.info(f"Using a fixed random seed: {seed}")


def add_wandb_args(parser):
    """
    Adds arguments which would be needed by any program that uses Weights & Biases:
        - project
        - entity
    """
    parser.add_argument("--project", help="Project to use for W&B logging.")
    parser.add_argument("--entity", help="Entity to use for W&B logging.")
    return parser
