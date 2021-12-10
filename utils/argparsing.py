"""
Utilities for frequently used command-line arguments and other main-script things.
"""

import argparse
import logging
from pathlib import Path


def configure_logging(parsed_args=None, **kwargs):
    """
    You are advised to call in your main script, after parsing the args. Then use `logging.info()` throughout the
    program, instead of `print()`.

    Delegates to `logging.basicConfig()`. Accepts all the same arguments, but supplies some better defaults. Also
    enables DEBUG level logging if the user supplies the `--verbose` argument (if using `add_verbose_arg()`).

    Args:
        parsed_args (argparse.Namespace): Arguments from command line, if desired.
        kwargs: Any arguments supported by `logging.basicConfig()`.
    """
    # Default options.
    options = {
        "level": logging.INFO,
        "format": "[%(asctime)s] [%(levelname)s] %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }
    options.update(kwargs)  # allow caller's selections to override defaults
    if getattr(parsed_args, "verbose", False):
        # allow command line user's preference to override all others
        options["level"] = logging.DEBUG
    logging.basicConfig(**options)

    # In addition to our own setup, let's make Pillow a little quieter, because it's very aggressive with the DEBUG
    # messages. See here: https://github.com/camptocamp/pytest-odoo/issues/15
    logging.getLogger("PIL").setLevel(logging.INFO)


class HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """
    This class adds no new functionality, only is used to combine the existing functionality of two different
    formatters through multiple inheritance.
    """
    pass


def create_parser(desc, allow_abbrev=True):
    """
    A base parser with sensible default formatting.
    Args:
        desc (str): Description of the program.
        allow_abbrev (bool): An argument to the ArgumentParser constructor; whether to allow long options to be
            abbreviated.
    Returns:
        ArgumentParser: A new parser.
    """
    return argparse.ArgumentParser(description=desc, formatter_class=HelpFormatter, allow_abbrev=allow_abbrev)


def add_verbose_arg(parser):
    """
    Adds an argument which turns on verbose logging, if using `configure_logging()`.

    Args:
        parser (ArgumentParser): The parser to modify.
    """
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Use verbose logging. Use this flag multiple times to request extra verbosity.")
    return parser


def add_dataset_arg(parser):
    """
    Add an argument for the user to specify a dataset.
    """
    parser.add_argument("--dataset", choices=["omni", "miniimagenet"], type=str.lower, default="omni",
                        help="The dataset to use.")
    parser.add_argument("--data-path", "--data-dir", metavar="PATH", type=Path, default="../data",
                        help="The root path in which to look for the dataset (or store a new one if it isn't already"
                             " present).")
    parser.add_argument("--no-download", dest="download", action="store_false",
                        help="Do not download the dataset automatically if it doesn't already exist; raise an error.")
    return parser


def get_OML_dataset_sampler(parser, args, im_size=None, greyscale=True):
    """
    Parses the dataset arguments, as given by `add_dataset_args()`. Also requires a `seed` argument.

    Args:
        parser (argparse.ArgumentParser): The argument parser.
        args (argparse.Namespace): The parsed args.
        im_size (int): Image size (single integer, to be used as height and width).
        greyscale (bool): Whether to convert images to greyscale.
    Returns:
        ContinualMetaLearningSampler: A sampler for the user-specified dataset.
        tuple: The shape of the images that will be returned by the sampler (they will all be the same size).
    """
    import datasets.mini_imagenet as imagenet
    import datasets.omniglot as omniglot

    if args.dataset == "omni":
        if not greyscale:
            raise ValueError("Omniglot is only available in greyscale.")
        return omniglot.create_OML_sampler(root=args.data_path / "omni", download=args.download, im_size=im_size,
                                           seed=args.seed)
    elif args.dataset == "miniimagenet":
        return imagenet.create_OML_sampler(root=args.data_path / "mini-imagenet", download=args.download,
                                           im_size=im_size, seed=args.seed)
    else:
        parser.error(f"Unknown dataset: {args.dataset}")


def add_device_arg(parser):
    """
    Adds an argument which allows the user to specify their device to use for PyTorch.

    Args:
        parser (ArgumentParser): The parser to modify.
    """
    parser.add_argument("-d", "--device", choices=["cpu", "cuda"], type=str.lower, help="Device to use for PyTorch.", )
    return parser


def get_device(parser, parsed_args):
    """
    Get the PyTorch device from args, for use with `add_device_arg()`.
    Args:
        parser (ArgumentParser): The parser which parsed the args.
        parsed_args (argparse.Namespace): Arguments from command line.
    """
    # Import in this scope so clients can still use the other utilities in this module without Numpy/Torch.
    import torch

    if parsed_args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif parsed_args.device == "cuda" and not torch.cuda.is_available():
        error_msg = "Torch says CUDA is not available. Remove it from your command to proceed on CPU."
        parser.error(error_msg)  # Exits.
        device = "invalid"  # Unreachable, but silences a warning.
    else:
        device = parsed_args.device

    logging.info(f"Using device: {device}")
    return device


def add_seed_arg(parser, default_seed=None):
    """
    Adds an argument which allows the user to specify a seed for deterministic random number generation.

    Args:
        parser (ArgumentParser): The parser to modify.
        default_seed (int or list[int] or None): Supply a custom seed if you want your program to be deterministic by
            default. Otherwise, defaults to true stochasticity.
    """
    parser.add_argument("--seed", type=int, default=default_seed, help="Random seed.")
    return parser


def set_seed(seed):
    """
    Seeds Python, NumPy, and PyTorch random number generators.
    """
    # Import in this scope so clients can still use the other utilities in this module without Torch.
    import numpy as np
    import random
    import torch

    if seed is None:
        logging.info(f"Using a non-deterministic random seed.")
    else:
        random.seed(seed)
        # Mask out higher bits, b/c the two RNGs below can't handle larger than 32-bit seeds. We still need to support
        # larger seeds because newer NumPy code might have used a larger seed and we may want to reproduce that result.
        seed32 = seed & (2 ** 32 - 1)
        np.random.seed(seed32)
        torch.manual_seed(seed32)
        addl_str = ""
        if seed != seed32:
            addl_str = f" (Torch and legacy NumPy will use the 32-bit version: {seed32})"
        logging.info(f"Using a fixed random seed: {seed}" + addl_str)


def set_seed_from_args(parsed_args):
    """
    Interprets the user's seed argument as given by `add_torch_args()` and seeds Python, NumPy, and PyTorch.
    Args:
        parsed_args (argparse.Namespace): Arguments from command line.
    """
    set_seed(parsed_args.seed)


def add_wandb_args(parser):
    """
    Adds arguments which would be needed by any program that uses Weights & Biases:
        - project
        - entity
    """
    parser.add_argument("--project", help="Project to use for W&B logging.")
    parser.add_argument("--entity", help="Entity to use for W&B logging.")
    return parser
