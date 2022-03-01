"""
Utilities for frequently used command-line arguments and other main-script things.
"""

import argparse
import json
import logging
import os
import pwd
import sys
from datetime import datetime
from pathlib import Path

from utils import load_yaml, make_pretty


def resolved_path(str_path):
    """
    This function can be used as an argument type to fully resolve a user-supplied path:
        parser.add_argument(..., type=argutils.resolved_path, ...)
    The path may not exist, but if it is a relative path it will become fully resolved.

    Args:
        str_path: The user-supplied path.

    Returns:
        pathlib.Path: The fully-resolved path object.
    """
    return Path(str_path).resolve()


def existing_path(str_path):
    """
    This function can be used as an argument type to fully resolve a user-supplied path and ensure it exists:
        parser.add_argument(..., type=argutils.existing_path, ...)
    An exception will be raised if the path does not exist.

    Args:
        str_path: The user-supplied path.

    Returns:
        pathlib.Path: The fully-resolved path object, if it exists.
    """
    path = Path(str_path).resolve()
    if path.exists():
        return path
    else:
        raise argparse.ArgumentTypeError(f"{str_path} ({path}) is not a valid path")


def args_as_dict(parsed_args):
    """
    Returns a copy of the given object as a dictionary.
    Args:
        parsed_args (argparse.Namespace or dict): The args to copy.
    Returns:
        dict: The arguments as a dictionary.
    """
    # Turn namespace into dict.
    if isinstance(parsed_args, argparse.Namespace):
        # Grab all args because we will store them later if `save_args` is enabled.
        return vars(parsed_args)
    else:
        # Do not modify the config that was passed in.
        return parsed_args.copy()


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

    # We want to keep the storage module in INFO mode unless we really want to debug it.
    logging.getLogger("utils.storage").setLevel(logging.INFO)

    # In addition to our own setup, let's make Pillow a little quieter, because it's very aggressive with the DEBUG
    # messages. See here: https://github.com/camptocamp/pytest-odoo/issues/15
    logging.getLogger("PIL").setLevel(logging.INFO)


class HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """
    This class adds no new functionality, only is used to combine the existing functionality of two different
    formatters through multiple inheritance.
    """
    pass


class ActionWrapper(argparse.Action):
    """
    A wrapper class which is used to detect which arguments were explicitly supplied by the user.
    """
    def __init__(self, action):
        super().__init__(**dict(action._get_kwargs()))
        self.action = action
        self.user_invoked = False

    def __call__(self, parser, namespace, values, option_string=None):
        self.user_invoked = True
        self.action(parser, namespace, values, option_string)

    def format_usage(self):
        return self.action.format_usage()


class ArgParser(argparse.ArgumentParser):
    """
    An ArgumentParser which provides one extra piece of functionality: it can tell whether the user explicitly supplied
    an argument on the command-line. It can tell the difference between when the default value is used and when the user
    explicitly supplies the default.
    """
    def _add_action(self, action):
        action = ActionWrapper(action)
        return super()._add_action(action)

    def add_argument_group(self, *args, **kwargs):
        # HACK: We are monkey-patching the group here so we can inject our action wrappers.
        group = super().add_argument_group(*args, **kwargs)
        group._add_action_orig = group._add_action
        group._add_action = lambda action: group._add_action_orig(ActionWrapper(action))
        return group

    def add_mutually_exclusive_group(self, **kwargs):
        # HACK: We are monkey-patching the group here so we can inject our action wrappers.
        group = super().add_mutually_exclusive_group(**kwargs)
        group._add_action_orig = group._add_action
        group._add_action = lambda action: group._add_action_orig(ActionWrapper(action))
        return group

    def get_user_specified_args(self):
        return [a.dest for a in self._actions if a.user_invoked]

    def reset_user_specified_args(self):
        """If you wish to use the parser multiple times, you must call this function before each usage."""
        for a in self._actions:
            a.user_invoked = False


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
    return ArgParser(description=desc, formatter_class=HelpFormatter, allow_abbrev=allow_abbrev)


def load_config_from_args(parser, parsed_args, overrideable_args):
    """
    Load a config from the `--config` argument, and then overwrite config values with any values that the user supplied
    on the command line. The list of keys that should be taken from the command line is given by `overrideable_args`. If
    the key isn't already present in the config, it will be taken from the args. Otherwise, it will only be taken from
    the args **if the user chose a non-default value**.

    Args:
        parser (ArgParser): This must be the local ArgParser type; not just any argparse.ArgumentParser.
        parsed_args (argparse.Namespace): The arguments from the command line.
        overrideable_args (list[str]): A list of keys that can be optionally overwritten with command-line values.

    Returns:
        dict: The parsed config object.
    """
    config = load_yaml(parsed_args.config)

    # Command line args optionally override config.
    user_supplied_args = parser.get_user_specified_args()
    for arg in overrideable_args:
        # Only replace if value was explicitly specified by the user, or if the value doesn't already exist in config.
        if arg not in config or arg in user_supplied_args:
            config[arg] = getattr(parsed_args, arg, None)

    config = make_pretty(config)
    return config


def add_verbose_arg(parser):
    """
    Adds an argument which turns on verbose logging, if using `configure_logging()`.

    Args:
        parser (ArgumentParser): The parser to modify.
    """
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Use verbose logging. Use this flag multiple times to request extra verbosity.")
    return parser


def add_dataset_arg(parser, dflt_data_dir="experiments/data", add_resize_arg=True, add_train_size_arg=False):
    """
    Add an argument for the user to specify a dataset.
    """
    parser.add_argument("--dataset", choices=["omni", "miniimagenet"], type=str.lower, default="omni",
                        help="The dataset to use.")
    parser.add_argument("--data-path", "--data-dir", metavar="PATH", type=resolved_path, default=dflt_data_dir,
                        help="The root path in which to look for the dataset (or store a new one if it isn't already"
                             " present).")
    parser.add_argument("--no-download", dest="download", action="store_false",
                        help="Do not download the dataset automatically if it doesn't already exist; raise an error.")
    if add_resize_arg:
        parser.add_argument("--im-size", metavar="PX", type=int, default=None,
                            help="Resize all input images to the given size (in pixels).")
    if add_train_size_arg:
        parser.add_argument("--train-size", metavar="INT", type=int, default=500,
                            help="Number of examples per class to use in training split. Remainder (if any) will be"
                                 " reserved for validation.")
    return parser


def get_OML_dataset_sampler(args, greyscale=None):
    """
    Parses the dataset arguments, as given by `add_dataset_args()`. Also requires a `seed` argument.

    Args:
        args (argparse.Namespace or dict): The parsed args.
        greyscale (bool): Whether to convert images to greyscale, or None to use the default coloring.
    Returns:
        ContinualMetaLearningSampler: A sampler for the user-specified dataset.
        tuple: The shape of the images that will be returned by the sampler (they will all be the same size).
    """
    import datasets.mini_imagenet as imagenet
    import datasets.omniglot as omniglot

    # Turn namespace into dict.
    if isinstance(args, argparse.Namespace):
        old_args = args
        args = {}
        for k in ("dataset", "data_path", "download", "im_size", "train_size", "seed"):
            args[k] = getattr(old_args, k, None)
    else:
        # Do not modify the config that was passed in.
        args = args.copy()

    # These args are allowed to be missing.
    for arg in ("im_size", "train_size", "seed"):
        args.setdefault(arg)
    # Ensure we have a Path type here.
    args["data_path"] = Path(args["data_path"])

    if args["dataset"] == "omni":
        if greyscale is False:
            raise ValueError("Omniglot is only available in greyscale.")
        return omniglot.create_OML_sampler(root=args["data_path"] / "omni", download=args["download"],
                                           im_size=args["im_size"], train_size=args["train_size"], seed=args["seed"])
    elif args["dataset"] == "miniimagenet":
        return imagenet.create_OML_sampler(root=args["data_path"] / "mini-imagenet", download=args["download"],
                                           im_size=args["im_size"], greyscale=greyscale, train_size=args["train_size"],
                                           seed=args["seed"])
    else:
        raise ValueError(f"Unknown dataset: {args['dataset']}")


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
        parsed_args (argparse.Namespace or dict): Arguments from command line or config.
    """
    # Import in this scope so clients can still use the other utilities in this module without Numpy/Torch.
    import torch

    parsed_args = args_as_dict(parsed_args)

    if parsed_args.get("device") is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif parsed_args.get("device") == "cuda" and not torch.cuda.is_available():
        error_msg = "Torch says CUDA is not available. Remove it from your command to proceed on CPU."
        parser.error(error_msg)  # Exits.
        device = "invalid"  # Unreachable, but silences a warning.
    else:
        device = parsed_args.get("device")

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


def add_wandb_args(parser, allow_id=False):
    """
    Adds arguments which would be needed by any program that uses Weights & Biases:
        - project
        - entity
        - [optional] id
    """
    id_text = " Ignored if --id is used." if allow_id else ""
    parser.add_argument("--project", default="higherANML", help="Project to use for W&B logging." + id_text)
    parser.add_argument("--entity", help="Entity to use for W&B logging." + id_text)
    parser.add_argument("--group", help="Name under which to group this run in W&B.")
    if allow_id:
        parser.add_argument("--id", help="ID to use for W&B logging. If this project already exists, it will be resumed.")
    return parser


def get_user():
    return pwd.getpwuid(os.getuid())[0]


def get_hostname():
    return os.uname().nodename


def get_folder():
    return os.path.realpath(os.path.dirname(sys.argv[0]))


def get_path():
    return os.path.realpath(sys.argv[0])


def get_location():
    user = get_user()
    host = get_hostname()
    path = get_path()
    loc = f"{user}@{host}:{path}"
    return loc


def prepare_wandb(parsed_args, job_type=None, create_folder=True, root_path="experiments", save_args=False,
                  allow_reinit=None, dry_run=False):
    """
    Calls `wandb.init()` and (optionally) sets up the result folder, based on the arguments from `add_wandb_args()`.

    If the `--id` argument was supplied, we assume we are already in the target folder. Otherwise we create and move to
    the target folder, if `create_folder` is `True`.

    Args:
        parsed_args (argparse.Namespace or dict): Arguments from command line.
        job_type (str): The type of program creating this run, such as "train" or "eval".
        create_folder (bool): Whether to create a folder for this run.
        root_path (str): The root path in which to create result folders. Not used if `args.id` is present.
        save_args (bool): Whether to also save the arguments locally in the result folder. They will be saved on W&B
            regardless.
        allow_reinit (bool): If true, you may call this function multiple times; see the `reinit` argument to
            `wandb.init()`.
        dry_run (bool): If true, don't actually take actions, just print what actions would be taken.

    Returns:
        wandb.run: The run object created by `wandb.init()`.
    """
    import wandb

    parsed_args = args_as_dict(parsed_args)

    parsed_args["job_type"] = job_type
    parsed_args["location"] = get_location()
    parsed_args["date"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    if not dry_run:
        kwargs = {
            "config": parsed_args,
            "group": parsed_args.get("group"),
            "job_type": job_type,
            "reinit": allow_reinit,
        }
        if parsed_args.get("id"):
            kwargs["id"] = parsed_args["id"]
        else:
            kwargs["entity"] = parsed_args["entity"]
            kwargs["project"] = parsed_args["project"]
        run = wandb.init(**kwargs)
    else:
        from collections import namedtuple
        Run = namedtuple("Run", ["id", "config", "project", "name"])
        run = Run("abcd1234", {"foo": "bar"}, parsed_args.get("project", get_user()), "fake-name-8")
        if parsed_args.get("id", None):
            print(f"Would overwrite an existing W&B run with ID={parsed_args['id']}.")
        else:
            print(f"Would launch a new W&B run.")

    if create_folder and not parsed_args.get("id", None):
        # Only create a new folder if the ID wasn't pre-existing.
        folder = (Path(root_path) / run.project / run.name).resolve()
        if not dry_run:
            folder.mkdir(parents=True, exist_ok=True)
            os.chdir(folder)
        else:
            print(f"Would create output folder: {folder}. Subsequent actions would be relative to this folder instead"
                  f" of {os.getcwd()}.")

    if save_args:
        # We are now in the output folder, so we can save directly there.
        args_file = Path("wandb-run.json")
        if not dry_run:
            with open(args_file, "w") as f:
                json.dump(dict(run.config), f, indent=2)
        else:
            print(f"Would save config to file: {args_file}")

    return run
