import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from datasets.OmniSampler import OmniSampler
from anml import test_train

warnings.filterwarnings("ignore")


def check_path(path):
    if Path(path).exists():
        return path
    else:
        raise argparse.ArgumentTypeError(f"model:{path} is not a valid path")


def repeats(runs, path, classes, train_examples, lr, device):

    omni_sampler = OmniSampler(root="../data/omni")

    run = lambda: test_train(
        path,
        sampler=omni_sampler,
        num_classes=classes,
        train_examples=train_examples,
        device=device,
        lr=lr,
    )

    results = []
    for _ in trange(runs):
        results.append(run().mean())

    print(
        f"Classes {classes} Accuracy {np.mean(results):.2f} (std {np.std(results):.2f})"
    )


if __name__ == "__main__":
    # Training setting
    parser = argparse.ArgumentParser(description="ANML training")

    parser.add_argument(
        "-l",
        "--lr",
        type=float,
        help="learning rate to use (check README for suggestions)",
    )
    parser.add_argument(
        "-c", "--classes", type=int, help="number of classes to test",
    )
    parser.add_argument(
        "-r", "--runs", type=int, help="number of repetitions to run",
    )
    parser.add_argument(
        "-t",
        "--train_examples",
        type=int,
        default=15,
        help="how many examples to use for training (max 20, default 15)",
    )
    parser.add_argument(
        "-m", "--model", type=check_path, help="path to the model to use"
    )
    parser.add_argument("-d", "--device", type=str, default=None, help="cuda/cpu")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device.lower() == "cuda" and not torch.cuda.is_available():
        print("Torch says CUDA is not available. Remove it from your command to proceed on CPU.", file=sys.stderr)
        sys.exit(os.EX_UNAVAILABLE)

    repeats(
        runs=args.runs,
        path=args.model,
        classes=args.classes,
        train_examples=args.train_examples,
        lr=args.lr,
        device=device,
    )
