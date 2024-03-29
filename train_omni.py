import argparse
import logging
import os
import sys

import torch
from torch import manual_seed

from anml import train

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="ANML training")
    parser.add_argument("--rln", type=int, default=256, help="number of channels to use in the RLN")
    parser.add_argument("--nm", type=int, default=112, help="number of channels to use in the NM")
    parser.add_argument(
        "--mask",
        type=int,
        default=2304,
        help="size of the modulatory mask, needs to match extracted features size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30000,
        help="number of epochs to train (default: 30000)",
    )
    parser.add_argument(
        "--inner-lr",
        type=float,
        default=1e-1,
        help="inner learning rate (default: 1e-1)",
    )
    parser.add_argument(
        "--outer-lr",
        type=float,
        default=1e-3,
        help="outer learning rate (default: 1e-3)",
    )
    parser.add_argument("-d", "--device", choices=["cpu", "cuda"], type=str.lower, help="Device to use for PyTorch.")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        logging.error("Torch says CUDA is not available. Remove it from your command to proceed on CPU.")
        sys.exit(os.EX_UNAVAILABLE)
    logging.info(f"Using device: {device}")

    manual_seed(args.seed)

    logging.info("Commencing training.")
    train(
        args.rln,
        args.nm,
        args.mask,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        its=args.epochs,
        device=device,
    )
