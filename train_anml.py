"""
ANML Training Script
"""

import logging
import sys

import utils.argparsing as argutils
from anml import train


if __name__ == "__main__":
    # Training settings
    parser = argutils.create_parser("ANML training")
    argutils.add_dataset_args(parser)
    parser.add_argument("--rln", type=int, default=256, help="number of channels to use in the RLN")
    parser.add_argument("--nm", type=int, default=112, help="number of channels to use in the NM")
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
    argutils.add_torch_args(parser, default_seed=1)
    argutils.add_verbose_arg(parser)

    args = parser.parse_args()
    argutils.configure_logging(args, level=logging.INFO)
    argutils.set_seed_from_args(args)
    sampler, input_shape = argutils.get_OML_dataset_sampler(parser, args)

    logging.info("Commencing training.")
    train(
        sampler,
        input_shape,
        args.rln,
        args.nm,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        its=args.epochs,
        device=args.device,
        verbose=args.verbose,
    )
    logging.info("Training complete.")
