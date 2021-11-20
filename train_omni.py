"""
ANML Training Script
"""

import logging

import utils.argparsing as argutils
from anml import train


if __name__ == "__main__":
    argutils.configure_logging(level=logging.INFO)

    # Training settings
    parser = argutils.create_parser("ANML training")
    argutils.add_dataset_args(parser)
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
    argutils.add_torch_args(parser, default_seed=1)

    args = parser.parse_args()

    argutils.set_seed(args.seed)
    sampler = argutils.get_OML_dataset_sampler(parser, args)

    logging.info("Commencing training.")
    train(
        sampler,
        args.rln,
        args.nm,
        args.mask,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        its=args.epochs,
        device=args.device,
    )
    logging.info("Training complete.")
