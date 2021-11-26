"""
ANML Training Script
"""

import logging

import utils.argparsing as argutils
from anml import train


if __name__ == "__main__":
    # Training settings
    parser = argutils.create_parser("ANML training")

    argutils.add_dataset_arg(parser)
    parser.add_argument("--rln", metavar="NUM_CHANNELS", type=int, default=256,
                        help="Number of channels to use in the RLN.")
    parser.add_argument("--nm", metavar="NUM_CHANNELS", type=int, default=112,
                        help="Number of channels to use in the NM.")
    parser.add_argument("--train-size", metavar="INT", type=int, default=20,
                        help="Number of training examples to use in each inner loop.")
    parser.add_argument("--remember-size", metavar="INT", type=int, default=64,
                        help="Number of extra examples to add to training examples to compute the meta-loss.")
    parser.add_argument("--inner-lr", metavar="RATE", type=float, default=1e-1, help="Inner learning rate.")
    parser.add_argument("--outer-lr", metavar="RATE", type=float, default=1e-3, help="Outer learning rate.")
    parser.add_argument("--epochs", type=int, default=30000, help="Number of epochs to train (default: 30000).")
    argutils.add_device_arg(parser)
    argutils.add_seed_arg(parser, default_seed=1)
    argutils.add_verbose_arg(parser)

    args = parser.parse_args()
    argutils.configure_logging(args, level=logging.INFO)
    device = argutils.get_device(parser, args)
    argutils.set_seed_from_args(args)
    sampler, input_shape = argutils.get_OML_dataset_sampler(parser, args)

    logging.info("Commencing training.")
    train(
        sampler,
        input_shape,
        args.rln,
        args.nm,
        train_size=args.train_size,
        remember_size=args.remember_size,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        its=args.epochs,
        device=device,
        verbose=args.verbose,
    )
    logging.info("Training complete.")
