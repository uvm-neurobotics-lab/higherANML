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
    parser.add_argument("--batch-size", metavar="INT", type=int, default=1,
                        help="Number of examples per training batch in the inner loop.")
    parser.add_argument("--num-batches", metavar="INT", type=int, default=20,
                        help="Number of training batches in the inner loop.")
    parser.add_argument("--train-cycles", metavar="INT", type=int, default=1,
                        help="Number of times to run through all training batches, to comprise a single outer loop."
                        " Total number of gradient updates will be num_batches * train_cycles.")
    parser.add_argument("--train-size", metavar="INT", type=int, default=500,
                        help="Number of examples per class to use in training split. Remainder (if any) will be"
                             " reserved for validation.")
    parser.add_argument("--val-size", metavar="INT", type=int, default=200,
                        help="Total number of test examples to sample from the validation set each iteration (for"
                             " testing generalization to never-seen examples).")
    parser.add_argument("--remember-size", metavar="INT", type=int, default=64,
                        help="Number of randomly sampled training examples to compute the meta-loss.")
    parser.add_argument("--remember-only", action="store_true",
                        help="Do not include the training examples from the inner loop into the meta-loss (only use"
                             " the remember set for the outer loop of training).")
    parser.add_argument("--inner-lr", metavar="RATE", type=float, default=1e-1, help="Inner learning rate.")
    parser.add_argument("--outer-lr", metavar="RATE", type=float, default=1e-3, help="Outer learning rate.")
    parser.add_argument("--save-freq", type=int, default=1000, help="Number of epochs between each saved model.")
    parser.add_argument("--epochs", type=int, default=30000, help="Number of epochs to train.")
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
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        train_size=args.train_size,
        val_size=args.val_size,
        remember_size=args.remember_size,
        remember_only=args.remember_only,
        train_cycles=args.train_cycles,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        its=args.epochs,
        save_freq=args.save_freq,
        device=device,
        verbose=args.verbose,
    )
    logging.info("Training complete.")
