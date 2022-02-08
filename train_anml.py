"""
ANML Training Script
"""

import logging

import yaml

import utils.argparsing as argutils
from anml import train


if __name__ == "__main__":
    # Training settings
    parser = argutils.create_parser("ANML training")

    parser.add_argument("-c", "--config", metavar="PATH", type=argutils.existing_path, required=True,
                        help="Training config file.")
    argutils.add_dataset_arg(parser, add_train_size_arg=True)
    parser.add_argument("--batch-size", metavar="INT", type=int, default=1,
                        help="Number of examples per training batch in the inner loop.")
    parser.add_argument("--num-batches", metavar="INT", type=int, default=20,
                        help="Number of training batches in the inner loop.")
    parser.add_argument("--train-cycles", metavar="INT", type=int, default=1,
                        help="Number of times to run through all training batches, to comprise a single outer loop."
                             " Total number of gradient updates will be num_batches * train_cycles.")
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
    argutils.add_wandb_args(parser)
    argutils.add_verbose_arg(parser)
    parser.add_argument("--no-full-test", dest="full_test", action="store_false",
                        help="Do not test the full train/test sets before saving each model. These tests take a long"
                             " time so this is useful when saving models frequently or running quick tests. This"
                             " setting is implied if --smoke-test is enabled.")
    parser.add_argument("--st", "--smoke-test", dest="smoke_test", action="store_true",
                        help="Conduct a quick, full test of the training pipeline. If enabled, then a number of"
                             " arguments will be overridden to make the training run as short as possible and print in"
                             " verbose/debug mode.")

    args = parser.parse_args()

    # If we're doing a smoke test, then we need to modify the verbosity before configuring the logger.
    if args.smoke_test and args.verbose < 2:
        args.verbose = 2

    argutils.configure_logging(args, level=logging.INFO)

    with open(args.config, 'r') as f:
        config = yaml.full_load(f)

    # Command line args optionally override config.
    user_supplied_args = parser.get_user_specified_args()
    overrideable_args = ["dataset", "data_path", "download", "im_size", "train_size", "batch_size", "num_batches",
                         "train_cycles", "val_size", "remember_size", "remember_only", "inner_lr", "outer_lr",
                         "save_freq", "epochs", "seed", "id", "project", "entity", "full_test"]
    for arg in overrideable_args:
        # Only replace if value was explicitly specified by the user, or if the value doesn't already exist in config.
        if arg not in config or arg in user_supplied_args:
            config[arg] = getattr(args, arg)

    # Conduct a quick test.
    if args.smoke_test:
        config["batch_size"] = 1
        config["num_batches"] = 2
        config["train_cycles"] = 1
        if config.get("val_size", 0) > 2:
            config["val_size"] = 2
        config["epochs"] = 2
        config["save_freq"] = 1
        config["full_test"] = False

    device = argutils.get_device(parser, args)
    argutils.set_seed(config["seed"])

    argutils.prepare_wandb(config)

    sampler, input_shape = argutils.get_OML_dataset_sampler(config)

    logging.info("Commencing training.")
    train(sampler, input_shape, config, device, args.verbose)
    logging.info("Training complete.")
