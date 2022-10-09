"""
ANML Training Script
"""
# NOTE: Use one of the following commands to test the functionality of this script:
#   time python train_anml.py -c configs/train-omni-anml.yml --st
#   time WANDB_MODE=disabled DEBUG=Y python train_anml.py -c configs/train-omni-anml.yml --val-sample-size 64 --epochs 10 --no-full-test --eval-steps -vv --group mygroup
#   time WANDB_MODE=disabled DEBUG=Y python train_anml.py -c configs/train-omni-anml.yml --val-sample-size 64 --epochs 1 --no-full-test -vv --group mygroup
#   time python train_anml.py -c configs/train-omni-anml.yml
# Which you use depends on how much of the pipeline you actually want to test. You can further remove the `DEBUG` and
# `WANDB_MODE` flags to actually test launching eval jobs and reporting results to W&B.

import logging
import sys

import utils.argparsing as argutils
from anml import train


def create_arg_parser(desc, allow_abbrev=True, allow_id=True):
    """
    Creates the argument parser for this program.

    Args:
        desc (str): The human-readable description for the arg parser.
        allow_abbrev (bool): The `allow_abbrev` argument to `argparse.ArgumentParser()`.
        allow_id (bool): The `allow_id` argument to the `argutils.add_wandb_args()` function.

    Returns:
        argutils.ArgParser: The parser.
    """
    parser = argutils.create_parser(desc, allow_abbrev=allow_abbrev)
    parser.add_argument("-c", "--config", metavar="PATH", type=argutils.existing_path, required=True,
                        help="Training config file.")
    argutils.add_dataset_arg(parser, add_train_size_arg=True)
    parser.add_argument("--train-method", choices=["meta", "sequential_episodic"], type=str.lower, default="meta",
                        help="Training method to use.")
    parser.add_argument("--sample-method", choices=["single", "uniform"], type=str.lower, default="single",
                        help="Method to use for sampling inner loop examples.")
    parser.add_argument("--batch-size", metavar="INT", type=int, default=1,
                        help="Number of examples per training batch in the inner loop.")
    parser.add_argument("--num-batches", metavar="INT", type=int, default=20,
                        help="Number of training batches in the inner loop.")
    parser.add_argument("--train-cycles", metavar="INT", type=int, default=1,
                        help="Number of times to run through all training batches, to comprise a single outer loop."
                             " Total number of gradient updates will be num_batches * train_cycles.")
    parser.add_argument("--val-sample-size", metavar="INT", type=int, default=200,
                        help="Total number of test examples to sample from the validation set each iteration (for"
                             " testing generalization to never-seen examples from the training domain).")
    parser.add_argument("--remember-size", metavar="INT", type=int, default=64,
                        help="Number of randomly sampled training examples to compute the meta-loss.")
    parser.add_argument("--remember-only", action="store_true",
                        help="Do not include the training examples from the inner loop into the meta-loss (only use"
                             " the remember set for the outer loop of training).")
    parser.add_argument("--no-lobotomize", dest="lobotomize", action="store_false",
                        help="Do not lobotomize. Do not reset the weights of the logits of a class just before learning"
                             " that class. (See code for explanation.)")
    parser.add_argument("--inner-lr", metavar="RATE", type=float, default=1e-1, help="Inner learning rate.")
    parser.add_argument("--outer-lr", metavar="RATE", type=float, default=1e-3, help="Outer learning rate.")
    parser.add_argument("--save-freq", type=int, default=1000, help="Number of epochs between each saved model.")
    parser.add_argument("--epochs", type=int, default=25000, help="Number of epochs to train.")
    argutils.add_device_arg(parser)
    argutils.add_seed_arg(parser, default_seed=1)
    argutils.add_wandb_args(parser, allow_id=allow_id)
    argutils.add_verbose_arg(parser)
    parser.add_argument("--no-full-test", dest="full_test", action="store_false",
                        help="Do not test the full train/test sets before saving each model. These tests take a long"
                             " time so this is useful when saving models frequently or running quick tests. This"
                             " setting is implied if --smoke-test is enabled.")
    parser.add_argument("--save-initial-model", action="store_true",
                        help="Save the state of the model just after initialization, before any training.")
    parser.add_argument("--eval-steps", metavar="INT", nargs="*", type=int,
                        help="Points in the training at which the model should be fully evaluated. At each of these"
                             " steps, the model will be saved and a full evaluation will be run (in a separate Slurm"
                             " job). The result of the evaluation will be recorded in the same W&B group. To report the"
                             " final trained model, enter any number larger than --epochs.")
    parser.add_argument("--cluster", metavar="NAME", default="dggpu",
                        help="The cluster on which to launch eval jobs. This must correspond to one of the resources in"
                             " your Neuromanager config.")
    parser.add_argument("--st", "--smoke-test", dest="smoke_test", action="store_true",
                        help="Conduct a quick, full test of the training pipeline. If enabled, then a number of"
                             " arguments will be overridden to make the training run as short as possible and print in"
                             " verbose/debug mode.")
    return parser


def prep_config(parser, args):
    """ Process command line arguments to produce a full training config. May also edit the arguments. """
    # If we're doing a smoke test, then we need to modify the verbosity before configuring the logger.
    if args.smoke_test and args.verbose < 2:
        args.verbose = 2

    argutils.configure_logging(args, level=logging.INFO)

    overrideable_args = ["dataset", "data_path", "download", "im_size", "train_size", "val_size", "augment",
                         "train_method", "sample_method", "batch_size", "num_batches", "train_cycles",
                         "val_sample_size", "remember_size", "remember_only", "lobotomize", "inner_lr", "outer_lr",
                         "save_freq", "epochs", "device", "seed", "id", "project", "entity", "group", "full_test",
                         "save_initial_model", "eval_steps", "cluster"]
    config = argutils.load_config_from_args(parser, args, overrideable_args)

    # Conduct a quick test.
    if args.smoke_test:
        config["batch_size"] = 1
        config["num_batches"] = 2
        config["train_cycles"] = 1
        if config.get("val_sample_size", 0) > 2:
            config["val_sample_size"] = 2
        config["epochs"] = 1
        config["save_freq"] = 1
        config["full_test"] = False
        config["save_initial_model"] = False
        config["eval_steps"] = []

    return config


def setup_and_train(parser, config, verbose):
    """ Setup W&B, load data, and commence training. """
    device = argutils.get_device(parser, config)
    argutils.set_seed(config["seed"])

    # Keep this before we load the dataset b/c we want to use a dataset location that's relative to the run directory.
    # The prepare_wandb function will change our run directory.
    argutils.prepare_wandb(config, job_type="train", autogroup=True)

    sampler, input_shape = argutils.get_dataset_sampler(config)

    logging.info("Commencing training.")
    train(sampler, input_shape, config, device, verbose)
    logging.info("Training complete.")


def main(argv=None):
    parser = create_arg_parser(__doc__)
    args = parser.parse_args(argv)

    config = prep_config(parser, args)

    setup_and_train(parser, config, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
