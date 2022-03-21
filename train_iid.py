"""
Standard Training Script
"""
# NOTE: Use one of the following commands to test the functionality of this script:
#   time WANDB_MODE=disabled DEBUG=Y python train_iid.py -c configs/train-omni-iid-sanml.yml --st
#   time WANDB_MODE=disabled DEBUG=Y python train_iid.py -c configs/train-omni-iid-sanml.yml --train-size 15 --epochs 1 --no-full-test -vv --group mygroup
#   time python train_iid.py -c configs/train-omni-iid-sanml.yml
# Which you use depends on how much of the pipeline you actually want to test. You can further remove the `DEBUG` and
# `WANDB_MODE` flags to actually test launching eval jobs and reporting results to W&B.

import logging
import sys

import utils.argparsing as argutils
from iid import train


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
    parser.add_argument("--batch-size", metavar="INT", type=int, default=1,
                        help="Number of examples per training batch in the inner loop.")
    parser.add_argument("--lr", metavar="RATE", type=float, default=0.1, help="Global learning rate.")
    parser.add_argument("--epochs", type=int, default=90, help="Number of epochs to train.")
    parser.add_argument("--save-freq", type=int, default=1000, help="Number of steps between each saved model.")
    argutils.add_device_arg(parser)
    argutils.add_seed_arg(parser, default_seed=1)
    argutils.add_wandb_args(parser, allow_id=allow_id)
    argutils.add_verbose_arg(parser)
    parser.add_argument("--no-full-test", dest="full_test", action="store_false",
                        help="Do not test the full train/test sets before saving each model. These tests take a long"
                             " time so this is useful when saving models frequently or running quick tests. This"
                             " setting is implied if --smoke-test is enabled.")
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

    overrideable_args = ["dataset", "data_path", "download", "im_size", "train_size", "batch_size", "lr", "epochs",
                         "save_freq", "device", "seed", "id", "project", "entity", "group", "full_test", "eval_steps",
                         "cluster"]
    config = argutils.load_config_from_args(parser, args, overrideable_args)

    # Conduct a quick test.
    if args.smoke_test:
        config["batch_size"] = 256
        config["train_size"] = 1
        config["max_steps"] = 1
        config["epochs"] = 1
        config["save_freq"] = 1
        config["full_test"] = False
        config["eval_steps"] = []

    return config


def setup_and_train(parser, config, verbose):
    """ Setup W&B, load data, and commence training. """
    device = argutils.get_device(parser, config)
    argutils.set_seed(config["seed"])

    # Keep this before we load the dataset b/c we want to use a dataset location that's relative to the run directory.
    # The prepare_wandb function will change our run directory.
    argutils.prepare_wandb(config, job_type="train")

    sampler, input_shape = argutils.get_dataset_sampler(config, sampler_type="iid")

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
