"""
The "map" portion of a map-reduce job for evaluating ANML and related models.
"""
# NOTE: Use the following command to test the functionality of this script:
#   WANDB_MODE=disabled python eval_map.py -c configs/eval-omni-anml.yml --model trained_anmls/anml-1-28-28-29999.net --output test.pkl --classes 10 --lr 0.0015 --group mygroup
# You should get a final accuracy somewhere around:
#   Train 96.8% (std: 3.4%) | Test 92.6% (std: 6.2%)
# Other learning rates will result in lower performance.
# Optionally add `--method iid` to test standard transfer learning ("oracle" style).
# Optionally add `--eval-freq 2` to test a full trajectory of evaluation numbers. This will take longer and will print
# the same result, but more data will end up in the resulting dataframe.

import argparse
import sys
from pathlib import Path

import yaml

import utils.argparsing as argutils
from anml import run_full_test as seq_test
from iid import run_full_test as iid_test


def check_path(path):
    if Path(path).exists():
        return path
    else:
        raise argparse.ArgumentTypeError(f"model: {path} is not a valid path")


def main(args=None):
    parser = argutils.create_parser(__doc__)

    parser.add_argument("-c", "--config", metavar="PATH", type=argutils.existing_path, required=True,
                        help="Evaluation config file.")
    parser.add_argument("--eval-method", choices=("sequential", "seq", "iid"), default="sequential",
                        help="The testing method to use: sequential (continual learning) or i.i.d. (standard transfer"
                             " learning.")
    parser.add_argument("--reinit-method", choices=("kaiming", "lstsq"), default="kaiming",
                        help="The method to use to reinitialize trainable parameters: typical kaiming normal"
                             "initialization or least squares estimate of the final linear layer.")
    argutils.add_dataset_arg(parser)
    parser.add_argument("-m", "--model", metavar="PATH", type=check_path, help="Path to the model to evaluate.")
    parser.add_argument("-l", "--lr", metavar="RATE", type=float,
                        help="Learning rate to use (check README for suggestions).")
    parser.add_argument("--classes", metavar="INT", type=int, help="Number of classes to test.")
    parser.add_argument("--train-examples", metavar="INT", type=int, default=15,
                        help="Number of examples per class, for training.")
    parser.add_argument("--test-examples", metavar="INT", type=int, default=5,
                        help="Number of examples per class, for testing.")
    parser.add_argument("--epochs", metavar="INT", type=int, default=1,
                        help="Number of epochs to fine-tune for. Only used in i.i.d. testing.")
    parser.add_argument("--batch-size", metavar="INT", type=int, default=256,
                        help="Size of batches to train on. Only used in i.i.d. testing.")
    parser.add_argument("--eval-freq", metavar="INT", type=int,
                        help="The frequency at which to evaluate performance of the model throughout the learning"
                             " process. This can be very expensive, if evaluating after every class learned (freq = 1)."
                             " By default we evaluate only at the very end.")
    parser.add_argument("-r", "--runs", metavar="INT", type=int, default=10, help="Number of repetitions to run.")
    parser.add_argument("-o", "--output", metavar="PATH", help="The location to save to.")
    argutils.add_device_arg(parser)
    argutils.add_seed_arg(parser)
    argutils.add_wandb_args(parser)
    argutils.add_verbose_arg(parser)

    args = parser.parse_args(args)
    argutils.configure_logging(args)
    overrideable_args = ["eval_method", "reinit_method", "dataset", "data_path", "download", "im_size", "augment",
                         "model", "classes", "train_examples", "test_examples", "epochs", "batch_size", "lr",
                         "eval_freq", "runs", "output", "device", "seed", "project", "entity", "group"]
    config = argutils.load_config_from_args(parser, args, overrideable_args)
    print("\n---- Test Config ----\n" + yaml.dump(config) + "----------------------")

    device = argutils.get_device(parser, config)
    argutils.set_seed(config["seed"])

    if config["eval_method"] not in ("sequential", "seq", "iid"):
        raise ValueError(f'Unrecognized evaluation method: "{config["eval_method"]}"')

    sampler_type = "iid" if config["eval_method"] == "iid" else "oml"
    sampler, input_shape = argutils.get_dataset_sampler(config, sampler_type=sampler_type)

    # Ensure the destination can be written.
    outpath = Path(config["output"]).resolve()
    if outpath.exists():
        print(f"WARNING: Will overwrite existing file: {outpath}", file=sys.stderr)
    else:
        outpath.parent.mkdir(parents=True, exist_ok=True)

    def wandb_init(job_type):
        return argutils.prepare_wandb(config, job_type=job_type, create_folder=False, allow_reinit=True)

    run_full_test = iid_test if config["eval_method"] == "iid" else seq_test
    run_full_test(config, wandb_init, sampler, input_shape, outpath, device)


if __name__ == "__main__":
    sys.exit(main())
