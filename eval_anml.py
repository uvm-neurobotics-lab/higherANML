"""
Script for evaluation of ANML using OML-style continual learning trajectories.
"""
# NOTE: Use the following command to test the functionality of this script:
#   python eval_anml.py -c configs/eval-omni-anml.yml --model trained_anmls/256_112_2304_ANML-29999.pth --classes 10 --lr 0.0015
# You should get a final accuracy somewhere around:
#   Train 96.8% (std: 3.4%) | Test 92.6% (std: 6.2%)
# Other learning rates will result in lower performance.

import warnings
import sys

import numpy as np
import yaml
from tqdm import trange

import utils.argparsing as argutils
from anml import test_train

warnings.filterwarnings("ignore")


def repeats(sampler, sampler_input_shape, config, device):
    train_results = []
    test_results = []
    for _ in trange(config["runs"]):
        train_traj, test_traj = test_train(sampler, sampler_input_shape, config, device)
        # For now, we are just reporting the final result, so just pluck off the last set of accuracies. This is a list
        # of accuracies per class, so taking the mean gives us overall accuracy.
        # NOTE: This averaging method assumes we have the same number of examples per each class.
        train_results.append(train_traj[-1].mean())
        test_results.append(test_traj[-1].mean())

    print(f"Classes: {config['classes']}"
          f" | Train Accuracy: {np.mean(train_results):.1%} (std {np.std(train_results):.1%})"
          f" | Test Accuracy: {np.mean(test_results):.1%} (std {np.std(test_results):.1%})")


def main(args=None):
    # Evaluation setting
    parser = argutils.create_parser(__doc__)

    parser.add_argument("-c", "--config", metavar="PATH", type=argutils.existing_path, required=True,
                        help="Evaluation config file.")
    argutils.add_dataset_arg(parser)
    parser.add_argument("-m", "--model", type=argutils.existing_path, help="Path to the model to evaluate.")
    parser.add_argument("-l", "--lr", metavar="RATE", type=float,
                        help="Learning rate to use (check README for suggestions).")
    parser.add_argument("--classes", type=int, help="Number of classes to test.")
    parser.add_argument("--train-examples", type=int, default=15, help="Number of examples per class, for training.")
    parser.add_argument("--test-examples", type=int, default=5, help="Number of examples per class, for testing.")
    parser.add_argument("-r", "--runs", type=int, default=10, help="Number of repetitions to run.")
    argutils.add_device_arg(parser)
    argutils.add_seed_arg(parser)
    argutils.add_verbose_arg(parser)

    args = parser.parse_args(args)
    argutils.configure_logging(args)
    overrideable_args = ["dataset", "data_path", "download", "im_size", "model", "classes", "train_examples",
                         "test_examples", "lr", "record_learning_curve", "runs", "device", "seed", "project", "entity",
                         "group"]
    config = argutils.load_config_from_args(parser, args, overrideable_args)
    if args.verbose:
        print("\n---- Test Config ----\n" + yaml.dump(config) + "----------------------")

    device = argutils.get_device(parser, config)
    argutils.set_seed(config["seed"])
    sampler, input_shape = argutils.get_OML_dataset_sampler(config)

    repeats(sampler, input_shape, config, device)


if __name__ == "__main__":
    sys.exit(main())
