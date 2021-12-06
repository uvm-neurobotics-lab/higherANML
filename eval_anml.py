"""
Script for evaluation of ANML using OML-style continual learning trajectories.
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
from tqdm import trange

import utils.argparsing as argutils
from anml import test_train

warnings.filterwarnings("ignore")


def check_path(path):
    if Path(path).exists():
        return path
    else:
        raise argparse.ArgumentTypeError(f"model: {path} is not a valid path")


def repeats(runs, sampler, sampler_input_shape, path, classes, train_examples, test_examples, lr, device):

    def run():
        train_traj, test_traj = test_train(
            path,
            sampler=sampler,
            sampler_input_shape=sampler_input_shape,
            num_classes=classes,
            num_train_examples=train_examples,
            num_test_examples=test_examples,
            lr=lr,
            device=device,
        )
        # For now, we are just reporting the final result, so just pluck off the last set of accuracies.
        return train_traj[-1], test_traj[-1]

    train_results = []
    test_results = []
    for _ in trange(runs):
        # NOTE: This averaging method assumes we have the same number of examples per each class.
        train_acc_per_class, test_acc_per_class = run()
        train_results.append(train_acc_per_class.mean())
        test_results.append(test_acc_per_class.mean())

    print(f"Classes: {classes} | Train Accuracy: {np.mean(train_results):.1%} (std {np.std(train_results):.1%})"
          f" | Test Accuracy: {np.mean(test_results):.1%} (std {np.std(test_results):.1%})")


if __name__ == "__main__":
    # Evaluation setting
    parser = argutils.create_parser("ANML testing")

    argutils.add_dataset_arg(parser)
    parser.add_argument("-m", "--model", type=check_path, help="Path to the model to evaluate.")
    parser.add_argument("-l", "--lr", type=float, help="Learning rate to use (check README for suggestions).")
    parser.add_argument("-c", "--classes", type=int, help="Number of classes to test.")
    parser.add_argument("-r", "--runs", type=int, help="Number of repetitions to run.")
    parser.add_argument("--train-examples", type=int, default=15, help="Number of examples per class, for training.")
    parser.add_argument("--test-examples", type=int, default=5, help="Number of examples per class, for testing.")
    argutils.add_device_arg(parser)
    argutils.add_seed_arg(parser)
    argutils.add_verbose_arg(parser)

    args = parser.parse_args()
    argutils.configure_logging(args, level=logging.INFO)
    device = argutils.get_device(parser, args)
    argutils.set_seed_from_args(args)
    sampler, input_shape = argutils.get_OML_dataset_sampler(parser, args)

    repeats(
        runs=args.runs,
        sampler=sampler,
        sampler_input_shape=input_shape,
        path=args.model,
        classes=args.classes,
        train_examples=args.train_examples,
        test_examples=args.test_examples,
        lr=args.lr,
        device=device,
    )
