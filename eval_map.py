"""
The "map" portion of a map-reduce job for evaluating ANML and related models.
"""

import argparse
import sys
from itertools import count
from pathlib import Path

import pandas as pd
from tqdm import trange

import utils.argparsing as argutils
from anml import test_train


def check_path(path):
    if Path(path).exists():
        return path
    else:
        raise argparse.ArgumentTypeError(f"model: {path} is not a valid path")


def repeats(runs, **kwargs):
    num_classes = kwargs["num_classes"]
    nanfill = [float("nan")]
    results = []

    # Run the tests `runs` times.
    for r in trange(runs):
        # RUN THE TEST
        train_traj, test_traj = test_train(**kwargs)
        # Train and test should be evaluated the same number of times.
        assert len(train_traj) == len(test_traj)
        # At the end of training, train and test should both have the full number of classes.
        assert len(train_traj[-1]) == len(test_traj[-1]) == num_classes
        # Accumulate the results.
        for epoch, tr, te in zip(count(), train_traj, test_traj):
            # Extend each result to make sure it has the full number of classes.
            tr = list(tr) + nanfill * (num_classes - len(tr))
            te = list(te) + nanfill * (num_classes - len(te))
            # NOTE: This assumes that 1 epoch == 1 class.
            # Capped b/c we might have one more eval at the end after the last epoch, but still the same number of
            # classes were trained.
            classes_trained = min(epoch + 1, num_classes)
            index = [r, epoch, classes_trained]
            results.append((index, tr, te))

    return results


def save_results(results, output_path, num_classes, **kwargs):
    # These params describing the evaluation should be prepended to each row in the table.
    eval_param_names = sorted(list(kwargs.keys()))
    eval_params = [kwargs[k] for k in eval_param_names]

    # Unflatten the data into one row per class, per epoch.
    full_data = []
    for idx, train_acc, test_acc in results:
        # All params which apply to all results in this epoch.
        prefix = eval_params + idx
        for c in range(num_classes):
            # A pair of (train, test) results per class.
            full_data.append(prefix + [c, train_acc[c], test_acc[c]])

    # Now assemble the data into a dataframe and save it.
    colnames = eval_param_names + ["trial", "epoch", "classes_trained", "class_id", "train_acc", "test_acc"]
    result_matrix = pd.DataFrame(full_data, columns=colnames)
    # Although it makes some operations more cumbersome, we can save a lot of space and maybe some time by treating
    # most of the columns as a MultiIndex. Also makes indexing easier. All but the final metrics columns.
    result_matrix.set_index(colnames[:-2], inplace=True)
    result_matrix.to_pickle(output_path)
    print(f"Saved result matrix of size {result_matrix.shape} to: {output_path}")


def main(args=None):
    parser = argutils.create_parser(__doc__)

    argutils.add_dataset_arg(parser)
    parser.add_argument("-m", "--model", metavar="PATH", type=check_path, required=True,
                        help="Path to the model to evaluate.")
    parser.add_argument("-l", "--lr", metavar="RATE", type=float, required=True,
                        help="Learning rate to use (check README for suggestions).")
    parser.add_argument("-c", "--classes", metavar="INT", type=int, required=True, help="Number of classes to test.")
    parser.add_argument("--train-examples", metavar="INT", type=int, default=15,
                        help="Number of examples per class, for training.")
    parser.add_argument("--test-examples", metavar="INT", type=int, default=5,
                        help="Number of examples per class, for testing.")
    parser.add_argument("--record-learning-curve", action="store_true",
                        help="Whether to record train/test performance throughout the whole training procedure, as"
                             " opposed to just recording final performance. This is very expensive.")
    parser.add_argument("-r", "--runs", metavar="INT", type=int, default=10, help="Number of repetitions to run.")
    parser.add_argument("-o", "--output", metavar="PATH", required=True, help="The location to save to.")
    argutils.add_device_arg(parser)
    argutils.add_seed_arg(parser)
    argutils.add_verbose_arg(parser)

    args = parser.parse_args(args)
    argutils.configure_logging(args)
    device = argutils.get_device(parser, args)
    argutils.set_seed_from_args(args)
    sampler, input_shape = argutils.get_OML_dataset_sampler(args)

    # Ensure the destination can be written.
    outpath = Path(args.output).resolve()
    if outpath.exists():
        print(f"WARNING: Will overwrite existing file: {outpath}", file=sys.stderr)
    else:
        outpath.parent.mkdir(parents=True, exist_ok=True)

    # The name of these keyword arguments needs to match the ones in `test_train()`, as we will pass them on.
    results = repeats(
        model_path=args.model,
        sampler=sampler,
        sampler_input_shape=input_shape,
        num_classes=args.classes,
        num_train_examples=args.train_examples,
        num_test_examples=args.test_examples,
        lr=args.lr,
        evaluate_complete_trajectory=args.record_learning_curve,
        runs=args.runs,
        device=device,
    )

    # Assemble and save the result matrix.
    model_abspath = str(Path(args.model).resolve())
    save_results(
        results,
        outpath,
        args.classes,
        model=model_abspath,
        dataset=args.dataset,
        num_train_examples=args.train_examples,
        num_test_examples=args.test_examples,
        lr=args.lr,
    )


if __name__ == "__main__":
    sys.exit(main())
