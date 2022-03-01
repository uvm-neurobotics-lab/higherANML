"""
The "map" portion of a map-reduce job for evaluating ANML and related models.
"""

import argparse
import sys
from itertools import count
from pathlib import Path

import pandas as pd
import yaml
from tqdm import trange

import utils.argparsing as argutils
from anml import test_train


def check_path(path):
    if Path(path).exists():
        return path
    else:
        raise argparse.ArgumentTypeError(f"model: {path} is not a valid path")


def repeats(num_runs, wandb_init, **kwargs):
    num_classes = kwargs["classes"]
    kwargs["log_to_wandb"] = True
    nanfill = [float("nan")]
    results = []

    # Run the tests `num_runs` times.
    for r in trange(num_runs):
        with wandb_init("eval"):
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


def save_results(results, output_path, num_classes, config):
    # These params describing the evaluation should be prepended to each row in the table.
    eval_param_names = ["model", "dataset", "train_examples", "test_examples", "reinit_params", "opt_params", "classes",
                        "lr"]
    eval_params = [config[k] for k in eval_param_names]

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
    return result_matrix


def report_summary(result_matrix):
    # Average over all classes to get overall performance numbers, by grouping by columns other than class.
    non_class_columns = list(filter(lambda x: x != "class_id", result_matrix.index.names))
    avg_over_classes = result_matrix.groupby(non_class_columns).mean()
    # Get just the final test accuracy of each run (when we've trained on all classes).
    classes_trained = avg_over_classes.index.get_level_values("classes_trained")
    num_classes = classes_trained.max()
    train_acc = avg_over_classes.loc[classes_trained == num_classes, "train_acc"]
    test_acc = avg_over_classes.loc[classes_trained == num_classes, "test_acc"]
    # NOTE: Right now we're just printing to console, but it may be useful in the future to report this back to the
    # original training job as a summary metric? Example here: https://docs.wandb.ai/guides/track/log#summary-metrics
    print(f"Final accuracy on {num_classes} classes:")
    print(f"Train {train_acc.mean():.1%} (std: {train_acc.std():.1%}) | "
          f"Test {test_acc.mean():.1%} (std: {test_acc.std():.1%})")


def main(args=None):
    parser = argutils.create_parser(__doc__)

    parser.add_argument("-c", "--config", metavar="PATH", type=argutils.existing_path, required=True,
                        help="Evaluation config file.")
    argutils.add_dataset_arg(parser)
    parser.add_argument("-m", "--model", metavar="PATH", type=check_path, help="Path to the model to evaluate.")
    parser.add_argument("-l", "--lr", metavar="RATE", type=float,
                        help="Learning rate to use (check README for suggestions).")
    parser.add_argument("--classes", metavar="INT", type=int, help="Number of classes to test.")
    parser.add_argument("--train-examples", metavar="INT", type=int, default=15,
                        help="Number of examples per class, for training.")
    parser.add_argument("--test-examples", metavar="INT", type=int, default=5,
                        help="Number of examples per class, for testing.")
    parser.add_argument("--record-learning-curve", action="store_true",
                        help="Whether to record train/test performance throughout the whole training procedure, as"
                             " opposed to just recording final performance. This is very expensive.")
    parser.add_argument("-r", "--runs", metavar="INT", type=int, default=10, help="Number of repetitions to run.")
    parser.add_argument("-o", "--output", metavar="PATH", help="The location to save to.")
    argutils.add_device_arg(parser)
    argutils.add_seed_arg(parser)
    argutils.add_wandb_args(parser)
    argutils.add_verbose_arg(parser)

    args = parser.parse_args(args)
    argutils.configure_logging(args)
    overrideable_args = ["dataset", "data_path", "download", "im_size", "model", "classes", "train_examples",
                         "test_examples", "lr", "record_learning_curve", "runs", "output", "device", "seed", "project",
                         "entity", "group"]
    config = argutils.load_config_from_args(parser, args, overrideable_args)
    print("\n---- Test Config ----\n" + yaml.dump(config) + "----------------------")

    device = argutils.get_device(parser, config)
    argutils.set_seed(config["seed"])
    sampler, input_shape = argutils.get_OML_dataset_sampler(config)

    # Ensure the destination can be written.
    outpath = Path(config["output"]).resolve()
    if outpath.exists():
        print(f"WARNING: Will overwrite existing file: {outpath}", file=sys.stderr)
    else:
        outpath.parent.mkdir(parents=True, exist_ok=True)

    def wandb_init(job_type):
        return argutils.prepare_wandb(config, job_type=job_type, create_folder=False, allow_reinit=True)

    # The name of these keyword arguments needs to match the ones in `test_train()`, as we will pass them on.
    results = repeats(
        num_runs=config["runs"],
        wandb_init=wandb_init,
        sampler=sampler,
        sampler_input_shape=input_shape,
        config=config,
        device=device,
    )

    # Assemble and save the result matrix.
    result_matrix = save_results(results, outpath, args.classes, config)

    # Print summary to console.
    report_summary(result_matrix)


if __name__ == "__main__":
    sys.exit(main())
