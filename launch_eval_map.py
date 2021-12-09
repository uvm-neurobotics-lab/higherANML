"""
A script to launch a full sweep of eval_map.py jobs.
"""

import argparse
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

import utils.argparsing as argutils


def get_output_directory(args, parser):
    if args.output:
        outpath = Path(args.output).resolve()
    elif len(args.model) == 1:
        # Assuming the model is inside a "trained_anmls/" folder, this places an "eval/" folder right next to it.
        outpath = args.model[0].parent.parent / "eval"
    else:
        parser.error("You must supply an output destination (-o/--output) when evaluating more than one model.")
        sys.exit(os.EX_USAGE)  # unreachable, but avoids a warning about outpath being potentially undefined.

    # Ensure the destination can be written.
    if outpath.is_file():
        parser.error(f"Output already exists as a file, not a directory: {outpath}")
    elif args.dry_run:
        print(f"Output directory that would be created: {outpath}")
    else:
        outpath.mkdir(parents=True, exist_ok=True)

    return outpath


def build_commands(args, outpath):
    # Build up the full product of all possible input choices.
    repeated_args = (args.model, args.dataset, args.classes, args.train_examples, args.test_examples, args.lr)
    repeated_args = [frozenset(v) for v in repeated_args]  # ensure no duplicates
    combos = product(*repeated_args)

    # Build the commands.
    commands = []
    for c in combos:
        unique_filename = "-".join([str(v) for v in c]).replace("/", "-").strip("-") + ".pkl"
        outfile = outpath / unique_filename
        cmd = [
            "eval_map.py",
            "--model", c[0],
            "--dataset", c[1],
            "--classes", c[2],
            "--train-examples", c[3],
            "--test-examples", c[4],
            "--lr", c[5],
            "--seed", args.seed,
            "--runs", args.runs,
            "--output", outfile,
        ]
        if not args.only_final_performance:
            cmd.append("--record-learning-curve")
        if args.device:
            cmd.extend(["--device", args.device])
        if args.verbose:
            cmd.append("-" + ("v" * args.verbose))
        commands.append([str(v) for v in cmd])
    return commands


def launch_jobs(commands, outdir, verbose=False, dry_run=False):
    if dry_run:
        print("Commands that would be launched:")
        for cmd in commands:
            print("    " + " ".join(cmd))
        return os.EX_OK

    for cmd in commands:
        try:
            print("Launching command: " + " ".join(cmd))
            stderr = subprocess.STDOUT if not verbose else None
            subprocess.run(["launcher", "dggpu", "-f", "-d", outdir] + cmd, text=True, check=True,
                           capture_output=(not verbose), stderr=stderr)
        except subprocess.CalledProcessError as e:
            # Print the output if we captured it, to allow for debugging.
            if not verbose:
                print(e.stdout)
            raise


def check_path(path):
    pathobj = Path(path)
    if pathobj.exists():
        return pathobj.resolve()
    else:
        raise argparse.ArgumentTypeError(f"model: {path} is not a valid path")


def main(args=None):
    parser = argutils.create_parser(__doc__)

    # Repeating Arguments
    # We want to accept most of the same arguments, but allow for multiple values of each.
    parser.add_argument("--dataset", nargs="+", choices=["omni", "miniimagenet"], type=str.lower, default=["omni"],
                        help="The dataset to use.")
    parser.add_argument("-m", "--model", metavar="PATH", nargs="+", type=check_path, required=True,
                        help="Path to the model to evaluate.")
    parser.add_argument("-c", "--classes", metavar="INT", nargs="+", type=int, required=True,
                        help="Number of classes to test.")
    parser.add_argument("--train-examples", metavar="INT", nargs="+", type=int, default=[15],
                        help="Number of examples per class, for training.")
    parser.add_argument("--test-examples", metavar="INT", nargs="+", type=int, default=[5],
                        help="Number of examples per class, for testing.")
    parser.add_argument("-l", "--lr", metavar="RATE", nargs="+", type=float, default=[0.001],
                        help="Learning rate to use (check README for suggestions).")

    # Remaining Arguments
    parser.add_argument("--only-final-performance", action="store_true",
                        help="Do not record train/test performance throughout the whole training procedure; only"
                             " record final performance. This saves a lot of time.")
    parser.add_argument("-r", "--runs", metavar="INT", type=int, default=10,
                        help="Number of repetitions to run for EACH unique combination of arguments.")
    parser.add_argument("-o", "--output", metavar="PATH",
                        help="The folder to save all results. This folder should NOT already contain any .pkl files,"
                             " because we will assume that ALL .pkl files are the result of this job.")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Launch a new job even if another job has already used this output directory. Does not"
                             " remove the previous output log, and only needed when specifying --rundir.")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="Do not actually launch jobs, but only print out the equivalent commands that would be"
                             " launched.")
    argutils.add_device_arg(parser)
    # We will require a fixed seed, so all runs are more comparable.
    argutils.add_seed_arg(parser, default_seed=12345)
    argutils.add_verbose_arg(parser)
    parser.add_argument("--launch-verbose", action="store_true",
                        help="Be verbose when launching the job (output all the launcher print statements).")

    # Parse
    args = parser.parse_args(args)

    # Get destination path.
    outpath = get_output_directory(args, parser)

    # Get all argument lists.
    commands = build_commands(args, outpath)

    # Launch the jobs.
    return launch_jobs(commands, outpath, args.launch_verbose, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
