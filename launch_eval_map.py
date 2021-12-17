"""
A script to launch a full sweep of eval_map.py jobs.

Launches the full combination of all options provided. You can provide multiple
possible learning rates, datasets, models, etc.

Furthermore, any settings accepted by `sbatch` OR `launcher` may be supplied on
the command line. These will override the options provided by the launch config,
as well as the ones this script would normally pass to `launcher`. For example,
to allocate more memory than the default, and to run in a location other than
the one specified by `--output` (meaning the slurm-*.out log will be placed
here):
    > python launch_eval_map.py [...] --mem=64G --rundir <my-output-dir>

IMPORTANT: The sbatch options must be supplied in "--name=value" fashion, with an
equals sign; "--name value" will NOT parse correctly. For any other options
(including script flags) you may use either format.

Note: To debug the Slurm-launching behavior of this script, you may run as:
    > DEBUG=1 python launch_eval_map.py [...] --launch-verbose
The above will not launch any jobs. To launch jobs, but still see the full output,
drop the `DEBUG=1` flag.
"""

import argparse
import os
import re
import subprocess
import sys
from itertools import product
from pathlib import Path

import utils.argparsing as argutils


def get_input_output_dirs(args, parser):
    # Output Directory
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

    # Input Directory
    if args.data_path:
        inpath = args.data_path
    else:
        # By default, expect data two levels above the output folder.
        inpath = outpath.parent.parent / "data"

    return inpath, outpath


def build_commands(args, inpath, outpath, launcher_args):
    # Build up the full product of all possible input choices.
    repeated_args = (args.model, args.dataset, args.classes, args.train_examples, args.test_examples, args.lr)
    repeated_args = [frozenset(v) for v in repeated_args]  # ensure no duplicates
    combos = product(*repeated_args)

    # Build the commands.
    commands = []
    existing_output_files = []
    for c in combos:
        # Determine output path based on arg settings.
        unique_filename = "-".join([str(v) for v in c]).replace("/", "-").strip("-") + ".pkl"
        outfile = outpath / unique_filename
        if outfile.exists():
            existing_output_files.append(outfile)
        # Build up inner command args.
        cmd = [
            "eval_map.py",
            "--model", c[0],
            "--dataset", c[1],
            "--data-path", inpath,
            "--no-download",
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
        # Add launcher wrapper.
        cmd = ["launcher", "dggpu", "-f", "-d", outpath] + launcher_args + cmd
        commands.append([str(v) for v in cmd])

    if existing_output_files and not args.force:
        msg = "Refusing to launch because the following files already exist:\n"
        for f in existing_output_files:
            msg += str(f) + "\n"
        msg += "Use -f/--force to proceed anyway and overwrite the files."
        raise RuntimeError(msg)

    return commands


def launch_jobs(commands, verbose=False, dry_run=False):
    if dry_run:
        print("Commands that would be run:")
        for cmd in commands:
            print("    " + " ".join(cmd))
        return os.EX_OK

    for cmd in commands:
        try:
            print("Running command: " + " ".join(cmd))
            if verbose:
                # If verbose, just let the launcher output directly to console.
                stderr = None
                stdout = None
            else:
                # Normally, redirect stderr -> stdout and capture them both into stdout.
                stderr = subprocess.STDOUT
                stdout = subprocess.PIPE
            res = subprocess.run(cmd, text=True, check=True, stdout=stdout, stderr=stderr)
            # Find the Slurm job ID in the output and print it, if we captured the output.
            if not verbose:
                match = re.search("Submitted batch job (\d+)", res.stdout)
                if not match:
                    print("    WARNING: Could not find Slurm job ID in launcher output. This should not happen.")
                else:
                    print("    " + match.group(0))
        except subprocess.CalledProcessError as e:
            # Print the output if we captured it, to allow for debugging.
            if not verbose:
                print("LAUNCH FAILED. Launcher output:")
                print("-" * 80)
                print(e.stdout)
                print("-" * 80)
            raise


def check_path(path):
    pathobj = Path(path)
    if pathobj.exists():
        return pathobj.resolve()
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")


def main(args=None):
    # Disable abbreviations to avoid some of the "unknown" args from potentially being swallowed.
    # See the warning about prefix matching here: https://docs.python.org/3/library/argparse.html#partial-parsing
    parser = argutils.create_parser(__doc__, allow_abbrev=False)

    # Repeating Arguments
    # We want to accept most of the same arguments, but allow for multiple values of each.
    repeat_group = parser.add_argument_group("Repeating Evaluation Arguments",
                                             "You can supply multiple values for each of these arguments, and all "
                                             "possible combinations of the arguments will be launched.")
    repeat_group.add_argument("--dataset", nargs="+", choices=["omni", "miniimagenet"], type=str.lower,
                              default=["omni"], help="The dataset to use.")
    repeat_group.add_argument("-m", "--model", metavar="PATH", nargs="+", type=check_path, required=True,
                              help="Path to the model to evaluate.")
    repeat_group.add_argument("-c", "--classes", metavar="INT", nargs="+", type=int, required=True,
                              help="Number of classes to test.")
    repeat_group.add_argument("--train-examples", metavar="INT", nargs="+", type=int, default=[15],
                              help="Number of examples per class, for training.")
    repeat_group.add_argument("--test-examples", metavar="INT", nargs="+", type=int, default=[5],
                              help="Number of examples per class, for testing.")
    repeat_group.add_argument("-l", "--lr", metavar="RATE", nargs="+", type=float, default=[0.001],
                              help="Learning rate to use (check README for suggestions).")

    # Non-Repeating Arguments
    non_repeat_group = parser.add_argument_group("Non-Repeating Evaluation Arguments",
                                                 "Arguments that will be the same across all eval_map.py jobs.")
    non_repeat_group.add_argument("--data-path", "--data-dir", metavar="PATH", type=check_path,
                                  help="The root path in which to look for the dataset(s). Default location will be"
                                       " relative to the output directory: <output>/../../data. IMPORTANT: The datasets"
                                       " will not be downloaded automatically, so make sure they exist before"
                                       " launching.")
    non_repeat_group.add_argument("--only-final-performance", action="store_true",
                                  help="Do not record train/test performance throughout the whole training procedure;"
                                       " only record final performance. This saves a lot of time and space, but"
                                       " obviously also limits the analysis that can be done.")
    non_repeat_group.add_argument("-r", "--runs", metavar="INT", type=int, default=10,
                                  help="Number of repetitions to run for each unique combination of arguments.")
    # We will require a fixed seed, so all runs are more comparable.
    non_repeat_group.add_argument("--seed", type=int, default=12345,
                                  help='Random seed. The same seed will be used for all jobs, but each "run" within'
                                       ' each "job" will have a different random sampling of data.')
    argutils.add_device_arg(non_repeat_group)
    argutils.add_verbose_arg(non_repeat_group)

    # Other/Launcher Arguments
    parser.add_argument("-o", "--output", metavar="PATH",
                        help="The folder to save all results. This folder should NOT already contain any .pkl files,"
                             " because we will assume that ALL .pkl files are the result of this job.")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Launch a new job even if one of the intended outputs already exists and will be"
                             " overwritten.")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="Do not actually launch jobs, but only print out the equivalent commands that would be"
                             " launched.")
    parser.add_argument("--lv", "--launch-verbose", dest="launch_verbose", action="store_true",
                        help="Be verbose when launching the job (output all the launcher print statements).")

    # Parse
    args, launcher_args = parser.parse_known_args(args)

    # Get destination path.
    inpath, outpath = get_input_output_dirs(args, parser)

    # Get all argument lists.
    commands = build_commands(args, inpath, outpath, launcher_args)

    # Launch the jobs.
    return launch_jobs(commands, args.launch_verbose, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())