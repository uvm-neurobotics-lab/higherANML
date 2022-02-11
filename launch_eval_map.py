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
import shutil
import sys
import uuid
from itertools import product
from pathlib import Path

import utils.argparsing as argutils
from utils import as_strings
from utils.slurm import call_sbatch


# Get the resolved path of this script, before we switch directories.
SCRIPT_DIR = Path(__file__).parent.resolve()


def place_eval_notebook(outpath, args):
    # Find the notebook relative to this script.
    nbfile = SCRIPT_DIR / "notebooks" / "anml-meta-test-eval.ipynb"
    assert nbfile.exists(), f"Script file ({nbfile}) not found."
    assert nbfile.is_file(), f"Script file ({nbfile}) is not a file."
    dest = outpath / nbfile.name
    if dest.exists() and not args.force:
        raise RuntimeError(f"Destination notebook already exists: {dest}. Use -f/--force to overwrite.")
    elif not args.dry_run:
        shutil.copy(nbfile, outpath)
    else:
        print(f"Would place eval notebook into the output folder: {nbfile.name}")


def get_input_output_dirs(args, parser):
    # Output Directory
    if args.output:
        outpath = Path(args.output).resolve()
    elif len(args.model) == 1:
        # Assuming the model is inside a "trained_anmls/" folder, this places an "eval/" folder right next to
        # "trained_anmls/". Assuming the model name is "NAME-<some-details>-EPOCH.net", this names the folder as
        # "eval-NAME-EPOCH/".
        model_file = args.model[0]
        model_spec = model_file.stem.split("-")
        outpath = model_file.parent.parent / f"eval-{model_spec[0]}-{model_spec[-1]}"
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

    # Copy eval notebook into the destination.
    place_eval_notebook(outpath, args)

    # Input Directory
    if args.data_path:
        inpath = args.data_path
    else:
        # By default, expect data to be in this repository, at `experiments/data/`.
        inpath = SCRIPT_DIR / "experiments" / "data"

    return inpath, outpath


def build_command_args(args, outpath):
    # Build up the full product of all possible input choices.
    repeated_args = (args.model, args.dataset, args.classes, args.train_examples, args.test_examples, args.lr)
    repeated_args = [frozenset(v) for v in repeated_args]  # ensure no duplicates
    combos = product(*repeated_args)

    # Convert each unique combo into a set of command line args.
    arglines = []
    existing_output_files = []
    for c in combos:
        # Determine output path based on arg settings.
        unique_filename = "-".join(as_strings(c)).replace("/", "-").strip("-") + ".pkl"
        outfile = outpath / unique_filename
        if outfile.exists():
            existing_output_files.append(outfile)
        arglines.append([
            "--model", c[0],
            "--dataset", c[1],
            "--classes", c[2],
            "--train-examples", c[3],
            "--test-examples", c[4],
            "--lr", c[5],
            "--output", outfile,
        ])

    # Handle already-existing files.
    if existing_output_files and not args.force:
        msg = "Refusing to launch because the following files already exist:\n"
        for f in existing_output_files:
            msg += str(f) + "\n"
        msg += "Use -f/--force to proceed anyway and overwrite the files."
        raise RuntimeError(msg)

    # Convert each line to a string before returning.
    return [" ".join(as_strings(a)) for a in arglines]


def build_commands(args, inpath, outpath, launcher_args):
    # Get one line of arguments for each unique command.
    arglines = build_command_args(args, outpath)

    # Write the argfile. It should be stored permanently in the destination directory so the running job can refer to
    # it. Use a UUID to avoid name collisions with jobs outputting to the same folder.
    argfile_path = outpath / ("args-" + uuid.uuid4().hex + ".txt")
    if not args.dry_run:
        with argfile_path.open("w") as argfile:
            for line in arglines:
                argfile.write(line + "\n")
    else:
        print(f"Argfile that would be created: {argfile_path}")
        for line in arglines:
            print("    " + line)

    # Build the command.
    cmd = [
        "eval_map.py",
        "--data-path", inpath,
        "--no-download",
        "--seed", args.seed,
        "--runs", args.runs,
    ]
    if not args.only_final_performance:
        cmd.append("--record-learning-curve")
    if args.im_size:
        cmd.extend(["--im-size", args.im_size])
    if args.device:
        cmd.extend(["--device", args.device])
    if args.verbose:
        cmd.append("-" + ("v" * args.verbose))
    # Add launcher wrapper.
    cmd = ["launcher", "dggpu", "-f", "-d", outpath, "--argfile", argfile_path] + launcher_args + cmd
    cmd = as_strings(cmd)

    return cmd


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
    non_repeat_group.add_argument("--im-size", metavar="PX", type=int, default=None,
                                  help="Resize all input images to the given size (in pixels).")
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

    # Get command and corresponding list of arguments.
    command = build_commands(args, inpath, outpath, launcher_args)

    # Launch the jobs.
    return call_sbatch(command, args.launch_verbose, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
