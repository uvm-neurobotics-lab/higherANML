"""
A script to launch a full sweep of eval_map.py jobs.

Launches the full combination of all options provided. You can provide multiple
possible learning rates, datasets, models, etc. Each parameter can be supplied
either through the yaml config OR via the command line. Command line arguments
override the config values.

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
    > DEBUG=y python launch_eval_map.py [...] --launch-verbose
The above will not launch any jobs. To launch jobs, but still see the full output,
drop the `DEBUG=y` flag.
"""
# NOTE: Use the following command to test the functionality of this script:
#   WANDB_MODE=disabled DEBUG=y python launch_eval_map.py -c configs/eval-sweep-inet-oml.yml --model trained_anmls/ANML-1-28-28-29999.net --mem=21G --group mygroup -f
# You can also use the -n/--dry-run argument. Drop the `DEBUG` flag to actually test launching of cluster jobs.

import argparse
import re
import shutil
import sys
import uuid
from itertools import product
from pathlib import Path

import yaml

import utils.argparsing as argutils
from utils import as_strings, ensure_config_param
from utils.slurm import call_sbatch


# Get the resolved path of this script, before we switch directories.
SCRIPT_DIR = Path(__file__).parent.resolve()


def prep_config(parser, args):
    overrideable_args = ["dataset", "data_path", "im_size", "model", "classes", "train_examples", "test_examples", "lr",
                         "eval_freq", "runs", "device", "seed", "project", "entity", "group"]
    config = argutils.load_config_from_args(parser, args, overrideable_args)
    ensure_config_param(config, "dataset")
    ensure_config_param(config, "model")
    ensure_config_param(config, "classes")
    ensure_config_param(config, "train_examples")
    ensure_config_param(config, "test_examples")
    ensure_config_param(config, "lr")
    ensure_config_param(config, "runs")
    ensure_config_param(config, "reinit_params")
    ensure_config_param(config, "opt_params")
    eval_freq = config.get("eval_freq")
    if eval_freq is None:
        config["eval_freq"] = max(1, config["classes"] // 20)
    return config


def get_input_output_dirs(config, output, flavor, dry_run):
    # Output Directory
    if output:
        outpath = Path(output).resolve()
    else:
        single_model = None
        if not isinstance(config["model"], (list, tuple)):
            single_model = config["model"]
        elif len(config["model"]) == 1:
            single_model = config["model"][0]

        if single_model:
            # Assuming the model is inside a "trained_anmls/" folder, this places an "eval/" folder right next to
            # "trained_anmls/". Assuming the model name is "NAME-<some-details>-EPOCH.net", this names the folder as
            # "eval-NAME-EPOCH/".
            model_file = Path(single_model).resolve()
            model_spec = model_file.stem.split("-")
            suffix = ("-" + str(flavor)) if flavor else ""
            outpath = model_file.parent.parent / (f"eval-{model_spec[0]}-{model_spec[-1]}" + suffix)
        else:
            raise RuntimeError("You must supply an output destination (-o/--output) when evaluating more than one model.")

    # Ensure the destination can be written.
    if outpath.is_file():
        raise RuntimeError(f"Output already exists as a file, not a directory: {outpath}")
    elif dry_run:
        print(f"Output directory that would be created: {outpath}")
    else:
        outpath.mkdir(parents=True, exist_ok=True)

    # Input Directory
    if config.get("data_path"):
        inpath = config["data_path"]
    else:
        # By default, expect data to be in this repository, at `experiments/data/`.
        inpath = SCRIPT_DIR / "experiments" / "data"

    return inpath, outpath


def build_command_args(config, outpath, force):
    # Figure out which config arguments have multiple choices. Extract them from the config.
    fixed_values = []
    variable_args = []
    variable_values = []
    for k in ("model", "dataset", "classes", "train_examples", "test_examples", "lr"):
        v = config[k]
        if isinstance(v, (list, tuple)):
            # Ensure no duplicates.
            v = set(v)
            if len(v) == 0:
                # Just remove the empty list from the config, it shouldn't be there.
                del config[k]
            elif len(v) == 1:
                # Just unwrap the single item and place in the config as-is.
                v = v.pop()
                fixed_values.append(v)
                config[k] = v
            else:
                # Remove the item from the config, and use it in the command lines instead.
                variable_args.append(k)
                variable_values.append(v)
                del config[k]
        else:
            fixed_values.append(v)

    # We want to use all args to name the output files. So prepend the fixed arguments onto the variable ones.
    variable_values = [{v} for v in fixed_values] + variable_values

    # Build up the full product of all possible input choices. Even if all values are fixed, this will result in at
    # least one unique combination of arguments.
    combos = product(*variable_values)

    # Convert each unique combo into a set of command line args.
    arglines = []
    existing_output_files = []
    for c in combos:
        # Determine output path based on arg settings.
        unique_filename = "-".join(as_strings(c)).replace("/", "-").strip("-") + ".pkl"
        outfile = outpath / unique_filename
        if outfile.exists():
            existing_output_files.append(outfile)
        # The first `len(fixed_values)` args are the fixed args, which are already specified by the config, so don't
        # put them on the command line.
        line = ["--output", outfile]
        for k, v in zip(variable_args, c[len(fixed_values):]):
            line.append("--" + k.replace("_", "-"))
            line.append(v)
        arglines.append(as_strings(line))

    # Handle already-existing files.
    if existing_output_files and not force:
        msg = "Refusing to launch because the following files already exist:\n"
        for f in existing_output_files:
            msg += str(f) + "\n"
        msg += "Use -f/--force to proceed anyway and overwrite the files."
        raise RuntimeError(msg)

    return arglines


def write_config(config, outpath, fid, dry_run):
    config_path = outpath / ("eval-config-" + fid + ".yml")
    if not dry_run:
        with open(config_path, "w") as f:
            yaml.dump(config, f)
    else:
        print(f"Config file that would be created: {config_path}")
        print("-" * 36 + " Config " + "-" * 36)
        print(yaml.dump(config))
        print("-" * 80)
    return config_path


def write_argfile(arglines, outpath, fid, dry_run):
    arglines = [" ".join(a) for a in arglines]
    argfile_path = outpath / ("args-" + fid + ".txt")
    if not dry_run:
        with argfile_path.open("w") as argfile:
            for line in arglines:
                argfile.write(line + "\n")
    else:
        print(f"Argfile that would be created: {argfile_path}")
        print("-" * 36 + " Argfile " + "-" * 35)
        for line in arglines:
            print("    " + line)
        print("-" * 80)
    return argfile_path


def place_eval_notebook(outpath, force, dry_run):
    # Find the notebook relative to this script.
    nbfile = SCRIPT_DIR / "notebooks" / "anml-meta-test-eval.ipynb"
    assert nbfile.exists(), f"Script file ({nbfile}) not found."
    assert nbfile.is_file(), f"Script file ({nbfile}) is not a file."
    dest = outpath / nbfile.name
    if dest.exists() and not force:
        raise RuntimeError(f"Destination notebook already exists: {dest}. Use -f/--force to overwrite.")
    elif not dry_run:
        shutil.copy(nbfile, outpath)
    else:
        print(f"Would place eval notebook into the output folder: {nbfile.name}")


def build_commands(config, inpath, outpath, cluster, verbose, force, dry_run, launcher_args):
    # Get one line of arguments for each unique command.
    # NOTE: This needs to be done before the config is written to disk.
    arglines = build_command_args(config, outpath, force)

    # For files we write to the destination folder, use a UUID to avoid name collisions with other jobs outputting to
    # the same folder.
    fid = uuid.uuid4().hex

    # Write the config into the destination.
    config_path = write_config(config, outpath, fid, dry_run)

    # Copy eval notebook into the destination.
    place_eval_notebook(outpath, force, dry_run)

    # Build the command.
    job_cmd = [
        SCRIPT_DIR / "eval_map.py",
        "--config", config_path,
        "--data-path", inpath,
        "--no-download",
    ]
    if verbose:
        job_cmd.append("-" + ("v" * verbose))

    # Add launcher wrapper.
    cmd = ["launcher", cluster, "-f", "-d", outpath]
    if len(arglines) == 1:
        # There's only one arg combination, so we can just append it directly to the job command.
        job_cmd += arglines[0]
    else:
        # There are multiple jobs to be launched, so they need to be written into an argfile.
        argfile_path = write_argfile(arglines, outpath, fid, dry_run)
        cmd += ["--argfile", argfile_path]
    if launcher_args:
        cmd += launcher_args
    cmd += job_cmd
    cmd = as_strings(cmd)

    return cmd


def launch(config, output=None, flavor=None, cluster="dggpu", verbose=0, force=False, dry_run=False,
           launch_verbose=False, launcher_args=None):

    # Get destination path.
    inpath, outpath = get_input_output_dirs(config, output, flavor, dry_run)

    # Get command and corresponding list of arguments.
    command = build_commands(config, inpath, outpath, cluster, verbose, force, dry_run, launcher_args)

    # Launch the jobs.
    return call_sbatch(command, launch_verbose, dry_run)


def check_path(path):
    pathobj = Path(path)
    if pathobj.exists():
        return pathobj.resolve()
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")


def no_whitespace(string):
    if re.search(r"\s", string):
        raise argparse.ArgumentTypeError(f'cannot have whitespace: "{string}"')
    return string


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
    repeat_group.add_argument("-m", "--model", metavar="PATH", nargs="+", type=check_path,
                              help="Path to the model to evaluate.")
    repeat_group.add_argument("--classes", metavar="INT", nargs="+", type=int, help="Number of classes to test.")
    repeat_group.add_argument("--train-examples", metavar="INT", nargs="+", type=int, default=[15],
                              help="Number of examples per class, for training.")
    repeat_group.add_argument("--test-examples", metavar="INT", nargs="+", type=int, default=[5],
                              help="Number of examples per class, for testing.")
    repeat_group.add_argument("-l", "--lr", metavar="RATE", nargs="+", type=float, default=[0.001],
                              help="Learning rate to use (check README for suggestions).")

    # Non-Repeating Arguments
    non_repeat_group = parser.add_argument_group("Non-Repeating Evaluation Arguments",
                                                 "Arguments that will be the same across all eval_map.py jobs.")
    non_repeat_group.add_argument("-c", "--config", metavar="PATH", type=argutils.existing_path, required=True,
                                  help="Evaluation config file.")
    non_repeat_group.add_argument("--data-path", "--data-dir", metavar="PATH", type=check_path,
                                  help="The root path in which to look for the dataset(s). Default location will be"
                                       " relative to the output directory: <output>/../../data. IMPORTANT: The datasets"
                                       " will not be downloaded automatically, so make sure they exist before"
                                       " launching.")
    non_repeat_group.add_argument("--im-size", metavar="PX", type=int, default=None,
                                  help="Resize all input images to the given size (in pixels).")
    non_repeat_group.add_argument("--eval-freq", metavar="INT", type=int, default=1,
                                  help="The frequency at which to evaluate performance of the model throughout the"
                                       " learning process. This can be very expensive, if evaluating after every class"
                                       " learned (freq = 1). To evaluate only at the end, supply 0. By default, we will"
                                       " evaluate at a rate which is 1/20th of the number of classes, so as to have 20"
                                       " or 21 data points in the end.")
    non_repeat_group.add_argument("-r", "--runs", metavar="INT", type=int, default=10,
                                  help="Number of repetitions to run for each unique combination of arguments.")
    # We will require a fixed seed, so all runs are more comparable.
    non_repeat_group.add_argument("--seed", type=int, default=12345,
                                  help='Random seed. The same seed will be used for all jobs, but each "run" within'
                                       ' each "job" will have a different random sampling of data.')
    argutils.add_device_arg(non_repeat_group)
    argutils.add_wandb_args(non_repeat_group)
    argutils.add_verbose_arg(non_repeat_group)

    # Other/Launcher Arguments
    parser.add_argument("-o", "--output", metavar="PATH",
                        help="The folder to save all results. This folder should NOT already contain any .pkl files,"
                             " because we will assume that ALL .pkl files are the result of this job.")
    parser.add_argument("--flavor", type=no_whitespace,
                        help="A string that describes the type of evaluation being performed. This will only be used"
                        " if --output is not given, to help name the output folder. It will be appended as a suffix to"
                        " the folder name.")
    parser.add_argument("--cluster", metavar="NAME", default="dggpu",
                        help="The cluster to launch on. This must correspond to one of the resources in your"
                             " Neuromanager config.")
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

    # Create the full config using all the command line arguments.
    config = prep_config(parser, args)

    # Run the bulk of the program.
    return launch(config, args.output, args.flavor, args.cluster, args.verbose, args.force, args.dry_run,
                  args.launch_verbose, launcher_args)


if __name__ == "__main__":
    sys.exit(main())
