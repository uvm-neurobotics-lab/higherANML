"""
A script to launch train_anml_batch_job.py on Slurm.
"""

import sys
from pathlib import Path

import yaml

import utils.argparsing as argutils
from train_anml import create_arg_parser, prep_config
from utils import as_strings
from utils.slurm import call_sbatch


# Get the resolved path of this script, before we switch directories.
SCRIPT_DIR = Path(__file__).parent.resolve()


def build_command(cluster, config_path, verbosity, launcher_args):
    # Find the script to run next to this file.
    target_script = SCRIPT_DIR / "train_anml_batch_job.py"
    assert target_script.exists(), f"Script file ({target_script}) not found."
    assert target_script.is_file(), f"Script file ({target_script}) is not a file."

    # train_anml_batch_job.py gets almost all its arguments from the config.
    train_cmd = [target_script, "--config", config_path]
    if verbosity:
        train_cmd.append("-" + ("v" * verbosity))

    # Add launcher wrapper.
    launch_cmd = ["launcher", cluster] + launcher_args + train_cmd
    launch_cmd = as_strings(launch_cmd)

    return launch_cmd


def main(argv=None):
    # Training Script Arguments
    # Disable abbreviations to avoid some of the "unknown" args from potentially being swallowed.
    # See the warning about prefix matching here: https://docs.python.org/3/library/argparse.html#partial-parsing
    # Also, we do not allow restarting from a given W&B ID at this time.
    parser = create_arg_parser(__doc__, allow_abbrev=False, allow_id=False)

    # Other/Launcher Arguments
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="Do not actually launch jobs, but only print out the equivalent commands that would be"
                             " launched.")
    parser.add_argument("--lv", "--launch-verbose", dest="launch_verbose", action="store_true",
                        help="Be verbose when launching the job (output all the launcher print statements).")

    # Parse
    args, launcher_args = parser.parse_known_args(argv)

    # Create the full config using all the command line arguments.
    config = prep_config(parser, args)

    # Set up, and jump into, the destination path.
    run = argutils.prepare_wandb(config, dry_run=args.dry_run)
    # Setting the ID in the config will cause the batch job to use the same W&B run which we've already created. We do
    # this so that we can create the output folder ahead of time and store the Slurm log into the same folder.
    config["id"] = run.id
    # We we reuse the name that W&B generated for our run as the group name, if the user didn't already provide one.
    # Both train and eval jobs will be put under this group in the UI.
    config.setdefault("group", run.name)

    # Write config into the destination folder (which is now our current directory), so that the batch job has its own
    # local copy of the config and doesn't conflict with other jobs.
    config_dest = Path("./train-config.yml")
    if not args.dry_run:
        with open(config_dest, "w") as f:
            yaml.dump(config, f)
    else:
        print(f"Would write training config to file: {config_dest}")

    # Get the launch command.
    command = build_command(config["cluster"], config_dest.resolve(), args.verbose, launcher_args)

    # Launch the job.
    return call_sbatch(command, args.launch_verbose, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
