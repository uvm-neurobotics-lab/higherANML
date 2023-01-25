"""
A script to launch training jobs on Slurm.
"""
# NOTE: Use one of the following commands to test the functionality of this script:
#   WANDB_MODE=disabled DEBUG=y python launch_train.py -c configs/train-omni-anml.yml --mem=64G
#   WANDB_MODE=disabled DEBUG=y python launch_train.py -c configs/train-omni-iid-sanml.yml
# Remove the `DEBUG=y` to actually test launching on the cluster.
# Remove the `WANDB_MODE=disabled` to actually test reporting to W&B.

import logging
import sys
from pathlib import Path

import yaml

import utils.argparsing as argutils
from utils import as_strings
from utils.slurm import call_sbatch


# Get the resolved path of this script, before we switch directories.
SCRIPT_DIR = Path(__file__).parent.resolve()

# Mapping from training method to corresponding script.
TRAINING_METHODS = {
    "meta": "train_anml.py",
    "sequential_episodic": "train_anml.py",
    "iid": "train_iid.py",
}


def build_command(config, config_path, smoke_test, verbosity, launcher_args):
    # Find the script to run next to this file.
    target_script = SCRIPT_DIR / TRAINING_METHODS[config["train_method"]]
    assert target_script.exists(), f"Script file ({target_script}) not found."
    assert target_script.is_file(), f"Script file ({target_script}) is not a file."

    # train_anml_batch_job.py gets almost all its arguments from the config.
    train_cmd = [target_script, "--config", config_path]
    if smoke_test:
        train_cmd.append("--st")
    if verbosity:
        train_cmd.append("-" + ("v" * verbosity))

    # Add launcher wrapper.
    launch_cmd = ["launcher", config["cluster"]] + launcher_args + train_cmd
    launch_cmd = as_strings(launch_cmd)

    return launch_cmd


def launch(config, args, launcher_args, allow_reinit=None):
    # For convenience of filtering, make sure model_name is set.
    if "model_name" not in config:
        config["model_name"] = config.get("model")

    # Set up, and jump into, the destination path.
    run = argutils.prepare_wandb(config, dry_run=args.dry_run, allow_reinit=allow_reinit)
    # Setting the ID in the config will cause the batch job to use the same W&B run which we've already created. We do
    # this so that we can create the output folder ahead of time and store the Slurm log into the same folder.
    config["id"] = run.id
    # We reuse the name that W&B generated for our run as the group name, if the user didn't already provide one. Both
    # train and eval jobs will be put under this group in the UI.
    if not config.get("group"):  # either group is missing, or it's None or empty string
        config["group"] = run.name

    # Write config into the destination folder (which is now our current directory), so that the batch job has its own
    # local copy of the config and doesn't conflict with other jobs.
    config_dest = Path("./train-config.yml")
    if not args.dry_run:
        config_dest = config_dest.resolve()
        with open(config_dest, "w") as f:
            yaml.dump(config, f)
    else:
        print(f"Would write training config to file: {config_dest}")
        print(f"\nconfig to be written:\n{config}\n\n")

    # Get the launch command.
    command = build_command(config, config_dest, args.smoke_test, args.verbose, launcher_args)

    # Launch the job.
    return call_sbatch(command, args.launch_verbose, args.dry_run)


def main(argv=None):
    # Training Script Arguments
    # Disable abbreviations to avoid some of the "unknown" args from potentially being swallowed.
    # See the warning about prefix matching here: https://docs.python.org/3/library/argparse.html#partial-parsing
    parser = argutils.create_parser(__doc__, allow_abbrev=False)
    parser.add_argument("-c", "--config", metavar="PATH", type=argutils.existing_path, required=True,
                        help="Training config file.")
    parser.add_argument("--train-method", choices=list(TRAINING_METHODS.keys()), default="meta",
                        help="The training method to use.")
    argutils.add_dataset_arg(parser, add_train_size_arg=True)
    argutils.add_device_arg(parser)
    argutils.add_seed_arg(parser, default_seed=12345)
    # We do not allow restarting from a given W&B ID at this time.
    argutils.add_wandb_args(parser, allow_id=False)
    argutils.add_verbose_arg(parser)
    parser.add_argument("--no-full-test", dest="full_test", action="store_false",
                        help="Do not test the full train/test sets before saving each model. These tests take a long"
                             " time so this is useful when saving models frequently or running quick tests. This"
                             " setting is implied if --smoke-test is enabled.")
    parser.add_argument("--eval-steps", metavar="INT", nargs="*", type=int,
                        help="Points in the training at which the model should be fully evaluated. At each of these"
                             " steps, the model will be saved and a full evaluation will be run (in a separate Slurm"
                             " job). The result of the evaluation will be recorded in the same W&B group. To report the"
                             " final trained model, enter any number larger than --epochs.")
    parser.add_argument("--cluster", metavar="NAME", default="dggpu",
                        help="The cluster on which to launch eval jobs. This must correspond to one of the resources in"
                             " your Neuromanager config.")
    parser.add_argument("--st", "--smoke-test", dest="smoke_test", action="store_true",
                        help="Conduct a quick, full test of the training pipeline. If enabled, then a number of"
                             " arguments will be overridden to make the training run as short as possible and print in"
                             " verbose/debug mode.")

    # Other/Launcher Arguments
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="Do not actually launch jobs, but only print out the equivalent commands that would be"
                             " launched.")
    parser.add_argument("--lv", "--launch-verbose", dest="launch_verbose", action="store_true",
                        help="Be verbose when launching the job (output all the launcher print statements).")

    # Parse
    args, launcher_args = parser.parse_known_args(argv)

    # We may not really need this, but we'll do it for completeness.
    argutils.configure_logging(args, level=logging.INFO)

    # Create the full config using all the command line arguments.
    overrideable_args = ["train_method", "dataset", "data_path", "download", "im_size", "train_size", "val_size",
                         "augment", "device", "seed", "id", "project", "entity", "group", "full_test", "eval_steps",
                         "cluster"]
    config = argutils.load_config_from_args(parser, args, overrideable_args)

    # Launch the job.
    return launch(config, args, launcher_args)


if __name__ == "__main__":
    sys.exit(main())
