"""
A script to launch all pre-train jobs, which will in turn launch all downstream transfer learning jobs.

Variables to vary:
 - train_method (encoded by base config)
 - dataset (also in base config)
 - LR/optimizer params
 - lobotomize
 - model
 - seed
"""
import os
import sys
import utils
from copy import copy
from pathlib import Path

import wandb

import launch_train
import utils.argparsing as argutils


datasets = ["omni", "oimg", "oimg100", "inet"]
train_method = ["-iid"]  # ["", "-seqep", "-iid"]  # blank means "meta"
base_config_files = [f"train-{d}{m}-sanml.yml" for d in datasets for m in train_method]

# LR variables are different depending on train_method. But we can sweep over various rates, regardless of model.
adam_LRs = [{"lr": lr} for lr in [0.003, 0.001, 0.0003]]
inner_outer_LRs = [
    {"inner_lr": 0.01, "outer_lr": 0.001},
    {"inner_lr": 0.001, "outer_lr": 0.001},
    {"inner_lr": 0.001, "outer_lr": 0.01},
]

# For some datasets, we want to run i.i.d. testing with a few different amounts of data. We want some settings to be
# an equivalent amount of data as the sequential learning runs, but for others we want them to be the maximum amount of
# data possible, so we can see what is the upper limit on what the model can learn.
oimg100_iid_train_test_splits = [(name, {"train_examples": t, "test_examples": e})
                                 for name, t, e in [("", 15, 85), ("-med", 30, 70), ("-lg", 85, 15)]]
inet_iid_train_test_splits = [(name, {"train_examples": t, "test_examples": e})
                              for name, t, e in [("", 30, 100), ("-lg", 500, 100)]]

# Different model types.
models = [
    {"model_name": "resnet18", "encoder": "resnet18", "encoder_args": {}},
    {"model_name": "sanml", "encoder": "convnet", "encoder_args": {"num_blocks": 4, "num_filters": 256}},
]

# Whether to lobotomize.
lobo_options = [None]  # [True, False]

# Nothing special about these numbers, just need to be different random selections.
seeds = [29384, 93242, 49289]


def launch(config, args, launcher_args):
    curdir = os.getcwd()
    ret = launch_train.launch(config, args, launcher_args, allow_reinit=True)
    # Need to undo the side effects of the launch method before we can launch another.
    if wandb.run is not None:
        wandb.run.finish()
    os.chdir(curdir)
    return ret


def launch_jobs(parser, args, launcher_args):
    exit_code = 0
    launched = 1

    # Estimate total number of jobs to be run.
    total = len(datasets) * len(models) * len(lobo_options) * len(seeds)
    num_iid = len([1 for m in train_method if m == "-iid"])
    num_inner_outer = len(train_method) - num_iid
    total *= (num_iid * len(adam_LRs) + num_inner_outer * len(inner_outer_LRs))
    print(f"Preparing to launch {total} jobs...")

    # Create a giant loop over all the different valid combinations of these settings.
    for dataset in datasets:
        eval_splits = inet_iid_train_test_splits if dataset == "inet" else oimg100_iid_train_test_splits

        for method in train_method:
            fpath = Path("configs") / f"train-{dataset}{method}-sanml.yml"
            config = utils.load_yaml(fpath)
            # Create the full config using all the command line arguments.
            overrideable_args = ["project", "entity", "cluster"]
            config = argutils.overwrite_command_line_args(config, parser, args, overrideable_args)

            # Set all models to just evaluate once at the very end.
            config["eval_steps"] = [300000]

            for model_desc in models:
                # no need to copy the config here, can just keep updating the same config instance
                config["model_name"] = model_desc["model_name"]
                config["model_args"]["encoder"] = model_desc["encoder"]
                config["model_args"]["encoder_args"] = model_desc["encoder_args"]

                for lobo in lobo_options:
                    if lobo is not None:  # Only change config if we explicitly request it.
                        # Lobo settings depend on train method.
                        if method == "-iid":
                            if lobo:
                                config["lobo_rate"] = 1
                                config["lobo_size"] = config["model_args"]["classifier_args"]["num_classes"]
                            else:
                                config["lobo_rate"] = 0
                                config["lobo_size"] = 0
                        else:
                            config["lobotomize"] = lobo

                    # LR settings depend on train method.
                    LRs = adam_LRs if method == "-iid" else inner_outer_LRs
                    for lr_cfg in LRs:
                        cfg = copy(config)  # copy config just to be safe; keep settings in each iteration separate
                        cfg.update(lr_cfg)

                        # Eval settings.
                        if dataset == "oimg100" or dataset == "inet":
                            # Need to reconfigure the evals for these datasets with higher number of images per class.
                            new_evals = []
                            for evset in cfg["eval"]:
                                assert len(evset) == 1
                                flavor, evcfg = next(iter(evset.items()))
                                if flavor == "iid-olft":
                                    # Do not add. Making an explicit choice here to drop the OLFT results.
                                    pass
                                elif flavor == "iid-unfrozen":
                                    # In this case we'll add a number of variations on this config.
                                    for name, settings in eval_splits:
                                        newflav = "iid-unfrozen" + name
                                        newconf = copy(evcfg)
                                        newconf.update(settings)
                                        new_evals.append({newflav: newconf})
                                else:
                                    new_evals.append(evset)
                            # Set the new set of eval descriptions.
                            cfg["eval"] = new_evals

                        # Seed settings.
                        for seed in seeds:
                            cfg["seed"] = seed
                            # Here we finally get to launch a pre-train job! Ensure a unique copy each time we run.
                            print(f"\n---- JOB {launched}/{total} ----")
                            res = launch(copy(cfg), args, launcher_args)
                            print(f"---------------------")
                            launched += 1
                            if res != 0:
                                exit_code = res

    return exit_code


def main(argv=None):
    # Training Script Arguments
    # Disable abbreviations to avoid some of the "unknown" args from potentially being swallowed.
    # See the warning about prefix matching here: https://docs.python.org/3/library/argparse.html#partial-parsing
    parser = argutils.create_parser(__doc__, allow_abbrev=False)
    parser.add_argument("--project", default="higherANML", help="Project to use for W&B logging.")
    parser.add_argument("--entity", help="Entity to use for W&B logging.")
    argutils.add_verbose_arg(parser)
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

    # Parse and run.
    args, launcher_args = parser.parse_known_args(argv)
    return launch_jobs(parser, args, launcher_args)


if __name__ == "__main__":
    sys.exit(main())