"""
The "map" portion of a map-reduce job for evaluating ANML and related models.
"""
# NOTE: Use the following command to test the functionality of this script:
#   WANDB_MODE=disabled python eval_map.py -c configs/eval-omni-anml.yml --model trained_anmls/anml-1-28-28-29999.net --output test.pkl --classes 10 --lr 0.0015 --group mygroup
# You should get a final accuracy somewhere around:
#   Train 96.8% (std: 3.4%) | Test 92.6% (std: 6.2%)
# Other learning rates will result in lower performance.
# Optionally add `--method iid` to test standard transfer learning ("oracle" style).
# Optionally add `--eval-freq 2` to test a full trajectory of evaluation numbers. This will take longer and will print
# the same result, but more data will end up in the resulting dataframe.

import sys
from pathlib import Path

import yaml

import utils.argparsing as argutils
from anml import run_full_test as seq_test
from iid import run_full_test as iid_test
from zero_shot import run_full_test as zero_shot_test


EVAL_METHODS = {
    "sequential": seq_test,
    "seq": seq_test,
    "iid": iid_test,
    "zero_shot": zero_shot_test,
}

# NOTE: launch_eval_map.py refers to these defaults so that it uses the same defaults.
EVAL_METHOD_DFLT = "sequential"
REINIT_METHOD_DFLT = "kaiming"
TRAIN_EX_DFLT = 15
TEST_EX_DFLT = 5
EPOCHS_DFLT = 1
BATCH_SIZE_DFLT = 256
INIT_SIZE_DFLT = 256
RUNS_DFLT = 10
SEED_DFLT = 12345


def main(args=None):
    parser = argutils.create_parser(__doc__)

    parser.add_argument("-c", "--config", metavar="PATH", type=argutils.existing_path, required=True,
                        help="Evaluation config file.")
    parser.add_argument("--eval-method", choices=("sequential", "seq", "iid", "zero_shot"), default=EVAL_METHOD_DFLT,
                        help="The testing method to use: sequential (continual learning) or i.i.d. (standard transfer"
                             " learning).")
    parser.add_argument("--reinit-method", choices=("kaiming", "lstsq"), default=REINIT_METHOD_DFLT,
                        help="The method to use to reinitialize trainable parameters: typical kaiming normal"
                             "initialization or least squares estimate of the final linear layer.")
    argutils.add_dataset_arg(parser)
    parser.add_argument("-m", "--model", metavar="PATH", type=argutils.existing_path,
                        help="Path to the model to evaluate.")
    parser.add_argument("-l", "--lr", metavar="RATE", type=float,
                        help="Learning rate to use (check README for suggestions).")
    parser.add_argument("--classes", metavar="INT", type=int, help="Number of classes to test.")
    parser.add_argument("--train-examples", metavar="INT", type=int, default=TRAIN_EX_DFLT,
                        help="Number of examples per class, for training.")
    parser.add_argument("--test-examples", metavar="INT", type=int, default=TEST_EX_DFLT,
                        help="Number of examples per class, for testing.")
    parser.add_argument("--epochs", metavar="INT", type=int, default=EPOCHS_DFLT,
                        help="Number of epochs to fine-tune for. Only used in i.i.d. testing.")
    parser.add_argument("--batch-size", metavar="INT", type=int, default=BATCH_SIZE_DFLT,
                        help="Size of batches to train on. Only used in i.i.d. testing.")
    parser.add_argument("--init-size", metavar="INT", type=int, default=INIT_SIZE_DFLT,
                        help="Number of samples from the support set allowed to be used for parameter initialization.")
    parser.add_argument("--eval-freq", metavar="INT", type=int,
                        help="The frequency at which to evaluate performance of the model throughout the learning"
                             " process. This can be very expensive, if evaluating after every class learned (freq = 1)."
                             " By default we evaluate only at the very end.")
    parser.add_argument("-r", "--runs", metavar="INT", type=int, default=RUNS_DFLT, help="Number of repetitions to run.")
    parser.add_argument("-o", "--output", metavar="PATH", help="The location to save to.")
    argutils.add_device_arg(parser)
    # NOTE: Enforcing a default seed here makes it impossible to launch a truly random run. But I think this is better
    # than the alternative which could have us run different tests with different seeds accidentally, so they would not
    # be comparable.
    argutils.add_seed_arg(parser, SEED_DFLT)
    argutils.add_wandb_args(parser)
    argutils.add_verbose_arg(parser)

    args = parser.parse_args(args)
    argutils.configure_logging(args)
    overrideable_args = ["eval_method", "reinit_method", "dataset", "data_path", "download", "im_size", "augment",
                         "model", "classes", "train_examples", "test_examples", "epochs", "batch_size", "init_size",
                         "lr", "eval_freq", "runs", "output", "device", "seed", "project", "entity", "group"]
    config = argutils.load_config_from_args(parser, args, overrideable_args)
    print("\n---- Test Config ----\n" + yaml.dump(config) + "----------------------")

    device = argutils.get_device(parser, config)
    argutils.set_seed(config["seed"])

    if config["eval_method"] not in ("sequential", "seq", "iid", "zero_shot"):
        raise ValueError(f'Unrecognized evaluation method: "{config["eval_method"]}"')
    elif config["eval_method"].startswith("seq"):
        # Make the naming uniform so they all report under the same name in W&B and pandas tables.
        config["eval_method"] = "sequential"

    sampler_type = "oml" if config["eval_method"].startswith("seq") else "iid"
    sampler, input_shape = argutils.get_dataset_sampler(config, sampler_type=sampler_type)

    # Ensure the destination can be written.
    outpath = Path(config["output"]).resolve()
    if outpath.exists():
        print(f"WARNING: Will overwrite existing file: {outpath}", file=sys.stderr)
    else:
        outpath.parent.mkdir(parents=True, exist_ok=True)

    def wandb_init(job_type):
        return argutils.prepare_wandb(config, job_type=job_type, create_folder=False, allow_reinit=True)

    run_full_test = EVAL_METHODS[config["eval_method"]]
    run_full_test(config, wandb_init, sampler, input_shape, outpath, device)


if __name__ == "__main__":
    sys.exit(main())
