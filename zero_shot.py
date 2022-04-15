from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import trange

from models import load_model, maybe_collect_init_sample, reinit_params
from utils import ensure_config_param, flatten
from utils.logging import overall_accuracy


def evaluate_and_log(model, name, loader, device, should_log):
    acc1, acc5 = overall_accuracy(model, loader, device)
    if should_log:
        import wandb
        wandb.log({f"test_{name}.acc": acc1, f"test_{name}.top5_acc": acc5}, step=0)
    return acc1, acc5


def check_test_config(config):
    def gt_zero(x):
        return x > 0

    def of_type(types):
        def test_fn(val):
            return isinstance(val, types)

        return test_fn

    ensure_config_param(config, "model", of_type((str, Path)))
    ensure_config_param(config, "classes", gt_zero)
    ensure_config_param(config, "batch_size", gt_zero)
    ensure_config_param(config, "train_examples", gt_zero)
    ensure_config_param(config, "test_examples", gt_zero)


def test_train(sampler, sampler_input_shape, config, device="cuda", log_to_wandb=False):
    check_test_config(config)

    model = load_model(config["model"], sampler_input_shape, device)
    model = model.to(device)
    model.eval()

    # Sample the support set we'll use to train and the query set we'll use to test.
    support_set, query_set = sampler.sample_support_and_query_sets(config["classes"], config["train_examples"],
                                                                   config["test_examples"])
    assert len(support_set) > 0
    assert len(query_set) > 0

    support_loader = DataLoader(support_set, batch_size=config["batch_size"], shuffle=True)
    query_loader = DataLoader(query_set, batch_size=config["batch_size"], shuffle=True)

    init_sample = maybe_collect_init_sample(model, config, support_set, device)
    reinit_params(model, config, init_sample)

    # Just directly evaluate once the model is fully initialized.
    train_perf = evaluate_and_log(model, "train", support_loader, device, should_log=log_to_wandb)
    test_perf = evaluate_and_log(model, "test", query_loader, device, should_log=log_to_wandb)

    return train_perf, test_perf


def test_repeats(num_runs, wandb_init, **kwargs):
    """ Run `num_runs` times and collect results into a single list. """
    kwargs["log_to_wandb"] = True
    results = []

    # Run the tests `num_runs` times.
    for r in trange(num_runs):
        with wandb_init("eval"):
            train_perf, test_perf = test_train(**kwargs)
            results.append((r, train_perf, test_perf))

    return results


def save_results(results, output_path, config):
    """ Transform results from `test_repeats()` into a pandas Dataframe and save to the given file. """
    # These params describing the evaluation should be prepended to each row in the table.
    eval_param_names = ["model", "dataset", "train_examples", "test_examples", "eval_method", "reinit_method",
                        "reinit_params", "classes"]
    eval_params = [config.get(k) for k in eval_param_names]

    # Flatten the data in each row into matrix form, while adding the above metadata.
    full_data = []
    for idx, train_acc, test_acc in results:
        full_data.append(flatten([eval_params, [idx], train_acc, test_acc]))

    # Now assemble the data into a dataframe and save it.
    colnames = eval_param_names + ["trial", "train_acc", "train_top5_acc", "test_acc", "test_top5_acc"]
    result_matrix = pd.DataFrame(full_data, columns=colnames)
    # Although it makes some operations more cumbersome, we can save a lot of space and maybe some time by treating
    # most of the columns as a MultiIndex. Also makes indexing easier. All but the final metrics columns.
    result_matrix.set_index(colnames[:-4], inplace=True)
    result_matrix.to_pickle(output_path)
    print(f"Saved result matrix of size {result_matrix.shape} to: {output_path}")
    return result_matrix


def report_summary(result_matrix):
    """ Take dataframe resulting from `save_results()` and compute and print some summary metrics. """
    # Get just the final test accuracy of each run (when step is max).
    train_acc = result_matrix.loc[:, "train_acc"]
    test_acc = result_matrix.loc[:, "test_acc"]
    # NOTE: Right now we're just printing to console, but it may be useful in the future to report this back to the
    # original training job as a summary metric? Example here: https://docs.wandb.ai/guides/track/log#summary-metrics
    num_classes = result_matrix.index.get_level_values("classes").max()
    print(f"Final accuracy on {num_classes} classes:")
    print(f"Train {train_acc.mean():.1%} (std: {train_acc.std():.1%}) | "
          f"Test {test_acc.mean():.1%} (std: {test_acc.std():.1%})")


def run_full_test(config, wandb_init, sampler, input_shape, outpath, device):
    """ Run `test_train()` a number of times and collect the results. """
    # The name of these keyword arguments needs to match the ones in `test_train()`, as we will pass them on.
    results = test_repeats(
        num_runs=config["runs"],
        wandb_init=wandb_init,
        sampler=sampler,
        sampler_input_shape=input_shape,
        config=config,
        device=device,
    )

    # Assemble and save the result matrix.
    result_matrix = save_results(results, outpath, config)

    # Print summary to console.
    report_summary(result_matrix)
