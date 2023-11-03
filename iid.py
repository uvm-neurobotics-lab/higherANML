import logging
from pathlib import Path

import pandas as pd
import yaml
from numpy.random import default_rng
from torch.nn.functional import cross_entropy
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import trange

import models
import utils.optimization
from models import fine_tuning_setup, load_model, maybe_collect_init_sample
from utils import (collect_matching_named_params, ensure_config_param, flatten, get_matching_module,
                   limit_model_optimization, lobotomize, restore_grad_state)
from utils.logging import forward_pass, overall_accuracy, StandardLog


def check_train_config(config):
    def gt_zero(x):
        return x > 0

    def gte_zero(x):
        return x >= 0

    def of_type(types):
        def test_fn(val):
            return isinstance(val, types)

        return test_fn

    ensure_config_param(config, "lr", gt_zero)
    ensure_config_param(config, "epochs", gte_zero)
    ensure_config_param(config, "save_freq", gt_zero)
    ensure_config_param(config, "model", of_type(str))
    config.setdefault("full_test", True)
    lobo_rate = config.get("lobo_rate")
    lobo_size = config.get("lobo_size")
    if lobo_rate and lobo_rate > 0 and lobo_size and lobo_size > 0:
        ensure_config_param(config, "lobo_rate", of_type(int))
        ensure_config_param(config, "lobo_size", of_type(int))
        # If we are performing lobotomy, then we need to know at which layer.
        ensure_config_param(config, "output_layer", of_type(str))
        if "lobo_biased_fraction" in config:
            ensure_config_param(config, "lobo_biased_fraction", lambda x: 0.0 <= x <= 1.0)


def train(sampler, input_shape, config, device="cuda", verbose=0):
    # Output config for reference. Do it before checking config to assist debugging.
    logging.info("\n---- Train Config ----\n" + yaml.dump(config) + "----------------------")
    check_train_config(config)

    # Create model.
    model, model_args = models.make_from_config(config, input_shape, device)
    logging.info(f"Model shape:\n{model}")

    # Set up progress/checkpoint logger. Name according to the supported input size, just for convenience.
    name = config.get("model", "ANML") + "-"
    name += "-".join(map(str, input_shape))
    # If double-verbose, print every iteration. Else, print at least as often as we save.
    print_freq = 1 if verbose > 1 else min(config["save_freq"], 100)
    log = StandardLog(name, model, model_args, print_freq, config["save_freq"], config["full_test"], config)
    log.begin(model, sampler, device)

    optimizer = utils.optimization.optimizer_from_config(config, model.parameters())
    scheduler = utils.optimization.scheduler_from_config(config, optimizer)
    rng = default_rng(config["seed"])

    # Output layer so we can periodically reset output nodes (see `lobotomize()`).
    output_layer = get_matching_module(model, config["output_layer"])
    lobo_rate = config.get("lobo_rate")
    lobo_size = config.get("lobo_size")
    num_biased_steps = config.get("lobo_biased_steps")
    lobo_biased_fraction = config.get("lobo_biased_fraction")
    if "lobo_biased_params" in config:
        # Only optimize these parameters during the biased steps.
        lobo_opt_params = [k for k, _ in collect_matching_named_params(model, config["lobo_biased_params"])]
    else:
        # Optimize all params in the biased steps.
        lobo_opt_params = None
    max_grad_norm = config.get("max_grad_norm", 0)

    # BEGIN TRAINING
    step = 1
    max_steps = config.get("max_steps", float("inf"))
    for epoch in range(1, config["epochs"] + 1):  # Epoch/step counts will be 1-based.
        if lobo_rate and lobo_size and epoch != 1 and epoch % lobo_rate == 0:
            num_zapped = min(lobo_size, len(output_layer.weight))
            zapped_classes = rng.choice(len(output_layer.weight), size=num_zapped, replace=False)
            lobotomize(output_layer, zapped_classes)
            for i in range(num_biased_steps):
                images, labels = sampler.get_biased_train_sample(zapped_classes, lobo_biased_fraction)
                run_one_step(images, labels, sampler, model, optimizer, log, epoch, step, lobo_opt_params,
                             max_grad_norm, device)
                step += 1
                if step > max_steps:
                    break

        step = run_one_epoch(sampler, model, optimizer, log, epoch, step, max_steps, max_grad_norm, device)
        if step > max_steps:
            break
        scheduler.step()

    log.close(step - 1, model, sampler, device)


def run_one_epoch(sampler, model, optimizer, log, epoch, step, max_steps=float("inf"), max_grad_norm=0, device=None):
    """ Run one training epoch. """
    log.epoch(step, epoch, sampler, optimizer)
    model.train()

    for images, labels in sampler.train_loader:
        run_one_step(images, labels, sampler, model, optimizer, log, epoch, step, max_grad_norm=max_grad_norm,
                     device=device)
        step += 1
        if step > max_steps:
            break

    return step


def run_one_step(images, labels, sampler, model, optimizer, log, epoch, step, opt_params=None, max_grad_norm=0,
                 device=None):
    # Move data to GPU once loaded.
    images, labels = images.to(device), labels.to(device)

    # Only optimize the given layers during this step.
    saved_opt_state = None
    if opt_params:
        saved_opt_state = limit_model_optimization(model, opt_params)

    # Forward pass.
    out, loss, acc = forward_pass(model, images, labels)

    # Backpropagate.
    optimizer.zero_grad()
    loss.backward()
    if max_grad_norm > 0:
        clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    # Reset the opt state.
    if opt_params:
        restore_grad_state(model, saved_opt_state)

    # Record accuracy and other metrics. Do this after the backward pass in case we want to record gradients or
    # save the latest model.
    log.step(step, epoch, loss, acc, out, labels, model, sampler, device)


def evaluate_and_log(model, name, loader, epoch, step, device, should_log):
    acc1, acc5 = overall_accuracy(model, loader, device)
    if should_log:
        import wandb
        wandb.log({f"test_{name}.acc": acc1, f"test_{name}.top5_acc": acc5}, step=step)
    return (epoch, step), (acc1, acc5)


def check_test_config(config):
    def gt_zero(x):
        return x > 0

    def gte_zero(x):
        return x >= 0

    def of_type(types):
        def test_fn(val):
            return isinstance(val, types)

        return test_fn

    ensure_config_param(config, "model", of_type((str, Path)))
    ensure_config_param(config, "epochs", gt_zero)
    ensure_config_param(config, "classes", gt_zero)
    ensure_config_param(config, "batch_size", gt_zero)
    ensure_config_param(config, "train_examples", gt_zero)
    ensure_config_param(config, "test_examples", gt_zero)
    if config.get("eval_freq") is None:
        config["eval_freq"] = 0
    ensure_config_param(config, "eval_freq", gte_zero)


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

    # NOTE: We need two copies of the support loader because we will use one inside the loop to test performance
    # on the entire training set mid-epoch.
    train_loader = DataLoader(support_set, batch_size=config["batch_size"], shuffle=True)
    support_loader = DataLoader(support_set, batch_size=config["batch_size"], shuffle=True)
    query_loader = DataLoader(query_set, batch_size=config["batch_size"], shuffle=True)

    init_sample = maybe_collect_init_sample(model, config, support_set, device)
    opt = fine_tuning_setup(model, config, sampler.num_test_classes(), init_sample)

    train_perf_trajectory = []
    test_perf_trajectory = []
    eval_freq = config["eval_freq"]
    should_eval = False
    step = 1

    # meta-test-TRAIN
    for epoch in range(config["epochs"]):
        for images, labels in train_loader:

            # Move data to GPU once loaded.
            images, labels = images.to(device), labels.to(device)

            # Gradient step.
            logits = model(images)
            opt.zero_grad()
            loss = cross_entropy(logits, labels)
            loss.backward()
            opt.step()

            # Evaluation, once per X steps. Additionally, record after the first step.
            should_eval = eval_freq and (step == 1 or (step % eval_freq == 0))
            if should_eval:
                train_perf_trajectory.append(evaluate_and_log(model, "train", support_loader, epoch, step, device,
                                                              should_log=log_to_wandb))
                test_perf_trajectory.append(evaluate_and_log(model, "test", query_loader, epoch, step, device,
                                                             should_log=log_to_wandb))

            step += 1

    # meta-test-TEST
    # We only need to do this if we didn't already do it in the last iteration of the loop.
    if not should_eval:
        train_perf_trajectory.append(evaluate_and_log(model, "train", support_loader, epoch, step - 1, device,
                                                      should_log=log_to_wandb))
        test_perf_trajectory.append(evaluate_and_log(model, "test", query_loader, epoch, step - 1, device,
                                                     should_log=log_to_wandb))

    return train_perf_trajectory, test_perf_trajectory


def test_repeats(num_runs, wandb_init, **kwargs):
    """ Run `num_runs` times and collect results into a single list. """
    kwargs["log_to_wandb"] = True
    results = []

    # Run the tests `num_runs` times.
    for r in trange(num_runs):
        with wandb_init("eval"):
            # RUN THE TEST
            train_traj, test_traj = test_train(**kwargs)
            # Train and test should be evaluated the same number of times and have the same indices.
            assert len(train_traj) == len(test_traj)  # run the same number of times
            assert train_traj[-1][0] == test_traj[-1][0]  # ending at the same index
            assert len(train_traj[0][1]) == len(test_traj[0][1])  # and having the same number of metrics in each row
            # Accumulate the results.
            for ((epoch, step), tr), (_, te) in zip(train_traj, test_traj):
                # Add the current run index to create an overall index.
                index = [r, epoch, step]
                results.append((index, tr, te))

    return results


def save_results(results, output_path, config):
    """ Transform results from `test_repeats()` into a pandas Dataframe and save to the given file. """
    # These params describing the evaluation should be prepended to each row in the table.
    eval_param_names = ["model", "dataset", "train_examples", "test_examples", "eval_method", "reinit_method",
                        "reinit_params", "opt_params", "classes", "epochs", "batch_size", "lr"]
    eval_params = [config.get(k) for k in eval_param_names]

    # Flatten the data in each row into matrix form, while adding the above metadata.
    full_data = []
    for idx, train_acc, test_acc in results:
        full_data.append(flatten([eval_params, idx, train_acc, test_acc]))

    # Now assemble the data into a dataframe and save it.
    colnames = eval_param_names + ["trial", "epoch", "step", "train_acc", "train_top5_acc", "test_acc", "test_top5_acc"]
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
    steps = result_matrix.index.get_level_values("step")
    final_step = steps.max()
    train_acc = result_matrix.loc[steps == final_step, "train_acc"]
    test_acc = result_matrix.loc[steps == final_step, "test_acc"]
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
