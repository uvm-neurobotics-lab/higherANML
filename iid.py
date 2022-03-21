import logging
from pathlib import Path

import wandb
import yaml
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

import models
import utils.optimization
from models import fine_tuning_setup, load_model
from utils import (ensure_config_param)
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
    log = StandardLog(name, model_args, print_freq, config["save_freq"], config["full_test"], config)

    optimizer = utils.optimization.optimizer_from_config(config, model.parameters())
    scheduler = utils.optimization.scheduler_from_config(config, optimizer)

    # BEGIN TRAINING
    step = 0
    max_steps = config.get("max_steps", float("inf"))
    for epoch in range(config["epochs"]):
        step = run_one_epoch(sampler, model, optimizer, log, epoch, step, max_steps, device)
        if step >= max_steps:
            break
        scheduler.step()

    log.close(step, model, sampler, device)


def run_one_epoch(sampler, model, optimizer, log, epoch, step, max_steps=float("inf"), device=None):
    """ Run one training epoch. """
    log.epoch(step, epoch, model, sampler, device)
    model.train()

    for images, labels in sampler.train_loader:

        # Move data to GPU once loaded.
        images, labels = images.to(device), labels.to(device)

        # Forward pass.
        out, loss, acc = forward_pass(model, images, labels)

        # Record accuracy and other metrics.
        log.step(step, epoch, loss, acc, out, labels, model, sampler, device)

        # Backpropagate.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        if step >= max_steps:
            break

    return step


def evaluate_and_log(model, name, loader, step, device):
    acc1, acc5 = overall_accuracy(model, loader, device)
    wandb.log({f"test_{name}.acc": acc1, f"test_{name}.top5_acc": acc5}, step=step)
    return acc1, acc5


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
    ensure_config_param(config, "reinit_params", of_type((str, list, tuple)))
    ensure_config_param(config, "opt_params", of_type((str, list, tuple)))
    ensure_config_param(config, "classes", gt_zero)
    ensure_config_param(config, "train_examples", gt_zero)
    ensure_config_param(config, "test_examples", gt_zero)
    ensure_config_param(config, "lr", gt_zero)
    if config.get("eval_freq") is None:
        config["eval_freq"] = 0
    ensure_config_param(config, "eval_freq", gte_zero)


def test_train(sampler, sampler_input_shape, config, device="cuda", log_to_wandb=False):
    check_test_config(config)

    model = load_model(config["model"], sampler_input_shape, device)
    model = model.to(device)
    model.eval()

    opt = fine_tuning_setup(model, config)

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

    train_perf_trajectory = []
    test_perf_trajectory = []
    eval_freq = config["eval_freq"]
    should_eval = False
    step = 0

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
            should_eval = eval_freq and (step == 0 or ((step + 1) % eval_freq == 0))
            if should_eval:
                train_perf_trajectory.append(evaluate_and_log(model, "train", query_loader, step + 1, device))
                test_perf_trajectory.append(evaluate_and_log(model, "test", support_loader, step + 1, device))

            step += 1

    # meta-test-TEST
    # We only need to do this if we didn't already do it in the last iteration of the loop.
    if not should_eval:
        train_perf_trajectory.append(evaluate_and_log(model, "train", query_loader, step, device))
        test_perf_trajectory.append(evaluate_and_log(model, "test", support_loader, step, device))

    return train_perf_trajectory, test_perf_trajectory
