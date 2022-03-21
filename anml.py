import logging
from pathlib import Path

import higher
import numpy as np
import torch
import yaml
from torch.nn.functional import cross_entropy
from torch.optim import SGD, Adam

import models
from models import fine_tuning_setup, load_model
from utils import collect_matching_params, ensure_config_param, get_matching_module, lobotomize
from utils.logging import forward_pass, MetaLearningLog


def check_train_config(config):
    def gt_zero(x):
        return x > 0

    def gte_zero(x):
        return x >= 0

    def of_type(types):
        def test_fn(val):
            return isinstance(val, types)
        return test_fn

    ensure_config_param(config, "batch_size", gt_zero)
    ensure_config_param(config, "num_batches", gt_zero)
    ensure_config_param(config, "val_size", gte_zero)
    ensure_config_param(config, "remember_size", gt_zero)
    ensure_config_param(config, "train_cycles", gt_zero)
    ensure_config_param(config, "inner_lr", gt_zero)
    ensure_config_param(config, "outer_lr", gt_zero)
    ensure_config_param(config, "epochs", gte_zero)
    ensure_config_param(config, "save_freq", gt_zero)
    ensure_config_param(config, "inner_params", of_type((str, list, tuple)))
    ensure_config_param(config, "outer_params", of_type((str, list, tuple)))
    ensure_config_param(config, "output_layer", of_type(str))
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
    print_freq = 1 if verbose > 1 else 10  # if double-verbose, print every iteration
    verbose_freq = print_freq if verbose > 0 else 0  # if verbose, then print verbose info at the same frequency
    log = MetaLearningLog(name, model_args, print_freq, verbose_freq, config["save_freq"], config["full_test"], config)

    # inner optimizer used during the learning phase
    inner_params = collect_matching_params(model, config["inner_params"])
    inner_opt = SGD(inner_params, lr=config["inner_lr"])
    # outer optimizer used during the remembering phase; the learning is propagated through the inner loop
    # optimizations, computing second order gradients.
    outer_params = collect_matching_params(model, config["outer_params"])
    outer_opt = Adam(outer_params, lr=config["outer_lr"])

    # (epochs + 1) because we want the iteration counts to be 1-based, but we still keep the 0th iteration as a sort of
    # "test run" where we make sure we can successfully run full test metrics and save the model checkpoints. This
    # allows the job to fail early if there are issues with any of these things.
    for it in range(config["epochs"] + 1):

        log.outer_begin(it)

        episode = sampler.sample_train(
            batch_size=config["batch_size"],
            num_batches=config["num_batches"],
            remember_size=config["remember_size"],
            val_size=config["val_size"],
            add_inner_train_to_outer_train=not config["remember_only"],
            device=device,
        )
        log.outer_info(it, episode.train_class)

        # To facilitate the propagation of gradients through the model we prevent memorization of training examples by
        # randomizing the weights in the last fully connected layer corresponding to the task that is about to be
        # learned. The config gives us the name of this final output layer.
        output_layer = get_matching_module(model, config["output_layer"])
        lobotomize(output_layer, episode.train_class)

        # higher turns a standard pytorch model into a functional version that can be used to
        # preserve the computation graph across multiple optimization steps
        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (
                fnet,
                diffopt,
        ):
            # Inner loop of 1 random task, in batches, for some number of cycles.
            for i, (ims, labels) in enumerate(episode.train_traj * config["train_cycles"]):
                out, loss, inner_acc = forward_pass(fnet, ims, labels)
                log.inner(it, i, loss, inner_acc, episode, fnet, verbose)
                diffopt.step(loss)

            # Outer "loop" of 1 task (all training batches) + `remember_size` random chars, in a single large batch.
            m_out, m_loss, m_acc = forward_pass(fnet, episode.meta_ims, episode.meta_labels)
            m_loss.backward()

        outer_opt.step()
        outer_opt.zero_grad()

        log.outer_end(it, m_loss, m_acc, episode, fnet, model, sampler, device, verbose)

    log.close(it, model, sampler, device)


def evaluate(model, classes):
    """
    Meta-test-test

    Given a meta-test-trained model, evaluate accuracy on the given data. Assumes the classes are ordered in the order
    in which they are trained on.

    Args:
        model (callable): The model to evaluate.
        classes (list): A list of tensor pairs (inputs, targets), where each tensor is a batch from a single class.

    Returns:
        numpy.ndarray: Array of accuracy per class.
    """
    # NOTE: It would be great to do this operation in one large batch over all classes, but unfortunately that may be
    # too large to fit onto the GPU, for some datasets. For now, for simplicity, we'll assume one batch per class is an
    # appropriate size. In the future we might like to chunk up data more cleverly so we are always maxing out the GPU
    # memory but never going over.
    acc_per_class = np.zeros(len(classes))
    with torch.no_grad():
        for i, (x, y) in enumerate(classes):
            logits = model(x)
            acc_per_class[i] = torch.eq(logits.argmax(dim=1), y).sum().item() / len(y)
    return acc_per_class


def evaluate_and_log(model, name, classes, step, num_seen=None, should_log=False):
    """
    Meta-test-test

    Given a meta-test-trained model, evaluate accuracy on the given data. Assumes the classes are ordered in the order
    in which they are trained on. Also logs the result to Weights & Biases.

    Args:
        model (callable): The model to evaluate.
        name (str): The name to use for the dataset being evaluated (e.g. "train", "test").
        classes (list): A list of tensor pairs (inputs, targets), where each tensor is a batch from a single class.
        step (int): The current step in meta-test-training.
        num_seen (int): The number of classes trained on so far (so we can report performance on "seen" vs. "unseen"
            classes). If `None`, assumes all classes have been seen.
        should_log (bool): Whether to log the results to W&B.

    Returns:
        tuple: An index of the current training step: (step, classes seen).
        numpy.ndarray: Array of accuracy per class.
    """
    import wandb

    acc_per_class = evaluate(model, classes)

    if num_seen is None:
        num_seen = len(classes)

    if should_log:
        wandb.log({
            f"test_{name}.acc": acc_per_class.mean(),
            f"test_{name}.seen_class_acc": acc_per_class[:num_seen].mean(),
        }, step=num_seen)

    return (step, num_seen), acc_per_class


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

    # Sample the learning trajectory.
    train_classes, test_classes = sampler.sample_test(config["classes"], config["train_examples"],
                                                      config["test_examples"], device)
    assert len(train_classes) > 0

    train_perf_trajectory = []
    test_perf_trajectory = []
    eval_freq = config["eval_freq"]
    should_eval = False

    # meta-test-TRAIN
    for idx, train_data in enumerate(train_classes):
        # One full episode: Go through samples one-at-a-time and learn on each one.
        for x, y in zip(*train_data):
            # Create a "batch" of one.
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

            # Gradient step.
            logits = model(x)
            opt.zero_grad()
            loss = cross_entropy(logits, y)
            loss.backward()
            opt.step()

        # Evaluation, once per X number of classes learned. Additionally, record after the first class learned.
        should_eval = eval_freq and (idx == 0 or ((idx + 1) % eval_freq == 0))
        if should_eval:
            # Evaluation on all classes.
            # NOTE: We often only care about performance on classes seen so far. We can extract this after-the-fact,
            # by slicing into the results: `acc_per_class[:idx + 1]`.
            # If we wanted to only run inference on classes already seen, we would take a slice of the data:
            #     evaluate(model, train_classes[:idx + 1])
            # where `idx` is keeping track of the current training class index.
            train_perf_trajectory.append(evaluate_and_log(model, "train", train_classes, idx, idx + 1,
                                                          should_log=log_to_wandb))
            # NOTE: Assumes that test classes are in the same ordering as train classes.
            test_perf_trajectory.append(evaluate_and_log(model, "test", test_classes, idx, idx + 1,
                                                         should_log=log_to_wandb))

    # meta-test-TEST
    # We only need to do this if we didn't already do it in the last iteration of the loop.
    if not should_eval:
        train_perf_trajectory.append(evaluate_and_log(model, "train", train_classes, idx, should_log=log_to_wandb))
        test_perf_trajectory.append(evaluate_and_log(model, "test", test_classes, idx, should_log=log_to_wandb))

    return train_perf_trajectory, test_perf_trajectory
