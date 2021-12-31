import logging
from pathlib import Path

import higher
import numpy as np
import torch
import yaml
from torch.nn.functional import cross_entropy
from torch.nn.init import kaiming_normal_
from torch.optim import SGD, Adam

import models
import utils.storage as storage
from models import LegacyANML
from utils import make_pretty
from utils.logging import forward_pass, Log


def load_model(model_path, sampler_input_shape, device=None):
    model_path = Path(model_path).resolve()
    if model_path.suffix == ".net":
        # Assume this was saved by the storage module, which pickles the entire model.
        model = storage.load(model_path, device=device)
    elif model_path.suffix == ".pt" or model_path.suffix == ".pth":
        # Assume the model was saved in the legacy format:
        #   - Only state_dict is stored.
        #   - Model shape is identified by the filename.
        sizes = [int(num) for num in model_path.name.split("_")[:-1]]
        if len(sizes) != 3:
            raise RuntimeError(f"Unsupported model shape: {sizes}")
        rln_chs, nm_chs, mask_size = sizes
        if mask_size != (rln_chs * 9):
            raise RuntimeError(f"Unsupported model shape: {sizes}")

        # Backward compatibility: Before we constructed the network based on `input_shape` and `num_classes`. At this
        # time, `num_classes` was always 1000 and we always used greyscale 28x28 images.
        input_shape = (1, 28, 28)
        out_classes = 1000
        model = LegacyANML(input_shape, rln_chs, nm_chs, out_classes)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        supported = (".net", ".pt", ".pth")
        raise RuntimeError(f"Unsupported model file type: {model_path}. Expected one of {supported}.")

    logging.debug(f"Model shape:\n{model}")

    # Check if the images we are testing on match the dimensions of the images this model was built for.
    if tuple(model.input_shape) != tuple(sampler_input_shape):
        raise RuntimeError("The specified dataset image sizes do not match the size this model was trained for.\n"
                           f"Data size:  {sampler_input_shape}\n"
                           f"Model size: {model.input_shape}")
    return model


def ensure_config_param(config, key, condition=None):
    if key not in config:
        raise RuntimeError(f'Required key "{key}" not found in config.')
    value = config[key]
    if condition and not condition(value):
        raise RuntimeError(f'Config parameter "{key}" has an invalid value: {value}')


def check_config(config):
    def gt_zero(x):
        return x > 0

    ensure_config_param(config, "batch_size", gt_zero)
    ensure_config_param(config, "num_batches", gt_zero)
    ensure_config_param(config, "val_size", lambda x: x >= 0)
    ensure_config_param(config, "remember_size", gt_zero)
    ensure_config_param(config, "train_cycles", gt_zero)
    ensure_config_param(config, "inner_lr", gt_zero)
    ensure_config_param(config, "outer_lr", gt_zero)
    ensure_config_param(config, "epochs", gt_zero)
    ensure_config_param(config, "save_freq", gt_zero)
    ensure_config_param(config, "inner_params")
    ensure_config_param(config, "outer_params")
    ensure_config_param(config, "output_layer")
    ensure_config_param(config, "model")
    ensure_config_param(config, "model_args")


def collect_opt_params(model, module_list):
    # Special keyword for "all parameters".
    if module_list == "all" or module_list[0] == "all":
        return list(model.parameters())

    params = []
    for name in module_list:
        module = getattr(model, name)
        params.extend(list(module.parameters()))
    return params


def lobotomize(layer, class_num):
    kaiming_normal_(layer.weight[class_num].unsqueeze(0))


def train(sampler, input_shape, config, device="cuda", verbose=0):
    # Output config for reference. Do it before checking config to assist debugging.
    config = make_pretty(config)
    logging.info("\n---- Train Config ----\n" + yaml.dump(config) + "----------------------")
    check_config(config)

    # Create model.
    model_name = config["model"]
    model_args = dict(config["model_args"])  # duplicate so as not to modify original config
    if "input_shape" in models.get_model_arg_names(model_name):
        model_args["input_shape"] = input_shape
    model, model_args = models.make(model_name, device, **model_args)
    logging.info(f"Model shape:\n{model}")

    # Set up progress/checkpoint logger. Name according to the supported input size, just for convenience.
    name = "ANML-" + "-".join(map(str, input_shape))
    print_freq = 1 if verbose > 1 else 10  # if double-verbose, print every iteration
    verbose_freq = print_freq if verbose > 0 else 0  # if verbose, then print verbose info at the same frequency
    log = Log(name, config, model_args, print_freq, verbose_freq, config["save_freq"])

    # inner optimizer used during the learning phase
    inner_params = collect_opt_params(model, config["inner_params"])
    inner_opt = SGD(inner_params, lr=config["inner_lr"])
    # outer optimizer used during the remembering phase; the learning is propagated through the inner loop
    # optimizations, computing second order gradients.
    outer_params = collect_opt_params(model, config["outer_params"])
    outer_opt = Adam(outer_params, lr=config["outer_lr"])

    for it in range(config["epochs"]):

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
        # output_layer = anml.fc
        output_layer = getattr(model, config["output_layer"])
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

    log.close(it, model)


def evaluate(model, classes):
    """
    Meta-test-test

    Given a meta-test-trained model, evaluate accuracy on the given data. Assumes the classes are ordered in the order
    in which they are trained on.

    Args:
        model (callable): The model to evaluate.
        classes (list): A list of tensor pairs (inputs, targets), where each tensor is a batch from a single class.

    Returns:
        list: A list of accuracy per class.
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


def test_train(
        model_path,
        sampler,
        sampler_input_shape,
        num_classes=10,
        num_train_examples=15,
        num_test_examples=5,
        lr=0.01,
        evaluate_complete_trajectory=False,
        device="cuda",
):
    model = load_model(model_path, sampler_input_shape, device)
    model = model.to(device)

    torch.nn.init.kaiming_normal_(model.fc.weight)
    model.nm.requires_grad_(False)
    model.rln.requires_grad_(False)

    train_classes, test_classes = sampler.sample_test(num_classes, num_train_examples, num_test_examples, device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_perf_trajectory = []
    test_perf_trajectory = []

    # meta-test-TRAIN
    for train_data in train_classes:
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

        # Evaluation, once per class.
        if evaluate_complete_trajectory:
            # Evaluation on all classes.
            # NOTE: We often only care about performance on classes seen so far. We can extract this after-the-fact,
            # by slicing into the results: `acc_per_class[:i + 1]`.
            # If we wanted to only run inference on classes already seen, we would take a slice of the data:
            #     evaluate(model, train_classes[:idx + 1])
            # where `idx` is keeping track of the current training class index.
            train_perf_trajectory.append(evaluate(model, train_classes))
            # NOTE: Assumes that test tasks are in the same ordering as train tasks.
            test_perf_trajectory.append(evaluate(model, test_classes))

    # meta-test-TEST
    # We only need to do this if we didn't already do it in the last iteration of the loop.
    if not evaluate_complete_trajectory:
        train_perf_trajectory.append(evaluate(model, train_classes))
        test_perf_trajectory.append(evaluate(model, test_classes))

    return train_perf_trajectory, test_perf_trajectory
