import logging
from itertools import count
from pathlib import Path

import higher
import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torch.nn.init import kaiming_normal_
from torch.optim import SGD, Adam

import utils.storage as storage
from models import ANML, LegacyANML, recommended_number_of_convblocks
from utils import collate_images, divide_chunks
from utils.logging import forward_pass, Log


def create_model(input_shape, nm_channels, rln_channels, device):
    # TODO: Auto-size this instead.
    # num_classes = max(sampler.num_train_classes(), sampler.num_test_classes())
    num_classes = 1000
    # For backward compatibility, we use the original ANML if the images are <=30 px.
    # Otherwise, we automatically size the net as appropriate.
    if input_shape[-1] <= 30:
        # Temporarily turn off the "legacy" model so we can test parity with the new model.
        # model_args = {
        #     "input_shape": input_shape,
        #     "rln_chs": rln_channels,
        #     "nm_chs": nm_channels,
        #     "num_classes": num_classes,
        # }
        # anml = LegacyANML(**model_args)
        model_args = {
            "input_shape": input_shape,
            "rln_chs": rln_channels,
            "nm_chs": nm_channels,
            "num_classes": num_classes,
            "num_conv_blocks": 3,
            "pool_rln_output": False,
        }
        anml = ANML(**model_args)
    else:
        model_args = {
            "input_shape": input_shape,
            "rln_chs": rln_channels,
            "nm_chs": nm_channels,
            "num_classes": num_classes,
            "num_conv_blocks": recommended_number_of_convblocks(input_shape),
            "pool_rln_output": True,
        }
        anml = ANML(**model_args)
    anml.to(device)
    logging.info(f"Model shape:\n{anml}")
    return anml, model_args


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


def lobotomize(layer, class_num):
    kaiming_normal_(layer.weight[class_num].unsqueeze(0))


def train(
        sampler,
        input_shape,
        rln_channels,
        nm_channels,
        batch_size=1,
        num_batches=20,
        remember_size=64,
        train_cycles=1,
        inner_lr=1e-1,
        outer_lr=1e-3,
        its=30000,
        device="cuda",
        verbose=0
):
    assert rln_channels > 0
    assert nm_channels > 0
    assert batch_size > 0
    assert num_batches > 0
    assert remember_size > 0
    assert train_cycles > 0
    assert inner_lr > 0
    assert outer_lr > 0
    assert its > 0

    anml, model_args = create_model(input_shape, nm_channels, rln_channels, device)

    # Set up progress/checkpoint logger. Name according to the supported input size, just for convenience.
    name = "ANML-" + "-".join(map(str, input_shape))
    print_freq = 1 if verbose > 1 else 10  # if double-verbose, print every iteration
    verbose_freq = print_freq if verbose > 0 else 0  # if verbose, then print verbose info at the same frequency
    log = Log(name, model_args, print_freq, verbose_freq)

    # inner optimizer used during the learning phase
    inner_opt = SGD(list(anml.rln.parameters()) + list(anml.fc.parameters()), lr=inner_lr)
    # outer optimizer used during the remembering phase; the learning is propagated through the inner loop
    # optimizations, computing second order gradients.
    outer_opt = Adam(anml.parameters(), lr=outer_lr)

    for it in range(its):

        log.outer_begin()

        num_train_ex = batch_size * num_batches
        train_data, train_class, (valid_ims, valid_labels) = sampler.sample_train(
            batch_size=batch_size,
            num_batches=num_batches,
            remember_size=remember_size,
            device=device,
        )
        log.outer_info(it, train_class)

        # To facilitate the propagation of gradients through the model we prevent memorization of
        # training examples by randomizing the weights in the last fully connected layer corresponding
        # to the task that is about to be learned
        lobotomize(anml.fc, train_class)

        # higher turns a standard pytorch model into a functional version that can be used to
        # preserve the computation graph across multiple optimization steps
        with higher.innerloop_ctx(anml, inner_opt, copy_initial_weights=False) as (
                fnet,
                diffopt,
        ):
            # Inner loop of 1 random task, in batches, for some number of cycles.
            for i, (ims, labels) in enumerate(train_data * train_cycles):
                out, loss, inner_acc = forward_pass(fnet, ims, labels)
                log.inner(it, i, train_class, loss, inner_acc, valid_ims, valid_labels, num_train_ex, fnet, verbose)
                diffopt.step(loss)

            # Outer "loop" of 1 task (all training batches) + `remember_size` random chars, in a single large batch.
            m_out, m_loss, m_acc = forward_pass(fnet, valid_ims, valid_labels)
            m_loss.backward()

        outer_opt.step()
        outer_opt.zero_grad()

        log.outer_end(it, train_class, m_out, m_loss, m_acc, valid_ims, valid_labels, num_train_ex, anml, verbose)

    log.close(it, anml)


def evaluate(model, data, num_examples_per_class=5):
    """
    Meta-test-test

    Given a meta-test-trained model, evaluate accuracy on the given data batch. Assumes this batch can be sub-divided
    into N classes where each class has `num_examples_per_class` examples.

    Args:
        model (callable): The model to evaluate.
        data (tuple): A pair of tensors (inputs, targets).
        num_examples_per_class (int): Number of examples per class; number of rows in the `data` should be a
            multiple of this.

    Returns:
        list: A list of accuracy per class.
    """
    x, y = data
    with torch.no_grad():
        logits = model(x)
        ys = list(divide_chunks(y, num_examples_per_class))
        classes = list(divide_chunks(logits, num_examples_per_class))
        acc_per_class = [
            torch.eq(preds.argmax(dim=1), ys).sum().item() / num_examples_per_class
            for preds, ys in zip(classes, ys)
        ]
    return np.array(acc_per_class)


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

    train_data, test_data = sampler.sample_test(num_classes, num_train_examples, num_test_examples, device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_perf_trajectory = []
    test_perf_trajectory = []

    # meta-test-TRAIN
    # Go through samples one-at-a-time.
    for i, x, y in zip(count(1), *train_data):
        # Create a "batch" of one.
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        # Gradient step.
        logits = model(x)
        opt.zero_grad()
        loss = cross_entropy(logits, y)
        loss.backward()
        opt.step()

        # Evaluation, once per class (hence the mod).
        if evaluate_complete_trajectory and i % num_train_examples == 0:
            # Evaluation on all classes.
            # NOTE: We often only care about performance on classes seen so far. We can extract this after-the-fact,
            # by slicing into the results: `acc_per_class[:i]` (if `i` is 1-based, as it is here).
            # If we wanted to only run inference on classes already seen, we would pre-slice the tensors:
            #     train_eval = tuple(d[:i] for d in train_data)
            # and for test data:
            #     num_classes_seen = i // num_train_examples
            #     test_eval = tuple(d[:num_classes_seen * num_test_examples] for d in test_data)
            train_perf_trajectory.append(evaluate(model, train_data, num_train_examples))
            # NOTE: Assumes that test tasks are in the same ordering as train tasks.
            test_perf_trajectory.append(evaluate(model, test_data, num_test_examples))

    # meta-test-TEST
    train_perf_trajectory.append(evaluate(model, train_data, num_train_examples))
    test_perf_trajectory.append(evaluate(model, test_data, num_test_examples))

    return train_perf_trajectory, test_perf_trajectory
