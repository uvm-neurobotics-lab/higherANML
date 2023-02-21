import logging
from pathlib import Path

import higher
import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn.functional import cross_entropy
from torch.optim import SGD, Adam
from tqdm import trange

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

    def one_of(options):
        def test_fn(val):
            return val in options

        return test_fn

    ensure_config_param(config, "batch_size", gt_zero)
    ensure_config_param(config, "num_batches", gt_zero)
    ensure_config_param(config, "val_sample_size", gte_zero)
    ensure_config_param(config, "remember_size", gt_zero)
    ensure_config_param(config, "train_cycles", gt_zero)
    ensure_config_param(config, "inner_lr", gt_zero)
    ensure_config_param(config, "outer_lr", gt_zero)
    ensure_config_param(config, "epochs", gt_zero)
    ensure_config_param(config, "save_freq", gt_zero)
    ensure_config_param(config, "inner_params", of_type((str, list, tuple)))
    ensure_config_param(config, "outer_params", of_type((str, list, tuple)))
    ensure_config_param(config, "output_layer", of_type(str))
    ensure_config_param(config, "train_method", of_type(str))
    ensure_config_param(config, "model", of_type(str))
    config.setdefault("sample_method", "single")
    ensure_config_param(config, "sample_method", one_of(("single", "uniform")))
    config.setdefault("lobotomize", True)
    config.setdefault("full_test", True)


def run_meta_episode(config, episode, model, inner_opt, outer_opt, log, it, verbose):
    # higher turns a standard pytorch model into a functional version that can be used to
    # preserve the computation graph across multiple optimization steps
    with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
        # Inner loop of 1 random task, in batches, for some number of cycles.
        for i, (ims, labels) in enumerate(episode.train_traj * config["train_cycles"]):
            _, inner_loss, inner_acc = forward_pass(fnet, ims, labels)
            log.inner(it, i, inner_loss, inner_acc, episode, fnet, verbose)
            diffopt.step(inner_loss)

        # Outer "loop" of 1 task (all training batches) + `remember_size` random chars, in a single large batch.
        _, m_loss, m_acc = forward_pass(fnet, episode.meta_ims, episode.meta_labels)
        log.outer_step(it, "adapted", m_loss, m_acc, episode, fnet, False, verbose)
        m_loss.backward()

    if config.get("max_grad_norm", 0) > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
    outer_opt.step()

    # The logging step is exactly here, after update and before zeroing out gradients. Thus we can log gradients, and
    # also evaluate the model with the updated weights.
    _, m_loss, m_acc = forward_pass(model, episode.meta_ims, episode.meta_labels)
    log.outer_step(it, "meta", m_loss, m_acc, episode, model, True, verbose)

    outer_opt.zero_grad()


def run_sequential_episode(config, episode, model, inner_opt, outer_opt, log, it, verbose):
    # Sequential loop of a single task, in batches, for some number of cycles.
    for i, (ims, labels) in enumerate(episode.train_traj * config["train_cycles"]):
        out, inner_loss, inner_acc = forward_pass(model, ims, labels)
        log.inner(it, i, inner_loss, inner_acc, episode, model, verbose)
        inner_loss.backward()
        inner_opt.step()
        inner_opt.zero_grad()

    log.outer_step(it, "adapted", inner_loss, inner_acc, episode, model, False, verbose)

    # Outer "loop" of 1 task (all training batches) + `remember_size` random chars, in a single large batch.
    m_out, m_loss, m_acc = forward_pass(model, episode.meta_ims, episode.meta_labels)
    m_loss.backward()

    if config.get("max_grad_norm", 0) > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
    outer_opt.step()

    # The logging step is exactly here, after update and before zeroing out gradients. Thus we can log gradients, and
    # also evaluate the model with the updated weights.
    log.outer_step(it, "meta", m_loss, m_acc, episode, model, True, verbose)

    outer_opt.zero_grad()


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
    log = MetaLearningLog(name, model, model_args, print_freq, verbose_freq, config["save_freq"], config["full_test"],
                          config)
    log.begin(model, sampler, device)

    # inner optimizer used during the learning phase
    inner_params = collect_matching_params(model, config["inner_params"])
    inner_opt = SGD(inner_params, lr=config["inner_lr"])
    # outer optimizer used during the remembering phase; the learning is propagated through the inner loop
    # optimizations, computing second order gradients.
    outer_params = collect_matching_params(model, config["outer_params"])
    outer_opt = Adam(outer_params, lr=config["outer_lr"])
    # Output layer so we can reset output classes when needed (see `lobotomize()`).
    output_layer = get_matching_module(model, config["output_layer"])

    # What kind of training will we be doing?
    train_method = config["train_method"]
    if train_method == "meta":
        run_one_episode = run_meta_episode
    elif train_method == "sequential_episodic":
        run_one_episode = run_sequential_episode
    else:
        raise RuntimeError(f'Unsupported train method: "{train_method}"')

    # MAIN TRAINING LOOP
    for it in range(1, config["epochs"] + 1):  # Epoch/step counts will be 1-based.

        log.outer_begin(it)

        episode = sampler.sample_train(
            batch_size=config["batch_size"],
            num_batches=config["num_batches"],
            remember_size=config["remember_size"],
            val_size=config["val_sample_size"],
            sample_method=config["sample_method"],
            add_inner_train_to_outer_train=not config["remember_only"],
            device=device,
        )
        log.outer_info(it, episode.train_class)

        # To facilitate the propagation of gradients through the model we prevent memorization of training examples by
        # randomizing the weights in the last fully connected layer corresponding to the task that is about to be
        # learned. The config gives us the name of this final output layer.
        if config["lobotomize"]:
            lobotomize(output_layer, episode.train_class)

        run_one_episode(config, episode, model, inner_opt, outer_opt, log, it, verbose)

        log.outer_end(it, model, sampler, device)

    log.close(config["epochs"], model, sampler, device)


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
    acc_per_class = evaluate(model, classes)

    if num_seen is None:
        num_seen = len(classes)

    if should_log:
        import wandb
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
    ensure_config_param(config, "classes", gt_zero)
    ensure_config_param(config, "train_examples", gt_zero)
    ensure_config_param(config, "test_examples", gt_zero)
    if config.get("eval_freq") is None:
        config["eval_freq"] = 0
    ensure_config_param(config, "eval_freq", gte_zero)


def collect_sparse_init_sample(encoder, train_classes, device):
    xs = []
    ys = []
    for images, labels in train_classes:
        # train_data is a pair of (list[img], list[label]). Take the first of each.
        xs.append(images[0])
        ys.append(labels[0])
    xs = encoder(torch.stack(xs).to(device))
    ys = torch.stack(ys)
    return xs, ys


def collect_dense_init_sample(encoder, config, train_classes, device):
    """ Collect as many samples as requested by the init_size parameter. """
    ensure_config_param(config, "init_size", lambda val: val > 0)

    # Sample as uniformly as possible from all classes; first figure out how many from each class.
    num_requested = config["init_size"]
    num_classes = len(train_classes)
    num_per_class = len(train_classes[0][0])  # NOTE: Assumes all classes have the same number of samples.
    num_available = num_per_class * num_classes
    if num_requested >= num_available:
        samples_per_class = num_per_class
        num_extra = 0
    else:
        samples_per_class = num_requested // num_classes
        num_extra = num_requested % num_classes

    xs = []
    ys = []
    for idx, (images, labels) in enumerate(train_classes):
        num_to_sample = samples_per_class
        # The first `num_extra` classes get sampled one extra time.
        if idx < num_extra:
            num_to_sample += 1
        # NOTE: We could use a random.choice() here, but the samples themselves are already randomly sampled.
        xs.append(encoder(images[:num_to_sample].to(device)))
        ys.append(labels[:num_to_sample])

    # Assemble N batches into one batch.
    xs = torch.cat(xs)
    ys = torch.cat(ys)
    return xs, ys


def maybe_collect_init_sample(model, config, train_classes, device):
    """
    Collect samples from the given support set that may be used for initialization of parameters. If the "init_size"
    variable isn't present in the config, then simply take one sample per class according to
    `collect_sparse_init_sample`.

    The samples will be pre-processed into a single batch of feature encodings, taken from the second-to-last layer of
    the model. This is a suitable format to use for, e.g., `lstsq_reinit()`.

    This will return None if no samples are needed according to the config.

    Args:
        model (torch.nn.Module): The model to be partially reinitialized (used to encode the samples).
        config (dict): The config with initialization parameters.
        train_classes (list): The support set containing samples which are allowed to be used for initialization
            purposes. A list of (image, target) tuples, where each element is a pair of tensors from a single class.
        device (torch.device or str): The device on which to run inference.

    Returns:
        xs (torch.Tensor or None): The feature-encodings of all samples in a single batch.
        ys (torch.Tensor or None): The labels corresponding to the samples.
    """
    # Shortcut: We can skip this whole procedure if the samples won't be used.
    if not ("reinit_params" in config) or config.get("reinit_method") != "lstsq":
        return None, None

    # TODO: Fix this so we can extract the next-to-last layer, whatever that is.
    if config.get("init_size"):
        return collect_dense_init_sample(model.encoder, config, train_classes, device)
    else:
        return collect_sparse_init_sample(model.encoder, train_classes, device)


def test_train(sampler, sampler_input_shape, config, device="cuda", log_to_wandb=False):
    check_test_config(config)

    model = load_model(config["model"], sampler_input_shape, device)
    model = model.to(device)
    model.eval()

    # Sample the learning trajectory.
    train_classes, test_classes = sampler.sample_test(config["classes"], config["train_examples"],
                                                      config["test_examples"], device)
    assert len(train_classes) > 0

    init_sample = maybe_collect_init_sample(model, config, train_classes, device)
    opt = fine_tuning_setup(model, config, sampler.num_test_classes(), init_sample)

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


def test_repeats(num_runs, wandb_init, **kwargs):
    """ Run `num_runs` times and collect results into a single list. """
    num_classes = kwargs["config"]["classes"]
    kwargs["log_to_wandb"] = True
    nanfill = [float("nan")]
    results = []

    # Run the tests `num_runs` times.
    for r in trange(num_runs):
        with wandb_init("eval"):
            # RUN THE TEST
            train_traj, test_traj = test_train(**kwargs)
            # Train and test should be evaluated the same number of times.
            assert len(train_traj) == len(test_traj)
            # At the end of training, train and test should both have the full number of classes.
            assert len(train_traj[-1][1]) == len(test_traj[-1][1]) == num_classes
            # Accumulate the results.
            for ((step, classes_trained), tr), (_, te) in zip(train_traj, test_traj):
                # Extend each result to make sure it has the full number of classes.
                tr = list(tr) + nanfill * (num_classes - len(tr))
                te = list(te) + nanfill * (num_classes - len(te))
                # Add the current run index to create an overall index.
                index = [r, step, classes_trained]
                results.append((index, tr, te))

    return results


def save_results(results, output_path, config):
    """ Transform results from `test_repeats()` into a pandas Dataframe and save to the given file. """
    # These params describing the evaluation should be prepended to each row in the table.
    eval_param_names = ["model", "dataset", "train_examples", "test_examples", "eval_method", "reinit_method",
                        "reinit_params", "opt_params", "classes", "lr"]
    eval_params = [config.get(k) for k in eval_param_names]

    # Unflatten the data into one row per class, per epoch.
    full_data = []
    for idx, train_acc, test_acc in results:
        # All params which apply to all results in this epoch.
        prefix = eval_params + idx
        for c in range(config["classes"]):
            # A pair of (train, test) results per class.
            full_data.append(prefix + [c, train_acc[c], test_acc[c]])

    # Now assemble the data into a dataframe and save it.
    colnames = eval_param_names + ["trial", "epoch", "classes_trained", "class_id", "train_acc", "test_acc"]
    result_matrix = pd.DataFrame(full_data, columns=colnames)
    # Although it makes some operations more cumbersome, we can save a lot of space and maybe some time by treating
    # most of the columns as a MultiIndex. Also makes indexing easier. All but the final metrics columns.
    result_matrix.set_index(colnames[:-2], inplace=True)
    result_matrix.to_pickle(output_path)
    print(f"Saved result matrix of size {result_matrix.shape} to: {output_path}")
    return result_matrix


def report_summary(result_matrix):
    """ Take dataframe resulting from `save_results()` and compute and print some summary metrics. """
    # Average over all classes to get overall performance numbers, by grouping by columns other than class.
    non_class_columns = list(filter(lambda x: x != "class_id", result_matrix.index.names))
    avg_over_classes = result_matrix.groupby(non_class_columns).mean()
    # Get just the final test accuracy of each run (when we've trained on all classes).
    classes_trained = avg_over_classes.index.get_level_values("classes_trained")
    num_classes = classes_trained.max()
    train_acc = avg_over_classes.loc[classes_trained == num_classes, "train_acc"]
    test_acc = avg_over_classes.loc[classes_trained == num_classes, "test_acc"]
    # NOTE: Right now we're just printing to console, but it may be useful in the future to report this back to the
    # original training job as a summary metric? Example here: https://docs.wandb.ai/guides/track/log#summary-metrics
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
