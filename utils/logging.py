"""
Utilities for logging progress metrics and saving checkpoints.
"""

import logging
from collections import Counter
from pathlib import Path
from time import time, strftime, gmtime
from typing import Union

import numpy as np
import scipy
import torch
import wandb
from torch.nn.functional import cross_entropy

import launch_eval_map as evaljob
from utils import ensure_config_param, update_with_keys
from utils.storage import save


def accuracy(preds, labels):
    assert len(preds) == len(labels)
    if len(preds) == 0:
        return np.nan
    return (preds.argmax(axis=1) == labels).sum().item() / len(labels)


def topk_accuracy(preds, labels, topk: Union[int, tuple, list] = 1):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    assert len(preds) == len(labels)
    if not isinstance(topk, (tuple, list)):
        topk = [topk]

    # Get largest K label indices in each row.
    maxk = max(topk)
    _, largest_indices = preds.topk(maxk, dim=1)
    largest_indices = largest_indices.t()
    # Mark as True any place where one of the top K indices matches the label index.
    # Now columns are examples and rows are 1 thru K.
    correct = largest_indices.eq(labels.view(1, -1).expand_as(largest_indices))

    res = []
    for k in topk:
        # Take only the first K rows (corresponding to the top K answers for each sample), and sum.
        # There cannot be more than one true value per sample, so the total sum is the total # correct.
        # If we want a vector that would tell us which samples were correct, we could use `.sum(0, keepdim=True)`.
        num_correct = correct[:k].sum().item()
        res.append(num_correct / len(labels))
    return res


def spread(preds):
    """
    Calculate the "spread" of a batch of classifications.

    Spread is inspired by the concept of entropy, but it is not really entropy. It is formulated such that: if all
    predictions have the same class, spread is at a minimum (0.0); if each prediction has a different value, spread is
    maximized (-log(1/len(preds))).
    """
    # Get a mapping of class index --> # of times this class appears.
    cnt = Counter(preds.argmax(axis=1).tolist())
    # Get the frequencies as a list of histogram bins.
    bins = list(cnt.values())
    # Force the histogram to have as many bins as there are predictions. Now the only way to maximize entropy is to
    # have a count of 1 in each bin.
    if len(bins) < len(preds):
        bins.extend([0] * (len(preds) - len(bins)))
    return scipy.stats.entropy(bins)


def normalized_spread(preds):
    """
    Calculate a normalized version of `spread()`, to the range [0.0, 1.0].
    """
    return spread(preds) / (-np.log(1 / len(preds)))


def classes_sorted_by_frequency(preds):
    # Map of class index --> frequency
    cnt = Counter(preds.argmax(axis=1).tolist())
    # Sort by decreasing order of frequency.
    return sorted(cnt.items(), key=lambda e: e[1], reverse=True)


def fraction_wrong_predicted_as_train_class(preds, labels, train_class):
    pred_classes = preds.argmax(axis=1)
    is_wrong = pred_classes != labels
    num_predicted_as_train = (pred_classes[is_wrong] == train_class).sum().item()
    if is_wrong.sum() == 0:
        return np.nan
    return num_predicted_as_train / is_wrong.sum().item()


def print_oneline(set_list, print_fn, metric):
    msg = ""
    for name, output, labels in set_list:
        if msg:
            msg += ", "
        msg += metric(name, output, labels)
    print_fn(msg)


def print_validation_stats(episode, train_out, rem_out, val_out, verbose, print_fn):
    # For convenience of running the same code on all three sets.
    set_list = [("Train", train_out, episode.train_labels)]
    if len(rem_out) > 0:
        set_list.append(("Remember", rem_out, episode.rem_labels))
    if len(val_out) > 0:
        set_list.append(("Val", val_out, episode.val_labels))

    # Loss & Accuracy
    print_oneline(set_list, print_fn, lambda name, output, labels: f"{name} Acc = {accuracy(output, labels) :.1%}")
    # TODO: report accuracy of non-train classes instead of remember set?
    # TODO: report how much of each set was "seen" before?
    # TODO: or separately report total fraction of classes seen and examples seen
    # TODO: report overall accuracy on "things seen so far"? Maybe explicitly sample from things seen.

    # "Entropy"
    print_oneline(set_list, print_fn, lambda name, output, labels: f"{name} Spread = {normalized_spread(output) :.2f}")

    # Top Classes
    # TODO: Start tracking seen and unseen classes here?
    # TODO: Also in outer loop printout.
    for name, output, labels in set_list:
        mode = classes_sorted_by_frequency(output)[0]
        mode_freq = mode[1] / len(output)
        print_fn(f"Most frequent {name} prediction: {mode[0]} ({mode_freq:.1%} of predictions)")

    # Print percentage of over-prediction of target class. (How many "remember" items were wrong b/c they were
    # predicted as the class currently being learned.)
    for name, output, labels in set_list[1:]:  # Skip the training set.
        print_fn(f"Portion of {name} wrongly predicted as {episode.train_class} = "
                 f"{fraction_wrong_predicted_as_train_class(output, labels, episode.train_class):.1%}")

    # If super verbose, print the entire prediction.
    if verbose >= 3:
        for name, output, labels in set_list:
            pred_label_pairs = np.array(list(zip(output.argmax(axis=1), labels)))
            print_fn(f"\n{name} (pred, label) pairs:")
            print_fn(str(pred_label_pairs))


def log_gradient_stats(metrics, model, name=None):
    if name and not name.endswith("/"):
        name += "/"
    elif name is None:
        name = ""

    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return

    norm_type = 2
    device = params[0].grad.device
    # Computed the same way as in torch.nn.utils.clip_grad_norm_().
    norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in params]), norm_type)
    metrics["gradients/" + name + "norm"] = norm


def forward_pass(model, ims, labels):
    if len(ims) == 0:
        out = torch.tensor([])
        loss = np.nan
        acc = np.nan
    else:
        out = model(ims)
        loss = cross_entropy(out, labels)
        acc = accuracy(out, labels)
    return out, loss, acc


class eval_mode:
    """
    Context-manager that both disables gradient calculation (`torch.no_grad()`) and sets modules in eval mode
    (`torch.nn.Module.eval()`).
    """
    def __init__(self, module):
        self.module = module
        self.prev_grad = False
        self.prev_train = False

    def __enter__(self):
        self.prev_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        self.prev_train = self.module.training
        self.module.train(False)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_grad_enabled(self.prev_grad)
        self.module.train(self.prev_train)


def overall_accuracy(model, all_batches, device, topk=(1, 5), print_fn=None):
    """
    Evaluate the model on each batch and return the average accuracy over all samples. If there are no batches, or all
    batches are empty, then this will return NaN.
    """
    # Allow the tensors to be empty.
    if len(all_batches) == 0:
        return [np.nan] * len(topk)

    if print_fn:
        print_fn(f"Computing accuracy of {len(all_batches)} batches...")

    acc_per_batch = []
    with eval_mode(model):
        for ims, labels in all_batches:
            ims, labels = ims.to(device), labels.to(device)
            out = model(ims)
            acc = topk_accuracy(out, labels, topk=topk)
            acc_per_batch.append((*acc, len(labels)))

    acc_per_batch = np.array(acc_per_batch)
    weights = acc_per_batch[:, -1]
    if weights.sum() == 0:
        return [np.nan] * len(topk)
    else:
        return [np.ma.average(acc_per_batch[:, i], weights=weights) for i in range(len(topk))]


def check_eval_config(eval_config):
    ensure_config_param(eval_config, "dataset")
    ensure_config_param(eval_config, "model")
    ensure_config_param(eval_config, "classes")
    eval_freq = eval_config.get("eval_freq")
    if eval_freq is None:
        eval_config["eval_freq"] = max(1, eval_config["classes"] // 20)


class BaseLog:

    def __init__(self, name, model_args, save_freq, full_test=True, config=None):
        self.name = name
        self.model_args = model_args
        self.save_freq = save_freq
        self.full_test = full_test
        if config is None:
            config = {}
        self.config = config
        self.eval_steps = config.get("eval_steps")
        if self.eval_steps is None:
            self.eval_steps = []
        self.eval_config = config.get("eval")
        if self.eval_steps and not self.eval_config:
            raise RuntimeError("You must supply an evaluation config, or else disable eval_steps.")
        self.logger = logging.getLogger(name)
        self.save_path = Path("./trained_anmls")
        self.save_path.mkdir(exist_ok=True)
        self.last_save_step = -1

    def warning(self, msg):
        self.logger.warning(msg)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def begin(self, model, sampler, device):
        if self.config.get("save_initial_model") or (0 in self.eval_steps):
            self.maybe_save_and_eval(0, model, sampler, device, should_save=True)

    @torch.no_grad()
    def maybe_save_and_eval(self, it, model, sampler, device, should_save=None, should_eval=None):
        if should_eval is None:
            should_eval = (it in self.eval_steps)
        if not should_save:
            # Turn on saving if it is time to save, or if evaluation requires it.
            should_save = should_eval or (it % self.save_freq == 0)
        # Do not save if the model for this iteration was already saved (the same iteration can be called twice).
        should_save &= (it != self.last_save_step)

        if not should_save and not should_eval:
            # Nothing to do.
            return

        model_path = self.save_path / f"{self.name}-{it}.net"
        if should_save:
            # Run full test on training data.
            if self.full_test:
                start = time()
                full_train_acc1, full_train_acc5 = overall_accuracy(model, sampler.full_train_data(device), device,
                                                                    print_fn=self.debug)
                full_val_acc1, full_val_acc5 = overall_accuracy(model, sampler.full_val_data(device), device,
                                                                print_fn=self.debug)
                end = time()
                elapsed = end - start
                self.info(f"Saved Model Performance:"
                          f" Train Top-1 Acc = {full_train_acc1:.1%} | Train Top-5 Acc = {full_train_acc5:.1%} |"
                          f" Validation Top-1 Acc = {full_val_acc1:.1%} | Validation Top-5 Acc = {full_val_acc5:.1%}"
                          f" (Time to Eval = {strftime('%H:%M:%S', gmtime(elapsed))})")
                wandb.log({
                    "overall/train.acc": full_train_acc1,
                    "overall/val.acc": full_val_acc1,
                    "overall/train.top5_acc": full_train_acc5,
                    "overall/val.top5_acc": full_val_acc5,
                }, step=it)

            # Save the model.
            self.last_save_step = it
            save(model, model_path, **self.model_args)

        # Launch full evaluation of the model as a separate job.
        if should_eval:
            self.launch_eval(model_path)

    def launch_eval(self, model_path):
        cfg_list = self.eval_config if isinstance(self.eval_config, list) else [self.eval_config]
        for eval_config in cfg_list:
            # If the config only has one key, then this names the "flavor" of the evaluation, and the corresponding
            # value is actually the config.
            flavor = None
            if len(eval_config) == 1:
                flavor, eval_config = next(iter(eval_config.items()))
            eval_config = eval_config.copy()  # copy before editing
            eval_config["model"] = str(model_path.resolve())
            update_with_keys(self.config, eval_config, ["project", "entity", "group"])
            if not eval_config.get("group"):
                # If group is not defined, try a couple more backups, because we really want these runs grouped.
                group = str(wandb.run.group)
                if group:
                    eval_config["group"] = group
                else:
                    eval_config["group"] = wandb.run.name
            check_eval_config(eval_config)
            retcode = evaljob.launch(eval_config, flavor=flavor, cluster=self.config["cluster"],
                                     launcher_args=["--mem=64G"], force=True)
            if retcode != 0:
                self.warning(f"Eval job may not have launched. Launcher exited with code {retcode}. See above for"
                             " possible errors.")

    def close(self, it, model, sampler, device):
        # Eval if there is is at least one desired eval beyond this point in training, but this point is not already
        # included.
        has_larger = any([s > it for s in self.eval_steps])
        has_equal = any([s == it for s in self.eval_steps])
        should_eval = has_larger and not has_equal
        self.maybe_save_and_eval(it, model, sampler, device, should_save=True, should_eval=should_eval)


class StandardLog(BaseLog):

    def __init__(self, name, model, model_args, print_freq=100, save_freq=2000, full_test=True, config=None):
        super().__init__(name, model_args, save_freq, full_test, config)
        self.start = -1
        self.print_freq = print_freq
        wandb.watch(model, log_freq=print_freq)  # log gradient histograms automatically

    def epoch(self, it, epoch, sampler, optimizer):
        wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=it)  # NOTE: assumes only one param group for now.
        self.info(f"---- Beginning Epoch {epoch}: {len(sampler.train_loader)} batches, {sampler.batch_size} samples"
                  " each ----")

    def step(self, it, epoch, loss, acc, out, labels, model, sampler, device):
        metrics = {}
        acc1, acc5 = topk_accuracy(out, labels, topk=(1, 5))
        metrics["epoch"] = epoch
        metrics["loss"] = loss.item()
        metrics["batch_train_acc"] = acc
        metrics["batch_train_acc_top1"] = acc1
        metrics["batch_train_acc_top5"] = acc5
        log_gradient_stats(metrics, model)

        time_to_print = (it % self.print_freq == 0)
        if time_to_print:
            # Track runtime since last printout.
            if self.start < 0:
                self.start = time()
                elapsed = 0
            else:
                end = time()
                elapsed = end - self.start
                self.start = end
                metrics["runtime"] = elapsed

            self.info(f"Step {it}: Batch Loss = {loss.item():.3f} | Batch Train Acc = {acc:.1%}"
                      f" | Batch Train Top 5 Acc = {acc5:.1%} ({strftime('%H:%M:%S', gmtime(elapsed))})")

        wandb.log(metrics, step=it)
        self.maybe_save_and_eval(it, model, sampler, device)


class MetaLearningLog(BaseLog):

    def __init__(self, name, model, model_args, print_freq=10, verbose_freq=None, save_freq=1000, full_test=True,
                 config=None):
        super().__init__(name, model_args, save_freq, full_test, config)
        self.start = -1
        self.print_freq = print_freq
        self.verbose_freq = verbose_freq
        wandb.watch(model, log_freq=print_freq * 10)  # log gradient histograms automatically

    def outer_begin(self, it):
        if self.start < 0:
            self.start = time()

    def outer_info(self, it, train_class):
        if it % self.print_freq == 0:
            self.info(f"**** Episode {it}: Learning on class {train_class} ****")

    @torch.no_grad()
    def inner(self, outer_it, inner_it, inner_loss, inner_acc, episode, model, verbose):
        wandb.log({"inner/loss": inner_loss, "inner/acc": inner_acc}, step=outer_it)

        # Only print inner loop info when verbose is turned on.
        if (self.verbose_freq > 0) and (outer_it % self.verbose_freq == 0):
            if inner_it < 2:
                # TODO: plot change in performance b/w first and second iteration.
                # TODO: plot change in performance b/w first and last iteration.
                # TODO: report when/if train acc reaches 100% - some idea of how fast learning is happening
                # TODO: and/or plot some idea of how much outputs are changing, in general
                val_out = forward_pass(model, episode.val_ims, episode.val_labels)[0]
                val_acc = accuracy(val_out, episode.val_labels)
                self.debug(f"  Inner iter {inner_it}: Batch Loss = {inner_loss:.5f}, Batch Acc = {inner_acc:.1%}"
                           f", Sampled Val Acc = {val_acc:.1%}")
                train_out = forward_pass(model, episode.train_ims, episode.train_labels)[0]
                rem_out = forward_pass(model, episode.rem_ims, episode.rem_labels)[0]
                print_validation_stats(episode, train_out, rem_out, val_out, verbose,
                                       lambda msg: self.debug("    " + msg))

    @torch.no_grad()
    def outer_step(self, it, name, loss, acc, episode, model, log_gradients, verbose):
        capname = name.capitalize()
        metrics = {}
        time_to_print = (it % self.print_freq == 0)
        time_to_verbose_print = (self.verbose_freq > 0) and (it % self.verbose_freq == 0)

        if time_to_print or time_to_verbose_print:
            train_out = forward_pass(model, episode.train_ims, episode.train_labels)[0]
            rem_out = forward_pass(model, episode.rem_ims, episode.rem_labels)[0]
            val_out = forward_pass(model, episode.val_ims, episode.val_labels)[0]

            if time_to_print:
                train_acc = accuracy(train_out, episode.train_labels)
                rem_acc = accuracy(rem_out, episode.rem_labels)
                val_acc = accuracy(val_out, episode.val_labels)
                metrics[name + "/loss"] = loss.item()
                metrics[name + "/acc"] = acc
                metrics[name + "/train_acc"] = train_acc
                metrics[name + "/remember_acc"] = rem_acc
                metrics[name + "/valid_acc"] = val_acc
                if log_gradients:
                    log_gradient_stats(metrics, model, name)
                self.info(f"  {capname} Model on Episode {it}: Meta-Loss = {loss.item():.3f} | Meta-Acc = {acc:.1%}"
                          f" | Train = {train_acc:.1%} | Remember = {rem_acc:.1%} | Sampled Val = {val_acc:.1%}")

            # If verbose, then also evaluate the new meta-model on the previous train/validation data so we can see the
            # impact of meta-learning.
            if time_to_verbose_print:
                # TODO: Idea: report difference b/w meta-model and end-model perf.
                self.debug(f"  {capname} Model Details:")
                print_validation_stats(episode, train_out, rem_out, val_out, verbose,
                                       lambda msg: self.debug("    " + msg))

        wandb.log(metrics, step=it)

    @torch.no_grad()
    def outer_end(self, it, model, sampler, device):
        metrics = {}
        if it % self.print_freq == 0:
            end = time()
            elapsed = end - self.start
            self.start = -1
            self.info(f"  Time since last print: {strftime('%H:%M:%S', gmtime(elapsed))}")

        wandb.log(metrics, step=it)
        self.maybe_save_and_eval(it, model, sampler, device)
