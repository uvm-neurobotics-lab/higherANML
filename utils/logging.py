"""
Utilities for logging progress metrics and saving checkpoints.
"""

import logging
import os
from collections import Counter
from pathlib import Path
from time import time, strftime, gmtime

import numpy as np
import scipy
import torch
import yaml
from torch.nn.functional import cross_entropy

from utils.storage import save


def accuracy(preds, labels):
    assert len(preds) == len(labels)
    if len(preds) == 0:
        return np.nan
    return (preds.argmax(axis=1) == labels).sum().item() / len(labels)


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
        set_list.append(("Test", val_out, episode.val_labels))

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


def overall_accuracy(model, all_batches, print_fn):
    """
    Evaluate the model on each batch and return the average accuracy over all samples. If there are no batches, or all
    batches are empty, then this will return NaN.
    """
    # Allow the tensors to be empty.
    if len(all_batches) == 0:
        return np.nan
    print_fn(f"Computing accuracy of {len(all_batches)} batches, {len(all_batches[0][0])} samples per batch...")
    acc_per_batch = np.array([(forward_pass(model, ims, labels)[2], len(labels)) for ims, labels in all_batches])
    accs = acc_per_batch[:, 0]
    weights = acc_per_batch[:, 1]
    if weights.sum() == 0:
        return np.nan
    else:
        return np.ma.average(accs, weights=weights)


class Log:
    def __init__(self, name, config, model_args, print_freq=10, verbose_freq=None, save_freq=1000):
        self.start = -1
        self.name = name
        self.config = config
        self.model_args = model_args
        self.print_freq = print_freq
        self.verbose_freq = verbose_freq
        self.save_freq = save_freq
        self.logger = logging.getLogger(name)
        self.save_path = Path("./trained_anmls")
        self.save_path.mkdir(exist_ok=True)
        with open(self.save_path / "train-config.yml", "w") as f:
            yaml.dump(config, f)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def outer_begin(self, it):
        if self.start < 0:
            self.start = time()

    def outer_info(self, it, train_class):
        if it % self.print_freq == 0:
            self.info(f"**** Outer loop {it}: Learning on class {train_class}...")

    def inner(self, outer_it, inner_it, inner_loss, inner_acc, episode, model, verbose):
        # Only print inner loop info when verbose is turned on.
        if (self.verbose_freq > 0) and (outer_it % self.verbose_freq == 0):
            if inner_it < 2:
                # TODO: plot change in performance b/w first and second iteration.
                # TODO: plot change in performance b/w first and last iteration.
                # TODO: report when/if train acc reaches 100% - some idea of how fast learning is happening
                # TODO: and/or plot some idea of how much outputs are changing, in general
                val_out = forward_pass(model, episode.val_ims, episode.val_labels)[0]
                val_acc = accuracy(val_out, episode.val_labels)
                self.debug(f"  Inner iter {inner_it}: Loss = {inner_loss:.5f}, Acc = {inner_acc:.1%}"
                           f", Test Acc = {val_acc:.1%}")
                train_out = forward_pass(model, episode.train_ims, episode.train_labels)[0]
                rem_out = forward_pass(model, episode.rem_ims, episode.rem_labels)[0]
                print_validation_stats(episode, train_out, rem_out, val_out, verbose,
                                       lambda msg: self.debug("    " + msg))

    def outer_end(self, it, loss, acc, episode, adapted_model, meta_model, sampler, device, verbose):
        time_to_print = (it % self.print_freq == 0)
        time_to_verbose_print = (self.verbose_freq > 0) and (it % self.verbose_freq == 0)
        if time_to_print or time_to_verbose_print:
            ada_train_out = forward_pass(adapted_model, episode.train_ims, episode.train_labels)[0]
            ada_rem_out = forward_pass(adapted_model, episode.rem_ims, episode.rem_labels)[0]
            ada_val_out = forward_pass(adapted_model, episode.val_ims, episode.val_labels)[0]

            if time_to_print:
                end = time()
                elapsed = end - self.start
                self.start = -1

                train_acc = accuracy(ada_train_out, episode.train_labels)
                rem_acc = accuracy(ada_rem_out, episode.rem_labels)
                val_acc = accuracy(ada_val_out, episode.val_labels)
                self.info(f"  Final Meta-Loss = {loss.item():.3f} | Meta-Acc = {acc:.1%} | Train Acc = {train_acc:.1%}"
                          f" | Remember Acc = {rem_acc:.1%} | Test Acc = {val_acc:.1%}"
                          f" ({strftime('%H:%M:%S', gmtime(elapsed))})")

            # If verbose, then also evaluate the new meta-model on the previous train/validation data so we can see the
            # impact of meta-learning.
            if time_to_verbose_print:
                # TODO: Idea: report difference b/w meta-model and end-model perf.
                self.debug("  End Model Performance:")
                print_validation_stats(episode, ada_train_out, ada_rem_out, ada_val_out, verbose,
                                       lambda msg: self.debug("    " + msg))

                self.debug("  Meta-Model Performance:")
                meta_train_out = forward_pass(meta_model, episode.train_ims, episode.train_labels)[0]
                meta_rem_out = forward_pass(meta_model, episode.rem_ims, episode.rem_labels)[0]
                meta_val_out = forward_pass(meta_model, episode.val_ims, episode.val_labels)[0]
                print_validation_stats(episode, meta_train_out, meta_rem_out, meta_val_out, verbose,
                                       lambda msg: self.debug("    " + msg))

        if it % self.save_freq == 0:
            meta_train_acc = overall_accuracy(meta_model, sampler.full_train_data(device), self.debug)
            meta_test_acc = overall_accuracy(meta_model, sampler.full_val_data(device), self.debug)
            self.info(f"Meta-Model Performance: Train Acc = {meta_train_acc:.1%} | Full Test Acc = {meta_test_acc:.1%}")
            save(meta_model, self.save_path / f"{self.name}-{it}.net", **self.model_args)

    def close(self, it, model):
        save(model, self.save_path / f"{self.name}-{it}.net", **self.model_args)
