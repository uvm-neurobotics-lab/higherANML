"""
Utilities for logging progress metrics and saving checkpoints.
"""

import logging
from collections import Counter
from pathlib import Path
from time import time, strftime, gmtime

import numpy as np
import scipy
from torch.nn.functional import cross_entropy

from utils.storage import save


def accuracy(preds, labels):
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


def extract_train(all_samples, num_train_ex):
    # NOTE: Assumes that all training examples are first in the batch.
    return all_samples[:num_train_ex]


def extract_remember(all_samples, num_train_ex):
    # NOTE: Assumes that all training examples are first in the batch.
    return all_samples[num_train_ex:]


def train_val_split(data, num_train_ex):
    return extract_train(data, num_train_ex), extract_remember(data, num_train_ex)


def print_validation_stats(output, labels, loss, num_train_ex, train_class, verbose, print_fn):
    # Separate out the train examples from the "remember" examples.
    train_out, val_out = train_val_split(output, num_train_ex)
    train_labels, val_labels = train_val_split(labels, num_train_ex)

    # Loss & Accuracy
    train_acc = accuracy(train_out, train_labels)
    val_acc = accuracy(val_out, val_labels)
    print_fn(f"Meta-Loss = {loss:.5f}")
    print_fn(f"Train Acc = {train_acc:.1%}, Remember Acc = {val_acc:.1%}")
    # TODO: report accuracy of non-train classes instead of remember set?
    # TODO: report how much of each set was "seen" before?
    # TODO: or separately report total fraction of classes seen and examples seen
    # TODO: report overall accuracy on "things seen so far"? Maybe explicitly sample from things seen.

    # "Entropy"
    train_spread = normalized_spread(train_out)
    val_spread = normalized_spread(val_out)
    print_fn(f"Train Spread = {train_spread:.2f}, Remember Spread = {val_spread:.2f}")

    # Top Classes
    # TODO: Start tracking seen and unseen classes here?
    # TODO: Also in outer loop printout.
    train_mode = classes_sorted_by_frequency(train_out)[0]
    train_mode_freq = train_mode[1] / len(train_out)
    print_fn(f"Most frequent train prediction: {train_mode[0]} ({train_mode_freq:.1%} of predictions)")
    val_mode = classes_sorted_by_frequency(val_out)[0]
    val_mode_freq = val_mode[1] / len(val_out)
    print_fn(f"Most frequent remember prediction: {val_mode[0]} ({val_mode_freq:.1%} of predictions)")

    # Print percentage of over-prediction of target class. (How many "remember" items were wrong b/c they were
    # predicted as the class currently being learned.)
    print_fn(f"Portion of remember wrongly predicted as {train_class} = "
             f"{fraction_wrong_predicted_as_train_class(val_out, val_labels, train_class):.1%}")

    # If super verbose, print the entire prediction.
    if verbose >= 3:
        pred_label_pairs = np.array(list(zip(output.argmax(axis=1), labels)))
        print_fn("\n" + str(pred_label_pairs))


def forward_pass(model, ims, labels):
    out = model(ims)
    loss = cross_entropy(out, labels)
    acc = accuracy(out, labels)
    return out, loss, acc


class Log:
    def __init__(self, name, model_args, print_freq=10, verbose_freq=None, save_freq=1000):
        self.start = -1
        self.name = name
        self.model_args = model_args
        self.print_freq = print_freq
        self.verbose_freq = verbose_freq
        self.save_freq = save_freq
        self.logger = logging.getLogger(name)
        Path("./trained_anmls").mkdir(exist_ok=True)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def outer_begin(self):
        if self.start < 0:
            self.start = time()

    def outer_info(self, it, train_class):
        if it % self.print_freq == 0:
            self.info(f"**** Outer loop {it}: Learning on class {train_class}...")

    def inner(self, outer_it, inner_it, train_class, inner_loss, inner_acc, valid_ims, valid_labels, num_train_ex,
              model, verbose):
        # Only print inner loop info when verbose is turned on.
        if (self.verbose_freq > 0) and (outer_it % self.verbose_freq == 0):
            m_out, m_loss, m_acc = forward_pass(model, valid_ims, valid_labels)
            if inner_it < 2:
                # TODO: plot change in performance b/w first and second iteration.
                # TODO: plot change in performance b/w first and last iteration.
                # TODO: report when/if train acc reaches 100% - some idea of how fast learning is happening
                # TODO: and/or plot some idea of how much outputs are changing, in general
                self.debug(f"  Inner iter {inner_it}: Loss = {inner_loss:.5f}, Acc = {inner_acc:.1%}")
                print_validation_stats(m_out, valid_labels, m_loss, num_train_ex, train_class, verbose,
                                       lambda msg: self.debug("    " + msg))

    def outer_end(self, it, train_class, out, loss, acc, valid_ims, valid_labels, num_train_ex, meta_model, verbose):
        if it % self.print_freq == 0:
            end = time()
            elapsed = end - self.start
            self.start = -1

            train_acc = accuracy(extract_train(out, num_train_ex), extract_train(valid_labels, num_train_ex))
            rem_acc = accuracy(extract_remember(out, num_train_ex), extract_remember(valid_labels, num_train_ex))
            self.info(f"  Final Meta-Loss = {loss.item():.3f} | Meta-Acc = {acc:.1%} | Train Acc = {train_acc:.1%}"
                      f" | Remember Acc = {rem_acc:.1%} ({strftime('%H:%M:%S', gmtime(elapsed))})")

        # If verbose, then also evaluate the new meta-model on the previous train/validation data so we can see the
        # impact of meta-learning.
        if (self.verbose_freq > 0) and (it % self.verbose_freq == 0):
            # TODO: Idea: report difference b/w meta-model and end-model perf.
            self.debug("  End Model Performance:")
            print_validation_stats(out, valid_labels, loss, num_train_ex, train_class, verbose,
                                   lambda msg: self.debug("    " + msg))

            m_out, m_loss, m_acc = forward_pass(meta_model, valid_ims, valid_labels)
            self.debug("  Meta-Model Performance:")
            print_validation_stats(m_out, valid_labels, m_loss, num_train_ex, train_class, verbose,
                                   lambda msg: self.debug("    " + msg))

        if it % self.save_freq == 0:
            save(meta_model, f"trained_anmls/{self.name}-{it}.net", **self.model_args)

    def close(self, it, model):
        save(model, f"trained_anmls/{self.name}-{it}.net", **self.model_args)
