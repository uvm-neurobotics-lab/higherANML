import logging

import numpy as np
import torch
from numpy.random import default_rng, SeedSequence
from sklearn.model_selection import train_test_split

from utils import collate_images, unzip


class MetaTrainingSample:
    """
    A structure containing samples for a single outer-loop episode of meta-training:
        - A training trajectory for inner-loop adaptation (a list of batches).
        - All training examples in a single batch (for single-pass evaluation).
        - A set of "remember" examples, for checking performance on examples from other classes.
        - The complete set of meta-training examples, for outer-loop meta-updates.
        - A set of test examples, for checking generalization to never-seen examples from all classes.
        - A sub-sampling of the above test examples, for more frequent evaluation.
    """
    def __init__(self, train_class, train_traj, train_ims, train_labels, rem_ims, rem_labels, meta_ims, meta_labels,
                 val_ims, val_labels, full_test_data):
        """
        Create the data structure.

        Args:
            train_class (int): The index of the class being trained on in this episode.
            train_traj (list): A list of (image, label) tuples. Each is either a single example or a batch.
            train_ims (tensor): All training images.
            train_labels (tensor): All training labels.
            rem_ims (tensor): All remember images.
            rem_labels (tensor): All remember labels.
            meta_ims (tensor): All images to be trained on in the meta-train test phase (outer loop update).
            meta_labels (tensor): All labels to be trained on in the meta-train test phase (outer loop update).
            val_ims (tensor): A sub-sampling of test images to use for estimating generalization performance.
            val_labels (tensor): A sub-sampling of test labels to use for estimating generalization performance.
            full_test_data (list): The full set of test examples. A list of lists of (image, target) tuples, where each
                list represents a class to test on.
        """
        self.train_class = train_class
        self.train_traj = train_traj
        self.train_ims = train_ims
        self.train_labels = train_labels
        self.rem_ims = rem_ims
        self.rem_labels = rem_labels
        self.meta_ims = meta_ims
        self.meta_labels = meta_labels
        self.val_ims = val_ims
        self.val_labels = val_labels
        self.full_test_data = full_test_data


class ContinualMetaLearningSampler:
    """
    A class which takes a pair of (train, test) Datasets and samples them according to the OML train/test procedure
    (Javed & White, 2019).
    """

    def __init__(self, train, test, seed=None):
        """
        Create the sampler.

        Args:
            train (ClassIndexedDataset): The training set.
            test (ClassIndexedDataset): The testing set.
            seed (int): (Optional) Random seed to use for sampling. Otherwise entropy will be pulled from the OS.
        """
        # If seed is None, then we will pull entropy from the OS and log it in case we need to reproduce this run.
        ss = SeedSequence(seed)
        logging.info(f"ContinualMetaLearningSampler is using seed = {ss.entropy}")
        self.rng = default_rng(ss)
        self.train = train
        self.test = test
        self.train_sample_index = np.arange(len(self.train))
        self.test_sample_index = np.arange(len(self.test))

    def num_train_classes(self):
        return len(self.train.class_index)

    def num_test_classes(self):
        return len(self.test.class_index)

    def num_total_classes(self):
        return len(self.train.class_index) + len(self.test.class_index)

    def sample_train(self, batch_size=1, num_batches=20, train_size=500, remember_size=64, val_size=200,
                     add_inner_train_to_outer_train=True, sample_full_test_data=False, device=None):
        """
        Samples a single episode ("outer loop") of the meta-train procedure.

        Args:
            batch_size (int): Number of examples per training batch in the inner loop.
            num_batches (int): Number of training batches in the inner loop.
            train_size (int): Number of examples per class to reserve for all training (remaining examples, if any, will
                form a test set that will never be seen).
            remember_size (int): Number of examples to train (meta-train-test) on in the outer loop.
            val_size (int): Number of never-seen examples to test on in the outer loop (for reporting generalization
                performance).
            add_inner_train_to_outer_train (bool): Whether to add the training examples into the validation set.
            sample_full_test_data (bool): Whether to populate the `full_test_data` field of the output. This could be
                very large, so it is only enabled when needed.
            device (str): The device to send the torch arrays to.

        Returns:
            MetaTrainingSample: All data needed for a single episode of training & logging.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # train set = inner loop training

        # Sample a random class for inner loop training. This gives us a list of example indices for that class.
        # TODO: Change this to use train_test_split(). Deterministic split for now.
        class_indices = self.rng.choice(self.train.class_index)[:train_size]
        sample_size = batch_size * num_batches
        # Only sample with replacement if necessary.
        # TODO: Should we change this to always sample w/ replacement? Would that hurt performance?
        replace = sample_size > len(class_indices)
        sample_indices = self.rng.choice(class_indices, size=sample_size, replace=replace)
        train_samples = [self.train[idx] for idx in sample_indices]
        # Split into batches.
        batched_train_samples = [train_samples[batch_size * i: batch_size * (i + 1)] for i in range(num_batches)]
        # Collate into tensors.
        train_traj = [collate_images(batch, device) for batch in batched_train_samples]
        train_ims, train_labels = collate_images(train_samples, device)  # all training samples together (unbatched)
        train_class = train_samples[0][1]  # just take the first label off the top, since they should all be the same.

        # remember set = outer loop training

        # Sample some number of random instances for meta-training.
        indices = self.rng.choice(self.train_sample_index, size=remember_size, replace=False)
        rem_samples = [self.train[idx] for idx in indices]
        rem_ims, rem_labels = collate_images(rem_samples, device)

        if add_inner_train_to_outer_train:
            # randomly sampled "remember" images + all images from the last training trajectory are concatenated in one
            # single batch of shape [B,C,H,W], where B = train_size + remember_size.
            meta_ims = torch.cat([train_ims, rem_ims])
            meta_labels = torch.cat([train_labels, rem_labels])
        else:
            # just the randomly sampled "remember" images; B = remember_size.
            meta_ims = rem_ims
            meta_labels = rem_labels

        # valid set = for tracking generalization performance to never-seen examples

        # First grab all indices reserved for validation.
        # TODO: Change this to use train_test_split(). Deterministic split for now.
        test_classes = [indices[train_size:] for indices in self.train.class_index]

        # Sub-Sampled Test Data: Sample from the full list and then stack into a single tensor.
        test_indices = [idx for indices in test_classes for idx in indices]
        if test_indices:
            subsampled_indices = self.rng.choice(test_indices, size=val_size)
            val_samples = [self.train[idx] for idx in subsampled_indices]
            val_ims, val_labels = collate_images(val_samples, device)
        else:  # All samples are reserved for training, so we will not perform validation.
            val_ims = torch.tensor([])
            val_labels = torch.tensor([])

        # Full Test Data: Gather all test examples if requested.
        # NOTE: After writing this, I've now realized it's pretty risky because for large datasets it will run OOM. It
        # probably needs a size limit of some kind, like # of examples per class.
        test_data = []
        if sample_full_test_data:
            # Grab all test indices, one list per class.
            test_data = [[self.train[idx] for idx in indices] for indices in test_classes]
            # Turn each list of samples into a separate tensor.
            test_data = [collate_images(batch, device) for batch in test_data]

        return MetaTrainingSample(train_class, train_traj, train_ims, train_labels, rem_ims, rem_labels, meta_ims,
                                  meta_labels, val_ims, val_labels, test_data)

    def sample_test(self, num_classes, train_size=15, test_size=5, device="cuda"):
        """
        Samples the entire continual learning trajectory for the meta-test procedure.

        Args:
            num_classes (int): Number of classes to include (number of "learning episodes" a.k.a. "outer loops").
            train_size (int): Number of examples to train on in each inner loop.
            test_size (int): Number of examples to test on from the classes that we learn.
            device (str): The device to send the torch arrays to.

        Returns:
            train_episodes (list): A list of lists of (image, target) tuples, where each list represents a new class to
                be learned.
            full_test_data (list): A list of lists of (image, target) tuples, where each list represents a class to test on.
        """
        if num_classes > len(self.test.class_index):
            raise ValueError(f"Number of classes requested is too large: {num_classes} > {len(self.test.class_index)}")

        # choose the n classes to use
        class_sets = self.rng.choice(self.test.class_index, size=num_classes, replace=False)

        # Split each group of task IDs into train and test; unzip to separate train and test sequences.
        # Now we have two lists, each with num_classes items. Each item in train_classes is a list of train_size IDs,
        # while each item in test_classes is a list of test_size IDs.
        train_classes, test_classes = unzip(
            train_test_split(indices, train_size=train_size, test_size=test_size, shuffle=True)
            for indices in class_sets
        )

        # Assemble the train/test trajectories; a list of classes, each of which is a list of (sample, target) tuples.
        train_traj = [[self.test[idx] for idx in train_task] for train_task in train_classes]
        test_traj = [[self.test[idx] for idx in test_task] for test_task in test_classes]

        # Now batch each class into a single tensor.
        train_data = [collate_images(samples, device) for samples in train_traj]
        test_data = [collate_images(samples, device) for samples in test_traj]

        return train_data, test_data
