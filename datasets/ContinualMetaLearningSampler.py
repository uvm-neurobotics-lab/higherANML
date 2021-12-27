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
        - A set of validation examples, for checking performance on examples from other classes.
        - The complete set of meta-training examples, for outer-loop meta-updates.
    """
    def __init__(self, train_class, train_traj, train_ims, train_labels, valid_ims, valid_labels, meta_ims,
                 meta_labels):
        """
        Create the data structure.

        Args:
            train_class (int): The index of the class being trained on in this episode.
            train_traj (list): A list of (image, label) tuples. Each is either a single example or a batch.
            train_ims (tensor): All training images.
            train_labels (tensor): All training labels.
            valid_ims (tensor): All validation images.
            valid_labels (tensor): All validation labels.
            meta_ims (tensor): All images to be trained on in the meta-train test phase (outer loop update).
            meta_labels (tensor): All labels to be trained on in the meta-train test phase (outer loop update).
        """
        self.train_class = train_class
        self.train_traj = train_traj
        self.train_ims = train_ims
        self.train_labels = train_labels
        self.valid_ims = valid_ims
        self.valid_labels = valid_labels
        self.meta_ims = meta_ims
        self.meta_labels = meta_labels


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

    def sample_train(self, batch_size=1, num_batches=20, remember_size=64, add_inner_train_to_outer_train=True,
                     device=None):
        """
        Samples a single episode ("outer loop") of the meta-train procedure.

        Args:
            batch_size (int): Number of examples per training batch in the inner loop.
            num_batches (int): Number of training batches in the inner loop.
            remember_size (int): Number of examples to test (meta-train) on in the outer loop.
            add_inner_train_to_outer_train (bool): Whether to add the training examples into the validation set.
            device (str): The device to send the torch arrays to.

        Returns:
            MetaTrainingSample: All data needed for a single episode of training & logging.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Sample a random class for inner loop training. This gives us a list of example indices for that class.
        class_indices = self.rng.choice(self.train.class_index)
        sample_indices = self.rng.choice(class_indices, size=batch_size * num_batches, replace=False)
        train_samples = [self.train[idx] for idx in sample_indices]
        # Split into batches.
        batched_train_samples = [train_samples[batch_size * i: batch_size * (i + 1)] for i in range(num_batches)]
        # Collate into tensors.
        train_traj = [collate_images(batch, device) for batch in batched_train_samples]
        train_ims, train_labels = collate_images(train_samples, device)  # all training samples together (unbatched)
        train_class = train_samples[0][1]  # just take the first label off the top, since they should all be the same.

        # valid = outer loop training
        # Sample some number of random instances for meta-training.
        indices = self.rng.choice(self.train_sample_index, size=remember_size, replace=False)
        valid_samples = [self.train[idx] for idx in indices]
        valid_ims, valid_labels = collate_images(valid_samples, device)

        if add_inner_train_to_outer_train:
            # randomly sampled "remember" images + all images from the last training trajectory are concatenated in one
            # single batch of shape [B,C,H,W], where B = train_size + remember_size.
            meta_ims = torch.cat([train_ims, valid_ims])
            meta_labels = torch.cat([train_labels, valid_labels])
        else:
            # just the randomly sampled "remember" images; B = remember_size.
            meta_ims = valid_ims
            meta_labels = valid_labels

        return MetaTrainingSample(train_class, train_traj, train_ims, train_labels, valid_ims, valid_labels, meta_ims,
                                  meta_labels)

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
            test_data (list): A list of lists of (image, target) tuples, where each list represents a class to test on.
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
