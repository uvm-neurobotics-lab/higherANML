import logging

import numpy as np
import torch
from numpy.random import default_rng, SeedSequence
from sklearn.model_selection import train_test_split

from utils import unzip, divide_chunks


def collate_fn(samples):
    # takes a list of (im,label) and returns [ims],[labels]
    xs = torch.stack([sample[0] for sample in samples])
    ys = torch.tensor([sample[1] for sample in samples])
    return xs, ys


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

    def sample_train(self, train_size=20, remember_size=64, device="cuda"):
        """
        Samples a single episode ("outer loop") of the meta-train procedure.

        Args:
            train_size (int): Number of examples to train on in the inner loop.
            remember_size (int): Number of examples to test (meta-train) on in the outer loop.
            device (str): The device to send the torch arrays to.

        Returns:
            train_data (list): A list of (image, label) tuples.
            train_class (int): The index of the class being trained on in this episode.
            tuple: A pair of torch arrays for the meta-train test phase (valid_ims, valid_labels).
        """
        # Sample a random class for inner loop training. This gives us a list of example indices for that class.
        class_indices = self.rng.choice(self.train.class_index)
        sample_indices = self.rng.choice(class_indices, size=train_size, replace=False)
        samples = [self.train[idx] for idx in sample_indices]
        train_ims, train_labels = collate_fn(samples)

        # Sample some number of random instances for meta-training.
        indices = self.rng.choice(self.train_sample_index, size=remember_size, replace=False)
        samples = [self.train[idx] for idx in indices]
        valid_ims, valid_labels = collate_fn(samples)

        # valid = outer loop training
        # randomly sampled "remember" images + all images from the last training trajectory are concatenated in one
        # single batch of shape [B,C,H,W], where B = train_size + remember_size.
        valid_ims = torch.cat([train_ims, valid_ims]).squeeze(1).to(device)
        valid_labels = torch.cat([train_labels, valid_labels]).to(device)

        # train = inner loop training
        train_labels = train_labels.unsqueeze(1).to(device)
        # training images are processed one at a time so we zip them together with labels
        train_data = list(zip(train_ims.to(device), train_labels))

        train_class = train_labels[0][0].cpu().item()

        return (
            train_data,
            train_class,
            (valid_ims, valid_labels),
        )

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
            test_data (tuple): A tuple of (images, labels) torch arrays, for evaluation in one large batch.
        """
        assert num_classes <= len(self.test.class_index), (
            f"Number of classes requested is too large: {num_classes} > {len(self.test.class_index)}")

        # choose the n classes to use
        class_sets = self.rng.choice(self.test.class_index, size=num_classes, replace=False)

        # Split each group of task IDs into train and test; unzip to separate train and test sequences.
        # Now we have two lists, each with num_classes items. Each item in train_classes is a list of train_size IDs,
        # while each item in test_classes is a list of test_size IDs.
        train_classes, test_classes = unzip(
            train_test_split(indices, train_size=train_size, test_size=test_size, shuffle=True)
            for indices in class_sets
        )

        # assemble the train/test trajectories, one long list of (sample, target) tuples
        train_traj = [self.test[idx] for train_task in train_classes for idx in train_task]
        test_traj = [self.test[idx] for test_task in test_classes for idx in test_task]

        # test-train examples are divided by task and sent to device (cpu/cuda)
        def chunk2device(chunk):
            return [(im.to(device), label.to(device)) for im, label in chunk]

        train_episodes = [chunk2device(chunk) for chunk in divide_chunks(train_traj, n=train_size)]

        # test-test classes are collected into a massive tensor for one-pass evaluation
        ims, labels = list(zip(*test_traj))
        test_data = (torch.cat(ims).to(device), torch.cat(labels).to(device))

        return train_episodes, test_data