"""
Sampling for standard i.i.d. training.
"""
import logging

import numpy as np
from numpy.random import default_rng, SeedSequence
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from .class_indexed_dataset import train_val_split
from utils import collate_images, unzip


def possibly_empty_loader(dataset, batch_size):
    if len(dataset) < 1:
        return []
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class IIDSampler:
    """
    A class which uses standard torch.utils.data.DataLoaders to sample randomly from a pair of (train, val) datasets.
    """

    def __init__(self, train, test, batch_size=128, train_size=None, val_size=None, seed=None):
        """
        Create the sampler.

        Args:
            train (ClassIndexedDataset): The training set.
            test (ClassIndexedDataset): The testing set.
            batch_size (int): Number of images per batch for both datasets.
            train_size (int or float): Number or fraction of examples per class to reserve for all training. If
                `None`, this will be set to the complement of `val_size`. If both are `None`, all examples will be used
                for training and the validation set will be empty.
            val_size (int or float): Number or fraction of examples per class to reserve for the validation set. If
                `None`, this will be set to the complement of `train_size`.
            seed (int): (Optional) Random seed to use for sampling. Otherwise entropy will be pulled from the OS.
        """
        self.train = train
        self.test = test
        self.batch_size = batch_size
        # If seed is None, then we will pull entropy from the OS and log it in case we need to reproduce this run.
        ss = SeedSequence(seed)
        logging.info(f"IIDSampler is using seed = {ss.entropy}")
        self.rng = default_rng(ss)

        # Select which examples will form the training/validation splits.
        self.train_class_index, self.val_class_index = train_val_split(self.train, train_size, val_size)

        # Get a flattened list of all samples from each category, for convenient sampling.
        self.train_sample_index = [idx for indices in self.train_class_index for idx in indices]
        self.val_sample_index = [idx for indices in self.val_class_index for idx in indices]
        self.test_sample_index = np.arange(len(self.test))

        # Finally, split the training dataset into (train, val) subsets.
        self.train_orig = self.train
        self.train = Subset(self.train_orig, self.train_sample_index)
        self.val = Subset(self.train_orig, self.val_sample_index)

        # Create data loaders for all.
        self.train_loader = possibly_empty_loader(self.train, batch_size)
        self.val_loader = possibly_empty_loader(self.val, batch_size)
        self.test_loader = possibly_empty_loader(self.test, batch_size)

    def num_train_classes(self):
        return len(self.train_orig.class_index)

    def num_test_classes(self):
        return len(self.test.class_index)

    def num_total_classes(self):
        return len(self.train_orig.class_index) + len(self.test.class_index)

    def full_train_data(self, device=None):
        """
        Return an object which can be used to iterate through the training set, in batches (such as a DataLoader).
        This creates a new iterable each time, so the iterables it generates can be used concurrently without
        interfering with each other.

        Args:
            device: This is unused, but kept for compatibility with `ContinualMetaLearningSampler`.

        Returns:
            DataLoader: An iterable object that will lazily load each batch as requested.
        """
        return possibly_empty_loader(self.train, self.batch_size)

    def full_val_data(self, device=None):
        """
        Return an object which can be used to iterate through the validation set, in batches (such as a DataLoader).
        This creates a new iterable each time, so the iterables it generates can be used concurrently without
        interfering with each other. Note that the validation set may be empty if we are using all the available
        data for training, in which case this returns an empty iterator.

        Args:
            device: This is unused, but kept for compatibility with `ContinualMetaLearningSampler`.

        Returns:
            DataLoader: An iterable object that will lazily load each batch as requested.
        """
        return possibly_empty_loader(self.val, self.batch_size)

    def get_biased_sample(self, dataset, class_index, sample_index, classes, fraction_biased):
        """
        Get a batch of samples which is biased toward the given `classes`. A fraction (`fraction_biased`) of the
        samples will be from the given set of classes, while the rest of the samples will be drawn from the full
        dataset.

        Args:
            dataset (ClassIndexedDataset): The dataset to sample from. E.g. `self.train`.
            class_index (list): A list of lists, where each element `i` is a list of indices of examples for class `i`.
                The indices must correspond to the given `dataset`. E.g. `self.train_class_index`.
            sample_index (list): A list of indices of examples from all classes. The indices must correspond to the
                given `dataset`. E.g. `self.train_sample_index`.
            classes (list): The indices of the classes to bias toward.
            fraction_biased (float): A fraction [0.0, 1.0] of the total `self.batch_size` samples which will be
                restricted to only the given classes.

        Returns:
            Tensor: The images sampled.
            Tensor: The labels sampled.
        """
        num_biased = int(self.batch_size * fraction_biased)
        num_unbiased = self.batch_size - num_biased
        # Get a flattened list of all the desired classes. The "if" clause is to handle the case where the network
        # output layer actually contains more classes than the dataset. In that case, some of the class indices may not
        # actually exist, so we skip them.
        desired_samples = [indices for cls in classes if cls < len(class_index) for indices in class_index[cls]]
        if not desired_samples:
            # This could happen if the network output layer actually contains more classes than the dataset itself. We
            # might actually end up with no real classes. In that case, just take a normal unbiased sample.
            num_biased = 0
            num_unbiased = self.batch_size
        # Sample `num_biased` only from this set of classes. Only sample with replacement if necessary.
        biased_indices = self.rng.choice(desired_samples, size=num_biased, replace=(num_biased > len(desired_samples)))
        # Now get the rest of the sample from all classes.
        unbiased_indices = self.rng.choice(sample_index, size=num_unbiased, replace=(num_unbiased > len(sample_index)))
        # Return the actual data.
        samples = [dataset[idx] for idx in np.concatenate((biased_indices, unbiased_indices)).astype(int)]
        return collate_images(samples)

    def get_biased_train_sample(self, classes, fraction_biased):
        """
        Get a batch of samples from the training set which is biased toward the given `classes`. See
        `get_biased_sample()` for full documentation.

        Args:
            classes (list): The indices of the classes to bias toward.
            fraction_biased (float): A fraction [0.0, 1.0] of the total `self.batch_size` samples which will be
                restricted to only the given classes.

        Returns:
            Tensor: The images sampled.
            Tensor: The labels sampled.
        """
        return self.get_biased_sample(self.train_orig, self.train_class_index, self.train_sample_index, classes,
                                      fraction_biased)

    def sample_support_and_query_sets(self, num_classes, support_size, query_size):
        """
        Sample a transfer learning problem.

        This produces a support set (for adaptation) and a query set (for testing) for a transfer learning problem.
        These sets are _subsets_ of the test set. We may use the support examples in any way to adapt the model to the
        new task, and then evaluation is performed using the query examples.

        Args:
            num_classes (int): Number of classes to use. This must be less than or equal to the total number of classes
                in the test set. This number of classes will be chosen at random to form an N-way classification.
            support_size (int or float): The number of examples per class, or fraction of total instances, to use for
                the support set.
            query_size (int or float): The number of examples per class, or fraction of total instances, to use for the
                query set.

        Returns:
            torch.utils.data.Dataset: The support set.
            torch.utils.data.Dataset: The query set.
        """
        if num_classes > len(self.test.class_index):
            raise ValueError(f"Number of classes requested is too large: {num_classes} > {len(self.test.class_index)}")

        # choose the n classes to use
        class_sets = self.rng.choice(self.test.class_index, size=num_classes, replace=False)

        # Split examples from each class into support and query.
        # NOTE: This line assumes all classes have the same number of examples.
        num_available = class_sets[0]
        if not support_size or support_size >= len(num_available):
            # Use all training data.
            raise ValueError(f"Support size is too large. Requested {support_size} but we only have {num_available},"
                             " leaving no room for a query set.")
        else:
            support_class_index, query_class_index = unzip(
                train_test_split(indices, train_size=support_size, test_size=query_size, shuffle=True)
                for indices in class_sets
            )
        # Get a flattened list of all samples from each category, for convenient sampling.
        support_sample_index = [idx for indices in support_class_index for idx in indices]
        query_sample_index = [idx for indices in query_class_index for idx in indices]

        # Finally, split the test dataset into (support, query) subsets.
        support_set = Subset(self.test, support_sample_index)
        query_set = Subset(self.test, query_sample_index)
        return support_set, query_set
