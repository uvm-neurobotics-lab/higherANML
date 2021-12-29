import logging

import numpy as np
import torch
from numpy.random import default_rng, SeedSequence
from sklearn.model_selection import train_test_split

from utils import collate_images, divide_chunks, unzip


# Heuristic: Based on experimentation, it seems like this is a reasonable max batch size for most of our current models
# and datasets to avoid running out of memory on the VACC.
_MAX_BATCH_SIZE = 200


class BatchList:
    """
    An object which stores a list of training batches, but where the batches are not stored in memory. Every time an
    item is requested it is loaded from disk.
    """
    def __init__(self, dataset, batches, device=None):
        """
        Create the batch list.
        Args:
            dataset (Dataset): The (random-access) dataset to load samples from.
            batches (list): A list of lists of ints, where each list contains the indices for a single batch.
            device (str): The device where the torch arrays should be stored.
        """
        self.dataset = dataset
        self.batches = batches
        self.device = device

    def __len__(self):
        """
        Returns:
            int: total number of batches
        """
        return len(self.batches)

    def __getitem__(self, index):
        """
        Loads batch corresponding to the given index.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where each is a tensor containing the full batch.
        """
        # Load all samples for this batch.
        samples = [self.dataset[i] for i in self.batches[index]]
        # Stack the samples into two tensors: (images, labels).
        return collate_images(samples, self.device)


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
                 val_ims, val_labels):
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


class ContinualMetaLearningSampler:
    """
    A class which takes a pair of (train, test) Datasets and samples them according to the OML train/test procedure
    (Javed & White, 2019).
    """

    def __init__(self, train, test, seed=None, train_size=None):
        """
        Create the sampler.

        Args:
            train (ClassIndexedDataset): The training set.
            test (ClassIndexedDataset): The testing set.
            seed (int): (Optional) Random seed to use for sampling. Otherwise entropy will be pulled from the OS.
            train_size (int or float): Number or fraction of examples per class to reserve for all training. The
                remaining examples, if any, will form a training validation set that will never be seen.
        """
        # If seed is None, then we will pull entropy from the OS and log it in case we need to reproduce this run.
        ss = SeedSequence(seed)
        logging.info(f"ContinualMetaLearningSampler is using seed = {ss.entropy}")
        self.rng = default_rng(ss)
        self.train = train
        self.test = test
        # Select which examples will form the training/validation splits.
        if not train_size or train_size >= len(self.train):
            # Use all training data.
            self.train_class_index = self.train.class_index
            self.val_class_index = []
        else:
            self.train_class_index, self.val_class_index = unzip(
                train_test_split(indices, train_size=train_size, shuffle=True)
                for indices in self.train.class_index
            )
        # Get a flattened list of all samples from each category, for convenient sampling.
        self.train_sample_index = [idx for indices in self.train_class_index for idx in indices]
        self.val_sample_index = [idx for indices in self.val_class_index for idx in indices]
        self.test_sample_index = np.arange(len(self.test))

    def num_train_classes(self):
        return len(self.train.class_index)

    def num_test_classes(self):
        return len(self.test.class_index)

    def num_total_classes(self):
        return len(self.train.class_index) + len(self.test.class_index)

    def full_train_data(self, device=None):
        """
        Return all training data, chunked up into batches of reasonable size, so as not to run out of memory on the GPU.
        Args:
            device: The device where batches should be instantiated.
        Returns:
            BatchList: A random-access object that will lazily load each batch as requested.
        """
        batches = list(divide_chunks(self.train_sample_index, _MAX_BATCH_SIZE))
        return BatchList(self.train, batches, device)

    def full_val_data(self, device=None):
        """
        Return all validation data, chunked up into batches of reasonable size, so as not to run out of memory on the
        GPU.
        Args:
            device: The device where batches should be instantiated.
        Returns:
            BatchList: A random-access object that will lazily load each batch as requested.
        """
        batches = list(divide_chunks(self.val_sample_index, _MAX_BATCH_SIZE))
        return BatchList(self.train, batches, device)

    def sample_train(self, batch_size=1, num_batches=20, remember_size=64, val_size=200,
                     add_inner_train_to_outer_train=True, device=None):
        """
        Samples a single episode ("outer loop") of the meta-train procedure.

        Args:
            batch_size (int): Number of examples per training batch in the inner loop.
            num_batches (int): Number of training batches in the inner loop.
            remember_size (int): Number of examples to train (meta-train-test) on in the outer loop.
            val_size (int): Number of never-seen examples to test on in the outer loop (for reporting generalization
                performance).
            add_inner_train_to_outer_train (bool): Whether to add the training examples into the validation set.
            device (str): The device to send the torch arrays to.

        Returns:
            MetaTrainingSample: All data needed for a single episode of training & logging.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # train set = inner loop training

        # Sample a random class for inner loop training. This gives us a list of example indices for that class.
        class_indices = self.rng.choice(self.train_class_index)
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

        # Sample from the full list and then stack into a single tensor.
        if self.val_sample_index:
            subsampled_indices = self.rng.choice(self.val_sample_index, size=val_size)
            val_samples = [self.train[idx] for idx in subsampled_indices]
            val_ims, val_labels = collate_images(val_samples, device)
        else:  # All samples are reserved for training, so we will not perform validation.
            val_ims = torch.tensor([])
            val_labels = torch.tensor([])

        return MetaTrainingSample(train_class, train_traj, train_ims, train_labels, rem_ims, rem_labels, meta_ims,
                                  meta_labels, val_ims, val_labels)

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

        # TODO: The below could be returned as BatchLists and be less susceptible to OOM.
        # Assemble the train/test trajectories; a list of classes, each of which is a list of (sample, target) tuples.
        train_traj = [[self.test[idx] for idx in train_task] for train_task in train_classes]
        test_traj = [[self.test[idx] for idx in test_task] for test_task in test_classes]

        # Now batch each class into a single tensor.
        train_data = [collate_images(samples, device) for samples in train_traj]
        test_data = [collate_images(samples, device) for samples in test_traj]

        return train_data, test_data
