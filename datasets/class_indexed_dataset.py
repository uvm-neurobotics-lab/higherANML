"""
Abstract class for datasets that also support indexing into the class labels.
"""
from abc import abstractmethod

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils import unzip


class ClassIndexedDataset(Dataset):
    """
    A dataset that also provides methods to index into the list of labels, and fetch examples per label.
    """

    @property
    @abstractmethod
    def class_index(self):
        """
        Get the index of all label classes in this dataset.
        Returns:
            list: A list of lists of indices. Each element in this list represents one class in the dataset. Each
            element is a list of indices, which are the indices of all samples that correspond to that class.
        """
        raise NotImplementedError()


def train_val_split(dataset, train_size=None, val_size=None):
    """
    Given a ClassIndexedDataset, produce per-class train/validation splits.

    This is mostly a wrapper around `sklearn.model_selection.train_test_split()`, except it defaults to using the full
    dataset as "train", and allows the user to explicitly request 100% for the training share (which is not allowed by
    the `sklearn` method).

    NOTE: Assumes all classes have the same number of examples.

    Args:
        dataset (ClassIndexedDataset): The training set.
        train_size (int or float): Number or fraction of examples per class to reserve for all training. If `None`, this
            will be set to the complement of `val_size`. If both are `None`, all examples will be used for training and
            the validation set will be empty.
        val_size (int or float): Number or fraction of examples per class to reserve for the validation set. If `None`,
            this will be set to the complement of `train_size`.

    Returns:
        train_class_index (list): A list of lists, where each element `i` is a list of indices of training examples for
            class `i`.
        val_class_index (list): A list of lists, where each element `i` is a list of indices of validation examples for
            class `i`. This list can be empty, or could be a list of empty lists.
    """
    train_class_index = None
    val_class_index = None

    if not train_size and not val_size:
        # No split specified. Use all training data.
        train_class_index = dataset.class_index
        val_class_index = [[]] * len(dataset.class_index)
    else:
        if train_size and not val_size:
            # Test if we are trying to use 100% of training data. `train_test_split` doesn't allow this,
            # so we need to do it manually.
            # NOTE: This line assumes all classes have the same number of examples.
            if (train_size >= len(dataset.class_index[0]) or
                    (isinstance(train_size, float) and train_size == 1.0)):
                train_class_index = dataset.class_index
                val_class_index = [[]] * len(dataset.class_index)

    # If still not defined, then we should be using the split function.
    if not train_class_index:
        train_class_index, val_class_index = unzip(
            train_test_split(indices, train_size=train_size, test_size=val_size, shuffle=True)
            for indices in dataset.class_index
        )

    return train_class_index, val_class_index
