"""
Abstract class for datasets that also support indexing into the class labels.
"""
from abc import abstractmethod

from torch.utils.data import Dataset


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
