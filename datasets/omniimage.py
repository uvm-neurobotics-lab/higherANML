"""
PyTorch Dataset class for OmniImage.
"""

from pathlib import Path

import torch
from omnimage.dataset import get_test_classes, OmnImageDataset
from torch.utils.data import Subset
from torchvision.transforms import Compose, Lambda, RandomResizedCrop, RandomHorizontalFlip, Resize

from .class_indexed_dataset import ClassIndexedDataset
from .ContinualMetaLearningSampler import ContinualMetaLearningSampler
from .IIDSampler import IIDSampler


class OmniImage(ClassIndexedDataset):
    """
    Mini-ImageNet dataset, as given by [OmniImage](https://github.com/lfrati/OmnImage).
    If this download ever stops working, those tools can be used to regenerate the dataset. You can also generate
    custom, modified versions of the dataset.
    """

    def __init__(
            self,
            root,
            num_images_per_class,
            split,
            transform=None,
            target_transform=None,
            greyscale=False,
            normalize=True,
            download=False,
    ):
        """
        Create the dataset.

        Args:
            root (string): Desired root directory for the dataset (usually named `mini-imagenet/`, one directory above
                the `processed_images/` folder).
            split (str): Which split of the dataset to use ("train" or "test").
            transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
                version. E.g, ``transforms.RandomCrop``.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            greyscale (bool, optional): If True, converts images to greyscale when sampling.
            download (bool, optional): If true, downloads the dataset zip files from the internet and
                puts it in root directory. If the zip files are already downloaded, they are not
                downloaded again.
        """
        self.dataset = OmnImageDataset(root, num_images_per_class, greyscale=greyscale, normalize=normalize,
                                       download=download)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Narrow the full dataset down to just the split we're interested in.
        test_classes = get_test_classes()
        is_test_split = (split.lower() == "test")

        def is_desired_class(clsname):
            is_test = clsname in test_classes
            # True when this class' split matches the one we want (train or test).
            return is_test == is_test_split

        # NOTE: This expects we will iterate in order of increasing index.
        self._from_subclass_index_mapping = [clsdex for clsname, clsdex in self.dataset.class_to_idx.items()
                                             if is_desired_class(clsname)]
        self._to_subclass_index_mapping = {clsdex: subdex
                                           for subdex, clsdex in enumerate(self._from_subclass_index_mapping)}
        # self._class_index is a list of lists of the kind specified by `ClassIndexedDataset.class_index()`.
        self._class_index = [[] for _ in self._from_subclass_index_mapping]
        self.sub_indices = []
        currdex = 0
        for index, (clsdex, clsname) in enumerate(zip(self.dataset.labels, self.dataset.classes)):
            if is_desired_class(clsname):
                self._class_index[self._to_subclass_index_mapping[clsdex]].append(currdex)
                self.sub_indices.append(index)
                currdex += 1
        self.subset = Subset(self.dataset, self.sub_indices)

    @property
    def class_index(self):
        return self._class_index

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        """
        Fetches and transforms the example corresponding to the given index.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the index of the target class.
        """
        image, label = self.subset[index]
        label = torch.tensor(self._to_subclass_index_mapping[label.item()])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def create_datasets(root, download=True, num_images_per_class=None, im_size=None, greyscale=False, augment=False):
    """
    Create a pair of (train, test) datasets for OmniImage.

    Args:
        root (str or Path): Folder where the dataset will be located.
        download (bool): If True, download the data if it doesn't already exist. If False, raise an error.
        num_images_per_class (int): Number of images per class.
        im_size (int): Desired size of images, or None to use the on-disk sizes.
        greyscale (bool): Whether to convert images to greyscale; False or None to keep the default coloring.
        augment (bool): Whether to apply data augmentation to the training set.

    Returns:
        OmniImage: The training set.
        OmniImage: The testing set.
        tuple: The shape of the images that will be returned by the sampler (they will all be the same size).
    """
    # We know that by default this dataset has 84x84 color images.
    actual_size = 84 if im_size is None else im_size
    # By default we use 20 images per class.
    if num_images_per_class is None:
        num_images_per_class = 20

    # Build train set.
    train_transforms = []
    if augment:
        # NOTE: This will also augment the "validation" set. It would be better if we didn't do this, but I don't have
        # the energy for the necessary refactor right now.
        train_transforms.append(RandomResizedCrop(actual_size))
        train_transforms.append(RandomHorizontalFlip())
    else:
        # Only resize if a specific size is requested.
        if im_size is not None:
            # Antialias for compatibility with our Mini-ImageNet dataset (uses PIL images when resizing, which
            # automatically anti-alias). This could be removed in the future if we refactor MiniImageNet.
            train_transforms.append(Resize(im_size, antialias=True))
    train_transforms = Compose(train_transforms)
    t_transforms = Lambda(lambda x: torch.tensor(x))
    train = OmniImage(
        root=root,
        num_images_per_class=num_images_per_class,
        split="train",
        transform=train_transforms,
        target_transform=t_transforms,
        greyscale=greyscale,
        download=download,
    )

    # Build test set.
    test_transforms = []
    # Only resize if a specific size is requested.
    if im_size is not None:
        # Antialias for compatibility with our Mini-ImageNet dataset (uses PIL images when resizing, which
        # automatically anti-alias). This could be removed in the future if we refactor MiniImageNet.
        test_transforms.append(Resize(im_size, antialias=True))
    test_transforms = Compose(test_transforms)
    test = OmniImage(
        root=root,
        num_images_per_class=num_images_per_class,
        split="test",
        transform=test_transforms,
        target_transform=t_transforms,
        greyscale=greyscale,
        download=download,
    )

    image_shape = (1 if greyscale else 3, actual_size, actual_size)
    return train, test, image_shape


def create_iid_sampler(root, download=True, num_images_per_class=None, im_size=None, greyscale=False, batch_size=128,
                       train_size=None, val_size=None, augment=False):
    """
    Create a sampler for OmniImage data which will sample shuffled batches in the standard way for i.i.d. training.

    Args:
        root (str or Path): Folder where the dataset will be located.
        download (bool): If True, download the data if it doesn't already exist. If False, raise an error.
        num_images_per_class (int): Number of images per class (only certain values are available---check the
                                    OmniImage library documentation).
        im_size (int): Desired size of images, or None to use the on-disk sizes.
        greyscale (bool): Whether to convert images to greyscale; False or None to keep the default coloring.
        batch_size (int): Number of images per batch for both datasets.
        train_size (int or float): Number (or fraction) of samples from the train set to actually use for training.
        val_size (int or float): Number (or fraction) of samples from the train set to use for validation.
        augment (bool): Whether to apply data augmentation to the training set.

    Returns:
        IIDSampler: The sampler class.
        tuple: The shape of the images that will be returned by the sampler (they will all be the same size).
    """
    train, test, image_shape = create_datasets(root, download, num_images_per_class, im_size, greyscale, augment)
    return IIDSampler(train, test, batch_size, train_size, val_size), image_shape


def create_OML_sampler(root, download=True, num_images_per_class=None, im_size=None, greyscale=False, train_size=None,
                       val_size=None, augment=False, seed=None):
    """
    Create a sampler for OmniImage data that will return examples in the framework specified by OML (see
    ContinualMetaLearningSampler).

    Args:
        root (str or Path): Folder where the dataset will be located.
        download (bool): If True, download the data if it doesn't already exist. If False, raise an error.
        num_images_per_class (int): Number of images per class (only certain values are available---check the
                                    OmniImage library documentation).
        im_size (int): Desired size of images, or None to use the on-disk sizes.
        greyscale (bool): Whether to convert images to greyscale; False or None to keep the default coloring.
        train_size (int or float): Number (or fraction) of samples from the train set to actually use for training.
        val_size (int or float): Number (or fraction) of samples from the train set to use for validation.
        augment (bool): Whether to apply data augmentation to the training set.
        seed (int or list[int]): Random seed for sampling.

    Returns:
        ContinualMetaLearningSampler: The sampler class.
        tuple: The shape of the images that will be returned by the sampler (they will all be the same size).
    """
    train, test, image_shape = create_datasets(root, download, num_images_per_class, im_size, greyscale, augment)
    return ContinualMetaLearningSampler(train, test, seed, train_size, val_size), image_shape
