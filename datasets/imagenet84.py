"""
PyTorch Dataset class for OmniImage.
"""
import logging
from os.path import basename
from pathlib import Path

import numpy as np
import torch
from numpy.random import default_rng, SeedSequence
from PIL import Image
from omnimage.dataset import get_fixed_random_test_classes, get_evolved_test_classes, OmnImageDataset
from torchvision.transforms import Compose, Lambda, Normalize, RandomResizedCrop, RandomHorizontalFlip, Resize, ToTensor

from .class_indexed_dataset import ClassIndexedDataset
from .ContinualMetaLearningSampler import ContinualMetaLearningSampler
from .IIDSampler import IIDSampler
from .utils import extract_archive, check_integrity, list_dir, list_files


class ImageNet84(ClassIndexedDataset):
    """
    ImageNet84 dataset, presented as an analogue to OmniImage where the images are randomly selected instead of
    selected by evolutionary algorithm.
    """

    def __init__(
            self,
            root,
            split,
            random_split=False,
            num_images_per_class=None,
            transform=None,
            target_transform=None,
            greyscale=False,
            seed=None,
    ):
        """
        Create the dataset.

        Args:
            root (string): Desired root directory for the dataset (usually named `ImageNet84/`).
            split (string): Which split of the dataset to use ("train" or "test").
            random_split (bool): Whether to use the pre-determined train/test split or a randomly sampled split.
            num_images_per_class (int): Number of images per class.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            greyscale (bool, optional): If True, converts images to greyscale when sampling.
            seed (int or list[int], optional): Seed used for sampling images per class.
        """
        self.target_folder = Path(root).resolve()
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.greyscale = greyscale

        if not self.target_folder.is_dir():
            raise FileNotFoundError(f"Not a valid directory: {self.target_folder}")

        # If seed is None, then we will pull entropy from the OS and log it in case we need to reproduce this run.
        ss = SeedSequence(seed)
        logging.info(f"ImageNet84 dataset is using seed = {ss.entropy}")
        rng = default_rng(ss)

        # Determine which split we're interested in.
        test_classes = get_fixed_random_test_classes() if random_split else get_evolved_test_classes()
        is_test_split = (split.lower() == "test")

        # Check that all classes are present before filtering. Sort alphabetically for determinism.
        all_class_folders = sorted(list_dir(self.target_folder, prefix=True))
        if len(all_class_folders) != 1000:
            raise RuntimeError(f"Expected to find ImageNet 1000 classes, but instead found {len(all_class_folders)}"
                               " class folders.")

        # Subsample images before filtering. This will cost us more time, but means we will always select the same
        # image set when using the same seed, even if using different classes.
        def sample_images(class_folder):
            all_images = sorted(list_files(class_folder, ".jpg", prefix=True))
            if num_images_per_class is not None:
                all_images = rng.choice(all_images, num_images_per_class)
            return all_images

        # self.classes is a mapping from class ID (ImageNet class IDs like "n01855672") to a sorted list of images, for
        # all classes and images in this dataset. Keys are sorted alphabetically for determinism.
        self.classes = {basename(cls): sample_images(cls) for cls in all_class_folders}

        # Now we can finally narrow the full dataset down to just the split we're interested in.
        def is_desired_class(item):
            is_test = basename(item[0]) in test_classes
            # True when this class' split matches the one we want (train or test).
            return is_test == is_test_split

        self.classes = dict(filter(is_desired_class, self.classes.items()))

        # TODO: TMP
        vals = list(self.classes.values())
        print(f"First 10 of first class: {[basename(p) for p in vals[0][:10]]}")
        print(f"First 10 of last class: {[basename(p) for p in vals[-1][:10]]}")

        # self._class_index is a list of lists of the kind specified by `ClassIndexedDataset.class_index()`.
        self._class_index = []
        # self.all_items is a flattened list of all items from all classes. They will be sorted in alphabetical order of
        # the full filepath of each image. Classes are identified by their index, not their string name.
        self.all_items = []
        currdex = 0
        for clsdex, (cls, images) in enumerate(self.classes.items()):
            self._class_index.append(np.arange(currdex, currdex + len(images)))
            for img in images:
                self.all_items.append((clsdex, img))
            currdex += len(images)

    @property
    def class_index(self):
        return self._class_index

    def __len__(self):
        return len(self.all_items)

    def __getitem__(self, index):
        """
        Fetches and transforms the example corresponding to the given index.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        label, image_path = self.all_items[index]
        image = Image.open(image_path)
        if self.greyscale:
            image = image.convert("L")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def create_datasets(root, num_images_per_class=None, im_size=None, greyscale=False, augment=False, random_split=False,
                    seed=None):
    """
    Create a pair of (train, test) datasets for ImageNet84.

    Args:
        root (str or Path): Folder where the dataset will be located.
        num_images_per_class (int): Number of images per class.
        im_size (int): Desired size of images, or None to use the on-disk sizes.
        greyscale (bool): Whether to convert images to greyscale; False or None to keep the default coloring.
        augment (bool): Whether to apply data augmentation to the training set.
        random_split (bool): Whether to randomly split the data, or use the predetermined OmniImage splits.
        seed (int or list[int]): Random seed for sampling.

    Returns:
        ImageNet84: The training set.
        ImageNet84: The testing set.
        tuple: The shape of the images that will be returned by the sampler (they will all be the same size).
    """
    # We know that by default this dataset has 84x84 color images.
    actual_size = 84 if im_size is None else im_size
    # Normalize images exactly the same was as is done in OmniImage:
    # https://github.com/lfrati/OmnImage/blob/a62e9d20ce377e8632a29d580f9312fe98294516/src/omnimage/dataset.py#L49-L52
    normalize = Normalize((0.48751324, 0.46667117, 0.41095525), (0.26073888, 0.2528451, 0.2677635))

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
            train_transforms.append(Resize(im_size))
    train_transforms.append(ToTensor())
    train_transforms.append(normalize)
    train_transforms = Compose(train_transforms)
    t_transforms = Lambda(lambda x: torch.tensor(x))
    train = ImageNet84(
        root=root,
        num_images_per_class=num_images_per_class,
        split="train",
        random_split=random_split,
        transform=train_transforms,
        target_transform=t_transforms,
        greyscale=greyscale,
        seed=seed,
    )

    # Build test set.
    test_transforms = []
    # Only resize if a specific size is requested.
    if im_size is not None:
        test_transforms.append(Resize(im_size))
    test_transforms.append(ToTensor())
    test_transforms.append(normalize)
    test_transforms = Compose(test_transforms)
    test = ImageNet84(
        root=root,
        num_images_per_class=num_images_per_class,
        split="test",
        random_split=random_split,
        transform=test_transforms,
        target_transform=t_transforms,
        greyscale=greyscale,
        seed=seed,
    )

    image_shape = (1 if greyscale else 3, actual_size, actual_size)
    return train, test, image_shape


def create_iid_sampler(root, num_images_per_class=None, im_size=None, greyscale=False, batch_size=128,
                       train_size=None, val_size=None, augment=False, random_split=False, seed=None):
    """
    Create a sampler for ImageNet84 data which will sample shuffled batches in the standard way for i.i.d. training.

    Args:
        root (str or Path): Folder where the dataset will be located.
        num_images_per_class (int): Number of images per class to sample.
        im_size (int): Desired size of images, or None to use the on-disk sizes.
        greyscale (bool): Whether to convert images to greyscale; False or None to keep the default coloring.
        batch_size (int): Number of images per batch for both datasets.
        train_size (int or float): Number (or fraction) of samples from the train set to actually use for training.
        val_size (int or float): Number (or fraction) of samples from the train set to use for validation.
        augment (bool): Whether to apply data augmentation to the training set.
        random_split (bool): Whether to randomly split the data, or use the predetermined OmniImage splits.
        seed (int or list[int]): Random seed for sampling.

    Returns:
        IIDSampler: The sampler class.
        tuple: The shape of the images that will be returned by the sampler (they will all be the same size).
    """
    train, test, image_shape = create_datasets(root, num_images_per_class, im_size, greyscale, augment, random_split,
                                               seed)
    return IIDSampler(train, test, batch_size, train_size, val_size), image_shape


def create_OML_sampler(root, num_images_per_class=None, im_size=None, greyscale=False, train_size=None,
                       val_size=None, augment=False, random_split=False, seed=None):
    """
    Create a sampler for ImageNet84 data that will return examples in the framework specified by OML (see
    ContinualMetaLearningSampler).

    Args:
        root (str or Path): Folder where the dataset will be located.
        num_images_per_class (int): Number of images per class to sample.
        im_size (int): Desired size of images, or None to use the on-disk sizes.
        greyscale (bool): Whether to convert images to greyscale; False or None to keep the default coloring.
        train_size (int or float): Number (or fraction) of samples from the train set to actually use for training.
        val_size (int or float): Number (or fraction) of samples from the train set to use for validation.
        augment (bool): Whether to apply data augmentation to the training set.
        random_split (bool): Whether to randomly split the data, or use the predetermined OmniImage splits.
        seed (int or list[int]): Random seed for sampling.

    Returns:
        ContinualMetaLearningSampler: The sampler class.
        tuple: The shape of the images that will be returned by the sampler (they will all be the same size).
    """
    train, test, image_shape = create_datasets(root, num_images_per_class, im_size, greyscale, augment, random_split,
                                               seed)
    return ContinualMetaLearningSampler(train, test, seed, train_size, val_size), image_shape
