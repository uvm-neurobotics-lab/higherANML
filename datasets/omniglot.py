import logging
from os.path import join
from time import time

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Grayscale, ToTensor, Resize, RandomResizedCrop, Lambda
from torchvision.transforms.functional import InterpolationMode

from .class_indexed_dataset import ClassIndexedDataset
from .ContinualMetaLearningSampler import ContinualMetaLearningSampler
from .IIDSampler import IIDSampler
from .utils import download_and_extract_archive, check_integrity, list_dir, list_files


class Omniglot(ClassIndexedDataset):
    """`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        background (bool, optional): If True, creates dataset from the "background" set, otherwise
            creates from the "evaluation" set. This terminology is defined by the authors.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        greyscale (bool, optional): If True, converts images to greyscale when sampling.
        download (bool, optional): If true, downloads the dataset zip files from the internet and
            puts it in root directory. If the zip files are already downloaded, they are not
            downloaded again.
    """

    folder = "omniglot-py"
    download_url_prefix = "https://github.com/brendenlake/omniglot/raw/master/python"
    zips_md5 = {
        "images_background": "68d2efa1b9178cc56df9314c21c6e718",
        "images_evaluation": "6b91aef0f799c5bb55b94e3f2daec811",
    }

    def __init__(
            self,
            root,
            background=True,
            transform=None,
            target_transform=None,
            greyscale=True,
            download=False,
            memoize=True,
    ):
        self.root = join(root, self.folder)
        self.background = background
        self.transform = transform
        self.target_transform = target_transform
        self.greyscale = greyscale
        self.memoize = memoize
        self.memo = {}

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        self.target_folder = join(self.root, self._get_target_folder())
        # _alphabets is a list of all alphabet names in this dataset.
        self._alphabets = list_dir(self.target_folder)
        # _characters is a list of strings like "<Alphabet>/<character>", e.g. "Atlantean/character09".
        self._characters = sum(
            [
                [join(a, c) for c in list_dir(join(self.target_folder, a))]
                for a in self._alphabets
            ],
            [],
        )
        # _character_images is a list of lists of filenames, one list per unique character (only the basename, not the
        # full paths).
        self._character_images = [
            [
                (image, idx)
                for image in list_files(join(self.target_folder, character), ".png")
            ]
            for idx, character in enumerate(self._characters)
        ]
        # _flat_character_images is just the above list but flattened. This serves as an index of the entire dataset.
        self._flat_character_images = sum(self._character_images, [])
        # self._class_index is a list of lists of the kind specified by `ClassIndexedDataset.class_index()`.
        self._class_index = []
        currdex = 0
        for img_list in self._character_images:
            self._class_index.append(np.arange(currdex, currdex + len(img_list)))
            currdex += len(img_list)

    @property
    def class_index(self):
        return self._class_index

    def __len__(self):
        return len(self._flat_character_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        if not self.memoize or index not in self.memo:
            image_name, character_class = self._flat_character_images[index]
            image_path = join(
                self.target_folder, self._characters[character_class], image_name
            )
            image = Image.open(image_path)
            if self.greyscale:
                image = image.convert("L")

            if self.transform:
                image = self.transform(image)

            if self.target_transform:
                character_class = self.target_transform(character_class)

            if self.memoize:
                self.memo[index] = (image, character_class)
        else:
            image, character_class = self.memo[index]

        return image, character_class

    def _check_integrity(self):
        zip_filename = self._get_target_folder()
        if not check_integrity(join(self.root, zip_filename + ".zip"), self.zips_md5[zip_filename]):
            return False
        return True

    def download(self):
        if not self._check_integrity():
            filename = self._get_target_folder()
            zip_filename = filename + ".zip"
            url = self.download_url_prefix + "/" + zip_filename
            download_and_extract_archive(url, self.root, filename=zip_filename, md5=self.zips_md5[filename])

    def _get_target_folder(self):
        return "images_background" if self.background else "images_evaluation"


def get_transform(im_size=28, greyscale=True, augment=False):
    """
    Get the image transform according to the given parameters.

    Args:
        im_size (int): Desired size of images, or None to default to 28x28.
        greyscale (bool, optional): If False, replicates the 1-channel images to 3 channels when sampling.
        augment (bool): Whether to apply data augmentation to the training set.

    Returns:
        callable: The transform.
    """
    transform_list = []
    if greyscale is False:  # Only if it's explicitly false, not None.
        transform_list.append(Grayscale(num_output_channels=3))
    if augment:
        transform_list.append(RandomResizedCrop(im_size, scale=(0.8, 1.0), interpolation=InterpolationMode.LANCZOS))
    else:
        transform_list.append(Resize(im_size, InterpolationMode.LANCZOS))
    transform_list.append(ToTensor())
    return Compose(transform_list)


def create_datasets(root, download=True, preload_train=False, preload_test=False, im_size=28, greyscale=True,
                    augment=False):
    """
    Create a pair of (train, test) datasets for Omniglot.

    Args:
        root (str or Path): Folder where the dataset will be located.
        download (bool): If True, download the data if it doesn't already exist. If False, raise an error.
        preload_train (bool): Whether to load all training images into memory up-front.
        preload_test (bool): Whether to load all testing images into memory up-front.
        im_size (int): Desired size of images, or None to default to 28x28.
        greyscale (bool, optional): If False, replicates the 1-channel images to 3 channels when sampling.
        augment (bool): Whether to apply data augmentation to the training set.

    Returns:
        Omniglot: The training ("images_background") set.
        Omniglot: The testing ("images_evaluation") set.
        tuple: The shape of the images that will be returned by both datasets (they will all be the same size).
    """
    if im_size is None:
        # For now, use 28x28 as default, instead of the Omniglot default of 105x105.
        im_size = 28
    t_transforms = Lambda(lambda x: torch.tensor(x))
    omni_train = Omniglot(
        root=root,
        background=True,
        download=download,
        transform=get_transform(im_size, greyscale, augment),
        target_transform=t_transforms,
        memoize=not augment,
    )
    omni_test = Omniglot(
        root=root,
        background=False,
        download=download,
        transform=get_transform(im_size, greyscale),
        target_transform=t_transforms,
    )

    if preload_train:
        start = time()
        logging.info("Pre-loading Omniglot train...")
        _ = [img for img in omni_train]
        end = time()
        logging.info(f"{end - start:.1f}s : Omniglot train pre-loaded.")

    if preload_test:
        start = time()
        logging.info("Pre-loading Omniglot test...")
        _ = [img for img in omni_test]
        end = time()
        logging.info(f"{end - start:.1f}s : Omniglot test pre-loaded.")

    image_shape = (3 if (greyscale is False) else 1, im_size, im_size)
    return omni_train, omni_test, image_shape


def create_iid_sampler(root, download=True, preload_train=False, preload_test=False, im_size=28, greyscale=True,
                       batch_size=256, train_size=None, val_size=None, augment=False, seed=None):
    """
    Create a sampler for Omniglot data which will sample shuffled batches in the standard way for i.i.d. training.

    Args:
        root (str or Path): Folder where the dataset will be located.
        download (bool): If True, download the data if it doesn't already exist. If False, raise an error.
        preload_train (bool): Whether to load all training images into memory up-front.
        preload_test (bool): Whether to load all testing images into memory up-front.
        im_size (int): Desired size of images, or None to default to 28x28.
        greyscale (bool, optional): If False, replicates the 1-channel images to 3 channels when sampling.
        batch_size (int): Number of images per batch for both datasets.
        train_size (int or float): Number (or fraction) of samples from the train set to actually use for training.
        val_size (int or float): Number (or fraction) of samples from the train set to use for validation.
        augment (bool): Whether to apply data augmentation to the training set.
        seed (int or list[int]): Random seed for sampling.

    Returns:
        IIDSampler: The sampler class.
        tuple: The shape of the images that will be returned by the sampler (they will all be the same size).
    """
    omni_train, omni_test, image_shape = create_datasets(root, download, preload_train, preload_test, im_size,
                                                         greyscale, augment)
    return IIDSampler(omni_train, omni_test, batch_size, train_size, val_size, seed), image_shape


def create_OML_sampler(root, download=True, preload_train=False, preload_test=False, im_size=28, greyscale=True,
                       train_size=None, val_size=None, augment=False, seed=None):
    """
    Create a sampler for Omniglot data that will return examples in the framework specified by OML (see
    ContinualMetaLearningSampler).

    Args:
        root (str or Path): Folder where the dataset will be located.
        download (bool): If True, download the data if it doesn't already exist. If False, raise an error.
        preload_train (bool): Whether to load all training images into memory up-front.
        preload_test (bool): Whether to load all testing images into memory up-front.
        im_size (int): Desired size of images, or None to default to 28x28.
        greyscale (bool, optional): If False, replicates the 1-channel images to 3 channels when sampling.
        train_size (int or float): Number (or fraction) of samples from the train set to actually use for training.
        val_size (int or float): Number (or fraction) of samples from the train set to use for validation.
        augment (bool): Whether to apply data augmentation to the training set.
        seed (int or list[int]): Random seed for sampling.

    Returns:
        ContinualMetaLearningSampler: The sampler class.
        tuple: The shape of the images that will be returned by the sampler (they will all be the same size).
    """
    omni_train, omni_test, image_shape = create_datasets(root, download, preload_train, preload_test, im_size,
                                                         greyscale, augment)
    return ContinualMetaLearningSampler(omni_train, omni_test, seed, train_size, val_size), image_shape
