"""
PyTorch Dataset class for Mini-ImageNet.
"""

from enum import Enum
from os.path import basename
from pathlib import Path

import gdown
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Lambda

from .class_indexed_dataset import ClassIndexedDataset
from .ContinualMetaLearningSampler import ContinualMetaLearningSampler
from .utils import extract_archive, check_integrity, list_dir, list_files


class Split(Enum):
    TRAIN = ("Training", "train.tar", "107FTosYIeBn5QbynR46YG91nHcJ70whs", "62af9b3c839974dad2d474e6325795af")
    TEST = ("Testing", "test.tar", "1yKyKgxcnGMIAnA_6Vr2ilbpHMc9COg-v", "318185fc3e3bf8bc57de887d9682c666")
    VAL = ("Validation", "val.tar", "1hSMUMj5IRpf-nQs1OwgiQLmGZCN0KDWl", "ab02f050b0bf66823e7acb0c1ac1bc6b")

    def __init__(self, set_name, filename, gid, md5):
        self.set_name = set_name
        self.filename = filename
        self.gid = gid
        self.md5 = md5

    def __str__(self):
        return self.set_name


class MiniImageNet(ClassIndexedDataset):
    """
    Mini-ImageNet dataset, as given by [Mini-ImageNet Tools](https://github.com/yaoyao-liu/mini-imagenet-tools/).
    If this download ever stops working, those tools can be used to regenerate the dataset. You can also generate
    custom, modified versions of the dataset.
    """

    def __init__(
        self,
        root,
        split,
        transform=None,
        target_transform=None,
        greyscale=False,
        download=False,
        quiet=False,
    ):
        """
        Create the dataset.

        Args:
            root (string): Desired root directory for the dataset (usually named `mini-imagenet/`, one directory above
                the `processed_images/` folder).
            split (Subset, optional): Which split of the dataset to use (train, test, or validation).
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            greyscale (bool, optional): If True, converts images to greyscale when sampling.
            download (bool, optional): If true, downloads the dataset zip files from the internet and
                puts it in root directory. If the zip files are already downloaded, they are not
                downloaded again.
            quiet (bool, optional): If True, do not print any messages during download and setup.
        """
        self.imagenet_folder = Path(root).resolve()
        self.split = split
        self.tarpath = self.imagenet_folder / self.split.filename
        # The mini-imagenet-tools usually puts the raw images into this folder structure, so let's conform to their
        # convention.
        self.target_folder = self.tarpath.parent / "processed_images" / self.tarpath.stem
        self.transform = transform
        self.target_transform = target_transform
        self.greyscale = greyscale
        self.quiet = quiet

        if download:
            self.download()
        else:
            self._extract_if_needed()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to re-download it.")

        # self.classes is a mapping from class ID (ImageNet class IDs like "n01855672") to a sorted list of images, for
        # all classes and images in this dataset. Keys are sorted alphabetically for determinism.
        all_class_folders = sorted(list_dir(self.target_folder, prefix=True))
        self.classes = {basename(cls): sorted(list_files(cls, ".jpg", prefix=True)) for cls in all_class_folders}
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

    def _check_integrity(self):
        if not check_integrity(self.tarpath, self.split.md5):
            return False
        return True

    def _extract_if_needed(self):
        """
        Untar the dataset if the folder isn't already present.
        """
        if not self.target_folder.exists():
            extract_archive(self.tarpath, self.target_folder.parent)
            if not self.quiet:
                print(self.split.filename, "unpacked to", self.target_folder)
        else:
            if not self.quiet:
                print(self.target_folder, "already exists.")

    def download(self):
        """
        Only downloads the dataset if it doesn't already exist, or if it seems corrupted.
        """
        # This will also check the integrity of the file for us and re-download if md5 doesn't match.
        gdown.cached_download(id=self.split.gid, path=self.tarpath, md5=self.split.md5, quiet=self.quiet)
        self._extract_if_needed()


def create_OML_sampler(root, im_size=28, seed=None):
    transforms = Compose(
        [
            Resize(im_size, Image.LANCZOS),
            ToTensor(),
            Lambda(lambda x: x.unsqueeze(0)),  # used to add batch dimension
        ]
    )
    t_transforms = Lambda(lambda x: torch.tensor(x).unsqueeze(0))
    train = MiniImageNet(
        root=root,
        split=Split.TRAIN,
        transform=transforms,
        target_transform=t_transforms,
        greyscale=True,
        download=True,
    )
    test = MiniImageNet(
        root=root,
        split=Split.TEST,
        transform=transforms,
        target_transform=t_transforms,
        greyscale=True,
        download=True,
    )
    return ContinualMetaLearningSampler(train, test, seed)
