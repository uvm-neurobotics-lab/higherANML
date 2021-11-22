from time import time
import numpy as np
from numpy.random import choice
from sklearn.model_selection import train_test_split

import torch
from torchvision.transforms import Compose, ToTensor, Resize, Lambda
from torchvision.transforms.functional import InterpolationMode

from .omniglot import Omniglot  # patched to use memoization
from utils import unzip, divide_chunks


def collate_fn(samples):
    # takes a list of (im,label) and returns [ims],[labels]
    xs = torch.stack([sample[0] for sample in samples])
    ys = torch.tensor([sample[1] for sample in samples])
    return xs, ys


class OmniSampler:
    def __init__(self, root, preload_train=False, preload_test=False, im_size=28):
        transforms = Compose(
            [
                Resize(im_size, InterpolationMode.LANCZOS),
                ToTensor(),
                Lambda(lambda x: x.unsqueeze(0)),  # used to add batch dimension
            ]
        )
        t_transforms = Lambda(lambda x: torch.tensor(x).unsqueeze(0))
        self.omni_train = Omniglot(
            root=root,
            background=True,
            download=True,
            transform=transforms,
            target_transform=t_transforms,
        )
        self.omni_test = Omniglot(
            root=root,
            background=False,
            download=True,
            transform=transforms,
            target_transform=t_transforms,
        )

        # task_train contains tasks i.e. 963 sets of 20 characters
        self.tasks_train = np.arange(len(self.omni_train._character_images))
        # chars_train contains the characters directly i.e. 964*20=19280 images-labels pairs
        self.chars_train = np.arange(len(self.omni_train._flat_character_images))
        # task_test contains tasks i.e. 659 sets of 20 characters
        self.tasks_test = np.arange(len(self.omni_test._character_images))
        # chars_train contains the characters directly i.e. 600*20=13180 images-labels pairs
        self.chars_test = np.arange(len(self.omni_test._flat_character_images))

        if preload_train:
            start = time()
            print("Pre-loading Omniglot train...")
            _ = [img for img in self.omni_train]
            end = time()
            print(f"{end - start:.1f}s : Omniglot train pre-loaded.")

        if preload_test:
            start = time()
            print("Pre-loading Omniglot test...")
            _ = [img for img in self.omni_test]
            end = time()
            print(f"{end - start:.1f}s : Omniglot test pre-loaded.")

    def num_train_classes(self):
        return len(self.omni_train._character_images)

    def num_test_classes(self):
        return len(self.omni_test._character_images)

    def num_total_classes(self):
        return len(self.omni_train._character_images) + len(self.omni_test._character_images)

    def sample_train(self, remember_size=64, device="cuda"):
        # Used during Meta-Training
        # prepare data for training and send it to the correct device

        # Sample a random class (i.e. 20 chars) for training
        task_id = choice(self.tasks_train)
        samples = [self.omni_train[i] for i in range(task_id * 20, (task_id + 1) * 20)]
        train_ims, train_labels = collate_fn(samples)

        # Sample 64 random characters for meta-training
        ids = choice(self.chars_train, size=remember_size, replace=False)
        samples = [self.omni_train[i] for i in ids]
        valid_ims, valid_labels = collate_fn(samples)

        # valid = outer loop training
        # 64 randomly sampled images + the 20 images from the last training trajectory
        # are concatenated in one single batch of shape (84,1,28,28) [B,C,H,W]
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

    def sample_test(self, num_tasks, train_examples=15, device="cuda"):
        assert num_tasks <= len(
            self.tasks_test
        ), f"Number of tasks requested is too large: {num_tasks} > {len(self.tasks_test)}"
        assert (
            train_examples <= 20
        ), f"Number of examples requested is too large: {train_examples} > 20"

        # chose the n tasks to use
        tasks = choice(self.tasks_test, size=num_tasks, replace=False)

        # get the 20 indexes of each task (they are sequential)
        task_ids = [range(task * 20, (task + 1) * 20) for task in tasks]

        # split each group of 20 ids into (usually) 15 train and 5 test, unzip to separate train and test sequences
        train_tasks, test_tasks = unzip(
            train_test_split(ids, train_size=train_examples, shuffle=True)
            for ids in task_ids
        )

        # assemble the train/test trajectories
        train_traj = [self.omni_test[i] for train_task in train_tasks for i in train_task]
        test_traj = [self.omni_test[i] for test_task in test_tasks for i in test_task]

        # test-train examples are divided by task and sent to device (cpu/cuda)
        chunk2device = lambda chunk: [(im.to(device), label.to(device)) for im, label in chunk]
        train_tasks = [chunk2device(chunk) for chunk in divide_chunks(train_traj, n=train_examples)]

        # test-test tasks are collected into a massive tensor for one-pass evaluation
        ims, labels = list(zip(*test_traj))
        test_data = (torch.cat(ims).to(device), torch.cat(labels).to(device))

        return train_tasks, test_data
