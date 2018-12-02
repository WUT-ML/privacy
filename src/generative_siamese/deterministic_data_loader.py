"""Quasi-deterministic data loader module for FERG_DB_256 and CelebA datasets."""
import os
import os.path
import pandas as pd
import numpy as np
import scipy.misc
import torch
import random
from torch.utils.data import Dataset


class TripletCelebA(Dataset):
    """Quasi-deterministic triplet data loader class for CelebA dataset."""

    def __init__(self, path, transform, is_training=True):
        """Construct data loader."""
        self.dataset_path = os.path.join(path, "CelebA_unzipped", "img_align_celeba")
        self.transform = transform
        if is_training:
            self.random = np.random.RandomState(seed=20180725)
            random.seed(70049)
        else:
            self.random = np.random.RandomState(seed=51729261)
            random.seed(12345)
        self.SIZE = 10000
        self.N_IDS = 10177
        self.filenames = pd.read_csv(
            os.path.join(path, "ids.txt"), header=None, sep=' ')

    def __len__(self):
        """Return length of dataset."""
        return self.SIZE * 2

    def get_img(self, id):
        """Get image of a given person."""
        assert id >= 0 or id < self.N_IDS

        # Index is 1-based
        id += 1

        # Get a random image of a person
        img_path = self.filenames[self.filenames.iloc[:, 1] == id].iloc[:, 0].sample(1).iloc[0]
        img = scipy.misc.imread(os.path.join(self.dataset_path, img_path), mode="RGBA")
        img[img[:, :, 3] == 0] = 255

        return img[:, :, 0:3]

    def __getitem__(self, index):
        """Access item from dataset."""
        if index < self.SIZE:
            # Return pairs of images for the same person with a given ID
            label = 0  # 0 for same, 1 for different
            person_id = random.randint(0, self.N_IDS - 1)
            img_1 = self.get_img(person_id)
            img_2 = self.get_img(person_id)
        else:
            # Return paris of images for different people
            label = 1
            person_id_1, person_id_2 = random.sample(range(self.N_IDS), 2)
            img_1 = self.get_img(person_id_1)
            img_2 = self.get_img(person_id_2)

        # Apply transformation
        if img_1.ndim == 4:
            img_1 = img_1[:, :, :, 0]
        if img_2.ndim == 4:
            img_2 = img_2[:, :, :, 0]

        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return torch.FloatTensor([label]), img_1, img_2


class TripletFERG(Dataset):
    """Quasi-deterministic triplet data loader class for FERG dataset.

    It returns pairs of images with corresponding label (0 or 1).
    Label 0 if images come from the same person, 1 otherwise.
    """

    def __init__(self, path, transform, is_evaluation=False):
        """Construct data loader."""
        self.transform = transform
        if is_evaluation:
            self.random = np.random.RandomState(seed=20180124)
            random.seed(52092)
        else:
            self.random = np.random.RandomState(seed=51729261)
            random.seed(12345)
        self.SIZE = 10000
        self.N_IDS = 6
        self.filenames = pd.read_csv(os.path.join(path, "images.csv"), header=None)
        self.range_dict = {0: (0, 7557), 1: (7558, 17560), 2: (17561, 25139),
                           3: (25140, 36409), 4: (36410, 45010), 5: (45011, 55765)}

    def __len__(self):
        """Return length of dataset."""
        return self.SIZE * 2

    def get_random_index(self, id):
        """For a given person represented by id get a random image represented by index."""
        range = self.range_dict[id]
        return random.randint(*range)

    def get_img(self, id):
        """Get image of a given person."""
        assert id >= 0 or id < self.N_IDS

        # Get a random image of a person
        index = self.get_random_index(id)
        img_path = self.filenames.iloc[index, 0]
        img = scipy.misc.imread(img_path, mode="RGBA")
        img[img[:, :, 3] == 0] = 255

        return img[:, :, 0:3]

    def __getitem__(self, index):
        """Access item from dataset."""
        if index < self.SIZE:
            # Return pairs of images for the same person with a given ID
            label = 0  # 0 for same, 1 for different
            person_id = random.randint(0, self.N_IDS - 1)
            img_1 = self.get_img(person_id)
            img_2 = self.get_img(person_id)
        else:
            # Return paris of images for different people
            label = 1
            person_id_1, person_id_2 = random.sample(range(self.N_IDS), 2)
            img_1 = self.get_img(person_id_1)
            img_2 = self.get_img(person_id_2)

        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return torch.FloatTensor([label]), img_1, img_2


class FERGDataset(Dataset):
    """Quasi-deterministic data loader class for FERG dataset."""

    def __init__(self, path, transform):
        """Construct data loader."""
        self.transform = transform
        self.random = np.random.RandomState(seed=20180419)
        self.filenames = pd.read_csv(os.path.join(path, "images.csv"), header=None)
        self.N_IMAGES = self.filenames.shape[0]
        random.seed(20180703)

    def __len__(self):
        """Return length of dataset."""
        return self.N_IMAGES

    def get_img(self, index):
        """Get image of a given person."""
        assert index >= 0 or index < self.N_IMAGES

        img_path = self.filenames.iloc[index, 0]
        img = scipy.misc.imread(img_path, mode="RGBA")[0]
        img[img[:, :, 3] == 0] = 255

        return img[:, :, 0:3], img_path

    def __getitem__(self, index):
        """Access item from dataset."""
        if index < self.N_IMAGES:
            img_1, img_path = self.get_img(index)

        if self.transform:
            img_1 = self.transform(img_1)

        return img_path, img_1
