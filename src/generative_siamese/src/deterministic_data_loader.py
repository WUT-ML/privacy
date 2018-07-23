"""Quasi-deterministic data loader module for fingerprint and toy datasets."""
from datetime import datetime
import glob
import os
import os.path
import re
import pandas as pd
import numpy as np
import scipy.misc
import torch
import random
from torch.utils.data import Dataset


class TripletFERG(Dataset):
    """Quasi-deterministic triplet data loader class for FERG dataset.

    It returns pairs of images with corresponding label (0 or 1).
    Label 0 if images come from the same person, 1 otherwise.
    """

    def __init__(self, path, transform):
        """Construct data loader."""
        self.root_path = path
        self.transform = transform
        self.random = np.random.RandomState(seed=20180124)
        self.SIZE = 10000
        self.N_IDS = 6
        self.filenames = pd.read_csv(os.path.join(path, "images.csv"), header=None)
        self.range_dict = {0: (0, 7557), 1: (7558, 17560), 2: (17561, 25139),
                           3: (25140, 36409), 4: (36410, 45010), 5: (45011, 55765)}
        random.seed(52092)

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
        img_name = self.filenames.iloc[index, 0]
        img_path = os.path.join(self.root_path, img_name)
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

        # Convert images to grayscale
        if(img_1.ndim == 4):
            img_1 = img_1[:, :, :, 0]
        if(img_2.ndim == 4):
            img_2 = img_2[:, :, :, 0]

        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return torch.FloatTensor([label]), img_1, img_2


class FERGDataset(Dataset):
    """Quasi-deterministic data loader class for FERG dataset."""

    def __init__(self, path, transform):
        """Construct data loader."""
        self.root_path = path
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

        img_name = self.filenames.iloc[index, 0]
        img_path = os.path.join(self.root_path, img_name)
        rel_path = os.path.relpath(img_path, self.root_path)
        img = scipy.misc.imread(img_path, mode="RGBA")
        img[img[:, :, 3] == 0] = 255

        return img[:, :, 0:3], rel_path

    def __getitem__(self, index):
        """Access item from dataset."""
        if index < self.N_IMAGES:
            img_1, img_path = self.get_img(index)

        if self.transform:
            img_1 = self.transform(img_1)

        return img_path, img_1


class TripletToyDataset(Dataset):
    """Quasi-deterministic data loader class for toy-dataset."""

    def __init__(self, path, transform):
        """Construct data loader."""
        self.root_path = path
        self.transform = transform
        self.random = np.random.RandomState(seed=20180324)
        self.N_IDENTITIES = 80
        self.N_VARIANTS = 10
        self.TOTAL_N_VARIANTS = self.N_VARIANTS * (self.N_VARIANTS - 1) // 2

        # Create list of pairs of variants
        self.variants = []
        for i in range(0, self.N_VARIANTS):
            for j in range(i + 1, self.N_VARIANTS):
                self.variants.append((i, j))

    def __len__(self):
        """Return length of dataset."""
        return self.N_IDENTITIES * self.TOTAL_N_VARIANTS * 2

    def get_img(self, person_id, variant_id):
        """Get image of a given person."""
        assert variant_id >= 0 or variant_id < self.N_VARIANTS
        assert person_id > 0 and person_id <= self.N_IDENTITIES

        img_name = '{id:02d}_{var:02d}_'.format(id=person_id, var=variant_id)
        img_path = glob.glob(self.root_path + img_name + '*.png')[0]

        return np.expand_dims(scipy.misc.imread(img_path, mode="L"), 2)

    def __getitem__(self, index):
        """Access item from dataset."""
        if index < self.TOTAL_N_VARIANTS * self.N_IDENTITIES:
            # For 0-7199 return pairs of images for the same identity
            # Dataset index is zero-based, person ID is one-based
            label = 0  # 0 for same, 1 for different
            person_id, variant_id = divmod(index + self.TOTAL_N_VARIANTS, self.TOTAL_N_VARIANTS)
            variant_1, variant_2 = self.variants[variant_id]
            img_1 = self.get_img(person_id, variant_1)
            img_2 = self.get_img(person_id, variant_2)
        else:
            # For "virtual" indexes 7200-14399 return first image of the given id
            # and an image of a different id
            label = 1
            person_id, variant_id = divmod(index - self.N_IDENTITIES * self.TOTAL_N_VARIANTS +
                                           self.TOTAL_N_VARIANTS, self.TOTAL_N_VARIANTS)
            variant_1, variant_2 = self.variants[variant_id]
            others = list(set(np.arange(1, self.N_IDENTITIES + 1)) - set([person_id]))
            img_1 = self.get_img(person_id, variant_1)
            img_2 = self.get_img(self.random.choice(others), variant_2)

        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
        return torch.FloatTensor([label]), img_1, img_2


class ToyDataset(Dataset):
    """Quasi-deterministic data loader class for toy-dataset."""

    def __init__(self, path, transform):
        """Construct data loader."""
        self.root_path = path
        self.transform = transform
        self.random = np.random.RandomState(seed=20180324)
        self.N_IDENTITIES = 80
        self.N_VARIANTS = 100

    def __len__(self):
        """Return length of dataset."""
        return self.N_IDENTITIES * self.N_VARIANTS

    def get_img(self, person_id, variant_id):
        """Get image of a given person."""
        assert variant_id >= 0 or variant_id < self.N_VARIANTS
        assert person_id > 0 and person_id <= self.N_IDENTITIES

        img_name = '{id:02d}_{var:02d}_'.format(id=person_id, var=variant_id)
        img_path = glob.glob(self.root_path + img_name + '*.png')[0]
        rel_path = os.path.relpath(img_path, self.root_path)

        return np.expand_dims(scipy.misc.imread(img_path, mode="L"), 2), rel_path

    def __getitem__(self, index):
        """Access item from dataset."""
        if index < self.N_VARIANTS * self.N_IDENTITIES:
            # For 0-7999 return image
            # Dataset index is zero-based, person ID is one-based
            person_id, variant_id = divmod(index + self.N_VARIANTS, self.N_VARIANTS)
            img_1, path = self.get_img(person_id, variant_id)

        if self.transform:
            img_1 = self.transform(img_1)

        return path, img_1


class TrFingerprints(Dataset):
    """Quasi-deterministic data loader class for fingerprint NIST dataset."""

    def __init__(self, path, transform):
        """Construct data loader."""
        self.root_path = path
        self.transform = transform
        self.random = np.random.RandomState(seed=20180124)

    def __len__(self):
        """Return length of dataset."""
        return 2000 * 2

    def get_img(self, person_id, instance):
        """Get image of a given person."""
        assert instance == 'f' or instance == 's'
        assert person_id > 0 and person_id <= 2000

        figs_dir = (person_id - 1) // 250
        dir_path = os.path.join(self.root_path, 'figs_{figs_dir}'.format(figs_dir=figs_dir), '')

        img_name = '{instance}{person_id:04d}_'.format(instance=instance, person_id=person_id)
        img_path = glob.glob(dir_path + img_name + '*.png')[0]

        return np.expand_dims(scipy.misc.imread(img_path), 2)

    def __getitem__(self, index):
        """Access item from dataset."""
        if index < 2000:
            # For 0-1999 return pairs of images for the same person with a given ID
            # Dataset index is zero-based, person ID is one-based
            label = 0  # 0 for same, 1 for different
            person_id = index + 1
            img_1 = self.get_img(person_id, 'f')
            img_2 = self.get_img(person_id, 's')
        else:
            # For "virtual" indexes 2000-3999 return first image of the given person
            # and an image of a different person
            label = 1
            person_id = index - 2000 + 1
            others = list(set(np.arange(1, 2001)) - set([person_id]))
            img_1 = self.get_img(person_id, 'f')
            img_2 = self.get_img(self.random.choice(others), self.random.choice(['f', 's']))

        # Convert images to grayscale
        if(img_1.ndim == 4):
            img_1 = img_1[:, :, :, 0]
        if(img_2.ndim == 4):
            img_2 = img_2[:, :, :, 0]

        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return torch.FloatTensor([label]), img_1, img_2


class Fingerprints(TrFingerprints):
    """Quasi-deterministic data loader class for fingerprint NIST dataset."""

    def __init__(self, path, transform):
        """Construct data loader."""
        self.root_path = path
        self.transform = transform
        self.random = np.random.RandomState(seed=20180301)

    def __len__(self):
        """Return length of dataset."""
        return 2000 * 2

    def get_img(self, person_id, instance):
        """Get image of a given person."""
        assert instance == 'f' or instance == 's'
        assert person_id > 0 and person_id <= 2000

        figs_dir = (person_id - 1) // 250
        dir_path = os.path.join('figs_{figs_dir}'.format(figs_dir=figs_dir), '')

        img_name = '{instance}{person_id:04d}_'.format(instance=instance, person_id=person_id)
        img_path = glob.glob(self.root_path + dir_path + img_name + '*.png')[0]
        return np.expand_dims(scipy.misc.imread(img_path), 2), os.path.relpath(img_path,
                                                                               self.root_path)

    def __getitem__(self, index):
        """Access item from dataset."""
        if index < 2000:
            # For 0-1999 return first image of person with a given ID
            # Dataset index is zero-based, person ID is one-based
            person_id = index + 1
            img_1, path = self.get_img(person_id, 'f')
        else:
            # For 2000-3999 return second image of person with a given ID
            # Dataset index is zero-based, person ID is one-based
            person_id = index - 2000 + 1
            img_1, path = self.get_img(person_id, 's')

        if self.transform:
            img_1 = self.transform(img_1)

        return path, img_1
