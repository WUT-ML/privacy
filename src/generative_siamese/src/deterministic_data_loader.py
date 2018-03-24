"""Quasi-deterministic data loader module for fingerprint dataset."""
from datetime import datetime
import glob
import os
import os.path
import re

import numpy as np
import scipy.misc
import torch
from torch.utils.data import Dataset


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
