"""Data loader module for fingerprint dataset."""
import os
import os.path
import re
from datetime import datetime
import numpy as np
import scipy.misc
import torch
from torch.utils.data import Dataset


class TrFingerprints(Dataset):
    """Data loader class for fingerprint NIST dataset."""

    def __init__(self, transform):
        """Construct data loader."""
        self.dirs_idx = (0, 1, 2, 3, 4, 5, 6, 7)
        self.root_path = '../../../data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt'
        self.transform = transform
        np.random.seed(seed=datetime.now().microsecond)

    def __len__(self):
        """Return length of dataset."""
        # each of figs directories has 250 pairs of fingerprints
        n = 0
        data_dirs = sorted(os.listdir(self.root_path))
        for i in self.dirs_idx:
            current_dir = os.listdir(os.path.join(self.root_path, data_dirs[i]))
            p = re.compile('f.*\.png$')
            n += len([x for x in current_dir if p.match(x)])
        return 2*n  # because we assume that we have n pairs (same person), and n pairs (different)

    def __getitem__(self, index):
        """Access item from dataset."""
        def get_img_name(prefix, path):
            begin = prefix + '{:04d}'.format(250 * dir_number + image_index) + '_'
            p = re.compile(begin + '[0-9][0-9]' + '.png')
            return [x for x in os.listdir(path) if p.match(x)][0]

        def get_img(prefix):
            name = get_img_name(prefix, os.path.join(self.root_path, data_dirs[dir_number]))
            path = os.path.join(self.root_path, data_dirs[dir_number], name)
            img = scipy.misc.imread(path)
            return np.expand_dims(img, 2)

        label = int(np.random.choice([0, 1]))
        data_dirs = sorted(os.listdir(self.root_path))
        if label == 1:
            dir_number = np.random.choice(self.dirs_idx)
            image_index = np.random.randint(low=1, high=251)
            img_1 = get_img('f')
            img_2 = get_img('s')
        elif label == 0:
            dir_numbers = np.random.choice(self.dirs_idx, size=2, replace=True)
            image_indices = np.random.choice(np.arange(1, 251), size=2, replace=False)
            fs = ['f', 's']

            dir_number = dir_numbers[0]
            image_index = image_indices[0]
            img_1 = get_img(np.random.choice(fs))

            dir_number = dir_numbers[1]
            image_index = image_indices[1]
            img_2 = get_img(np.random.choice(fs))
        else:
            raise ValueError('wrong label')

        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return torch.FloatTensor([label]), img_1, img_2
