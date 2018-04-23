"""Quasi-deterministic data loader module for FERG dataset."""
from datetime import datetime
import glob
import os
import os.path
import re
import pandas as pd

import numpy as np
import scipy.misc
import torch
from torch.utils.data import Dataset


class FERGDataset(Dataset):
    """Quasi-deterministic data loader class for FERG dataset."""

    def __init__(self, path, transform):
        """Construct data loader."""
        self.root_path = path
        self.transform = transform
        self.random = np.random.RandomState(seed=20180419)
        self.filenames = pd.read_csv(os.path.join(path, "images.csv"))
        self.N_IMAGES = self.filenames.shape[0]

    def __len__(self):
        """Return length of dataset."""
        return self.N_IMAGES

    def get_img(self, index):
        """Get image of a given person."""
        assert index >= 0 or index < self.N_IMAGES

        img_name = self.filenames.iloc[index, 0]
        img_path = os.path.join(self.root_path, img_name)
        rel_path = os.path.relpath(img_path, self.root_path)
        attributes = FERGAttributes(rel_path)
        attr = attributes.get_attr()
        id = attributes.get_id()

        return scipy.misc.imread(img_path, mode="RGB"), id, attr

    def __getitem__(self, index):
        """Access item from dataset."""
        if index < self.N_IMAGES:
            img_1, id, attr = self.get_img(index)

        if self.transform:
            img_1 = self.transform(img_1)

        return img_1, id, attr


class FERGAttributes():
    """Class representing id and attributes for images from FERG dataset."""

    dict_id = {'aia': 0, 'bonnie': 1, 'jules': 2, 'malcolm': 3, 'mery': 4, 'ray': 5}
    dict_attr = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5,
                 'surprise': 6}

    def __init__(self, path):
        """Initialize object, use file path to read id and attribute."""
        id_key, attr_key = (path.split('/')[1]).split('_')
        self.id = FERGAttributes.dict_id[id_key]
        self.attr = FERGAttributes.dict_attr[attr_key]

    def get_attr(self):
        """Get attribute."""
        return self.attr

    def get_id(self):
        """Get id."""
        return self.id
