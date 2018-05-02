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
        self.images = pd.read_csv(os.path.join(path, "images.csv"),
                                  header=None,
                                  names=["filenames", "id", "attr"])
        self.N_IMAGES = self.images.shape[0]

        # Dictionaries to get id and attr from filename
        self.dict_id = {'aia': 0, 'bonnie': 1, 'jules': 2, 'malcolm': 3, 'mery': 4, 'ray': 5}
        self.dict_attr = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5,
                          'surprise': 6}

        # Get id and attr from filename
        self.images['id'] = self.images['filenames'].map(
            lambda a: self.dict_id[(a.split('/')[1]).split('_')[0]])
        self.images['attr'] = self.images['filenames'].map(
            lambda a: self.dict_attr[(a.split('/')[1]).split('_')[1]])

    def __len__(self):
        """Return double length of dataset."""
        return self.N_IMAGES * 2

    def get_img(self, index):
        """Get image of a given person."""
        assert index >= 0 or index < self.N_IMAGES
        img_name = self.images.at[index, "filenames"]
        img = scipy.misc.imread(os.path.join(self.root_path, img_name), mode="RGBA")
        img[img[:, :, 3] == 0] = 255

        return img[:, :, 0:3], self.images.at[index, "id"], self.images.at[index, "attr"]

    def get_img_id(self, id):
        """Get random image of a given person."""
        return self.get_img(self.images[self.images.id == id].sample().index.tolist()[0])[0]

    def get_img_not_id(self, id):
        """Get random image of a person with different id."""
        return self.get_img(self.images[self.images.id != id].sample().index.tolist()[0])[0]

    def __getitem__(self, index):
        """Access item from dataset."""
        # Return pair of images of the same person
        if index < self.N_IMAGES:
            img_1, id_1, attr_1 = self.get_img(index)
            img_2 = self.get_img_id(id_1)
            label = 0  # 0 for same, 1 for different
        # Return pair of images of the different persons
        else:
            img_1, id_1, attr_1 = self.get_img(index % self.N_IMAGES)
            img_2 = self.get_img_not_id(id_1)
            label = 1

        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return img_1, id_1, attr_1, img_2, torch.FloatTensor([label])
