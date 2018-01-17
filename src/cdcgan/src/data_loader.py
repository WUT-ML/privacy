"""
Data loader module for loading images for neural networks.

Works for fingerprints 10 class dataset only.
"""

import os
from torch.utils import data
from torchvision import transforms
import torch
from PIL import Image


class ImageFolder(data.Dataset):
    """Custom Dataset compatible with prebuilt DataLoader."""

    def __init__(self, root, image_size, transform=None):
        """Initialize image paths and preprocessing module."""
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_size = image_size
        self.transform = transform

    def __getitem__(self, index):
        """Read an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        y = get_label(image_path)
        y_dim = get_label_dim()
        y_reshaped_1d = label_reshape_1d(y, y_dim)
        y_reshaped_2d = label_reshape_2d(y, y_dim, self.image_size)
        return image, y_reshaped_1d, y_reshaped_2d

    def __len__(self):
        """Return the total number of image files."""
        return len(self.image_paths)


def get_label(image_path):
    """Return label (class) for image file."""
    # Only for fingerprint dataset!
    return int(image_path[-6:-4]) - 1


def get_label_dim():
    """Return number of labels (classes) for image file."""
    # Only for fingerprint dataset!
    return 10


def label_reshape_1d(label, label_dim):
    """Reshape label to list of zeros and ones."""
    tensor = torch.FloatTensor(label_dim).zero_()
    tensor[label] = 1
    return tensor


def label_reshape_2d(label, label_dim, image_size):
    """Reshape label to matrices full of zeros or ones."""
    tensor = torch.FloatTensor(3 * label_dim, image_size, image_size).zero_()
    tensor[(3 * label):(3 * (label + 1)), :, :] = 1
    return tensor


def get_loader(image_size, image_path):
    """Build and return Dataloader."""
    transform = transforms.Compose([
                    transforms.Scale(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data_loader = data.DataLoader(dataset=ImageFolder(image_path, image_size, transform),
                                  batch_size=16,
                                  shuffle=True,
                                  num_workers=4)
    return data_loader
