"""Data loader module for loading images for neural networks."""
import os
from torch.utils import data
from torchvision import transforms
from PIL import Image


class ImageFolder(data.Dataset):
    """Custom Dataset compatible with prebuilt DataLoader."""

    def __init__(self, root, transform=None):
        """Initialize image paths and preprocessing module."""
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.transform = transform

    def __getitem__(self, index):
        """Read an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        """Return the total number of image files."""
        return len(self.image_paths)


def get_loader(image_size, image_path):
    """Build and return Dataloader."""
    transform = transforms.Compose([
                    transforms.Scale(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data_loader = data.DataLoader(dataset=ImageFolder(image_path, transform),
                                  batch_size=16,
                                  shuffle=True,
                                  num_workers=4)
    return data_loader
