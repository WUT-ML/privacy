"""Deep convolutional GAN with siamese discriminator."""

from data_loader import TrFingerprints
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from generator_plus_siamese_solver import SiameseGanSolver
import argparse
import os
from torch.backends import cudnn


def main():
    """Entry point for GAN with siamese discriminator (training or sampling)."""
    cudnn.benchmark = True

    # Load and trasnform dataset
    dataset_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    dataset = TrFingerprints(transform=dataset_transform, path=config.image_path)

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)

    # Train and sample the images
    if config.mode == 'train':
        # Prepare data loader for dataset
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=config.batch,
                                 num_workers=config.jobs,
                                 shuffle=True,
                                 drop_last=False)
        # Train neural network
        solver = SiameseGanSolver(config, data_loader)
        solver.train()
    elif config.mode == 'sample':
        # Prepare data loader for dataset
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=1,
                                 num_workers=1,
                                 shuffle=True,
                                 drop_last=False)

        # Sample images
        solver = SiameseGanSolver(config, data_loader)
        solver.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='../models')
    parser.add_argument('--sample_path', type=str, default='../samples')
    parser.add_argument('--image_path', type=str, default='../../../data/" \
                        "NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--max_L2', type=int, default=5000)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true')
    parser.set_defaults(tensorboard=False)

    config = parser.parse_args()
    main()
