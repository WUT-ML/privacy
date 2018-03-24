"""Deep convolutional GAN with siamese discriminator."""

import argparse
import os

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from deterministic_data_loader import TrFingerprints, Fingerprints
from generator_plus_siamese_solver import SiameseGanSolver


def main():
    """Entry point for GAN with siamese discriminator (training or sampling)."""
    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.manual_seed(20180124)

    # Load and trasnform dataset
    dataset_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=config.image_size),
        transforms.ToTensor(),
    ])

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.generate_path):
        os.makedirs(config.generate_path)

    # Train and sample the images
    if config.mode == 'train':

        dataset = TrFingerprints(transform=dataset_transform, path=config.image_path)

        # Prepare data loader for dataset
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=config.batch,
                                 num_workers=config.jobs,
                                 shuffle=True,
                                 drop_last=False)
        # Train neural network
        solver = SiameseGanSolver(config, data_loader)
        solver.train()
    elif config.mode == 'generate':

        dataset = Fingerprints(transform=dataset_transform, path=config.image_path)

        # Prepare data loader for dataset
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=1,
                                 num_workers=1,
                                 shuffle=True,
                                 drop_last=False)

        # Generate images
        solver = SiameseGanSolver(config, data_loader)
        solver.generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='../models')
    parser.add_argument('--generate_path', type=str, default='../samples')
    parser.add_argument('--image_path', type=str, default='../../../data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--max_L2', type=float, default=5000.0)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true')
    parser.set_defaults(tensorboard=False)

    config = parser.parse_args()
    main()
