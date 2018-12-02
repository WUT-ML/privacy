# coding=utf-8
"""Deep convolutional GAN with siamese discriminator."""

import argparse
import os

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from deterministic_data_loader import TripletFERG, FERGDataset, TripletCelebA
from generator_plus_siamese_solver import SiameseGanSolver
import FERGUtils
import CelebAUtils


def main():
    """Entry point for GAN with siamese discriminator (training or sampling)."""
    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.manual_seed(20180124)

    # Load and trasform dataset
    dataset_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])

    # If an error occurred while downloading or unzipping the dataset,
    # you might have to remove the directory with faulty files: data/{dataset}

    image_path = os.path.join("data", config.dataset)
    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.generate_path):
        os.makedirs(config.generate_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    # Download and unzip the dataset if not already there
    if config.dataset == "FERG":
        FERGUtils.get_dataset(image_path, config.dataset)
    elif config.dataset == "CelebA":
        CelebAUtils.get_dataset(image_path, config.dataset)
    else:
        print("dataset '" + config.dataset + "' unsupported")
        exit(1)

    # Train and sample the images
    if config.mode == 'train':

        if config.dataset == "FERG":
            dataset = TripletFERG(transform=dataset_transform, path=image_path)
        elif config.dataset == "CelebA":
            dataset = TripletCelebA(transform=dataset_transform, path=image_path)

        # Prepare data loader for dataset
        data_loader = DataLoader(dataset=dataset, batch_size=config.batch, num_workers=config.jobs,
                                 shuffle=True)

        # Train neural network
        solver = SiameseGanSolver(config, data_loader)
        solver.train()

    elif config.mode == 'generate':

        if config.dataset == "FERG":
            dataset = FERGDataset(transform=dataset_transform, path=image_path)
        elif config.dataset == "CelebA":
            # dataset = CelebADataset(transform=dataset_transform, path=image_path)
            exit(1111)

        # Prepare data loader for dataset
        data_loader = DataLoader(dataset=dataset, num_workers=1, shuffle=True)

        # Generate images
        solver = SiameseGanSolver(config, data_loader)
        solver.generate()

    elif config.mode == 'evaluate_privacy':

        if config.dataset == "FERG":
            dataset = TripletFERG(transform=dataset_transform, path=image_path, is_evaluation=True)
        elif config.dataset == "CelebA":
            dataset = TripletCelebA(transform=dataset_transform, path=image_path, is_evaluation=True)

        # Prepare data loader for dataset
        data_loader = DataLoader(dataset=dataset, batch_size=config.batch, num_workers=config.jobs,
                                 shuffle=True)

        # Generate images
        solver = SiameseGanSolver(config, data_loader)
        solver.check_discriminator_accuracy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--generate_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='CelebA')
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--distance_weight', type=float, default=1.0)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true')
    parser.set_defaults(tensorboard=True)

    config = parser.parse_args()
    if config.model_path == '':
        config.model_path = os.path.join(
            'results', config.dataset, 'models', str(config.distance_weight))
    if config.generate_path == '':
        config.generate_path = os.path.join(
            'results', config.dataset, 'samples', str(config.distance_weight))
    if config.dataset == 'CelebA' and config.num_epochs == 80:
        config.num_epochs = 340

    main()
