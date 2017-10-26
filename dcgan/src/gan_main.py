"""Main module for running deep conv GAN training and sampling."""
import argparse
import os
from gan_solver import GanSolver
from data_loader import get_loader
from torch.backends import cudnn


def main():
    """Entry point for GAN training or sampling."""
    cudnn.benchmark = True

    data_loader = get_loader(image_size=config.image_size, image_path=config.image_path)

    solver = GanSolver(config, data_loader)

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'sample':
        solver.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='../models')
    parser.add_argument('--sample_path', type=str, default='../samples')
    parser.add_argument('--image_path', type=str, default='../data/train')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_only_last_model', type=bool, default=False)

    config = parser.parse_args()
    main()
