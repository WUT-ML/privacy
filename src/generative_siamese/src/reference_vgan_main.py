"""Privacy-Preserving Representation-Learning Variational Generative Adversarial Network."""

import argparse
import torch
from torch.backends import cudnn
from reference_data_loader import FERGDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from reference_vgan_solver import SiameseVganSolver


def main():
    """Entry point for VGAN with siamese discriminator (training or sampling)."""
    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.manual_seed(20180521)

    # Define image transformation
    dataset_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    # Prepare dataset loader
    dataset = FERGDataset(transform=dataset_transform, path=config.image_path)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=config.batch_size,
                             num_workers=config.jobs,
                             shuffle=True,
                             drop_last=False)

    if config.mode == 'train':
        # Train model
        solver = SiameseVganSolver(config, data_loader)
        solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--n_ids', type=int, default=6)
    parser.add_argument('--n_attrs', type=int, default=7)
    parser.add_argument('--siam_factor', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--image_path', type=str, default="../../../../FERG_DB_256/")
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=2)

    config = parser.parse_args()
    main()
