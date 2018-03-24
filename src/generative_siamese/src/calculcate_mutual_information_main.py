"""Entry point for calculation of mutual information of image dataset."""
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from deterministic_data_loader import TrFingerprints
from dimension_reduction import PcaTsne
from mutual_information_estimator import MutualInfoEstimator
import numpy as np


def main():
    """Entry point."""
    # Load and trasnform dataset
    dataset_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=config.image_size),
        transforms.ToTensor()
    ])

    dataset = TrFingerprints(transform=dataset_transform, path=config.image_path)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=config.batch,
                             num_workers=config.jobs,
                             shuffle=True,
                             drop_last=False)

    # Use PCA and t-SNE to reduce dimensionality of dataset
    pca_tsne = PcaTsne(pca_components=config.pca_components,
                       tsne_perplexity=config.tsne_perplexity,
                       tsne_iter=config.tsne_iter,
                       tsne_components=config.tsne_components)
    data_reduced, labels = pca_tsne.reduce_dim(data_loader)

    # Estimate mutual entropy
    mi_estimator = MutualInfoEstimator()
    mi = mi_estimator.estimate_mutual_information(data_reduced, labels)
    print(mi)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=192)
    parser.add_argument('--image_path',
                        type=str,
                        default='../../../data' +
                        '/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--batch', type=int, default=1)

    # PCA and tSNE parameters
    parser.add_argument('--pca_components', type=int, default=50)
    parser.add_argument('--tsne_components', type=int, default=3)
    parser.add_argument('--tsne_perplexity', type=int, default=100)
    parser.add_argument('--tsne_iter', type=int, default=250)

    config = parser.parse_args()
    main()
