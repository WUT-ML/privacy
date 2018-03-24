"""Reduce dimensionality of images using PCA then t-SNE algorithms."""
import torchvision.transforms as transforms
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class PcaTsne():
    """Class responsible for reducing dimensionality of image dataset."""

    def __init__(self, pca_components=50, tsne_perplexity=100, tsne_iter=1000, tsne_components=3):
        """Initialize parameters of PCA and t-SNE."""
        self.pca_components = pca_components
        self.tsne_perplexity = tsne_perplexity
        self.tsne_iter = tsne_iter
        self.tsne_components = tsne_components

    def reduce_dim(self, data_loader):
        """Reduce dimensionality of image dataset."""
        # Load images and labels
        dataset = []
        labels = []
        for label, *images in data_loader:

            # Concatenate and flatten images
            merged_images = np.concatenate(images)
            dataset.append(merged_images.flatten())

            labels.append(label[0][0])

        # Use numpy array
        dataset = np.array(dataset)
        labels = np.array(labels)

        # Run PCA algorihtm
        pca = PCA(n_components=self.pca_components)
        pca_results = pca.fit_transform(dataset)

        # Run t-SNE algorithm
        tsne = TSNE(n_components=self.tsne_components,
                    verbose=0,
                    perplexity=self.tsne_perplexity,
                    n_iter=self.tsne_iter)
        tsne_results = tsne.fit_transform(pca_results)

        return tsne_results, labels
