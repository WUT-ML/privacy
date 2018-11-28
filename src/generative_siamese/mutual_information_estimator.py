# coding=utf-8
"""Estimate mutual information for image datasets."""
import numpy as np


class MutualInfoEstimator():
    """Estimation of mutual information for images using nearest neighbour estimator for entropy."""

    def estimate_mutual_information(self, dataset, labels):
        """Estimate mutual information."""
        """I(X;Y) = I(X) - I(X|Y)"""

        # Estimate I(X)
        entropy_x = self.estimate_entropy(self.distance_nearest_neighbour(dataset))

        # Estimate I(X|Y)
        entropy_x_cond_y = 0.0

        # For each y belongs to Y ...
        for unique_label in np.unique(labels):
            # ... get a subset of dataset given one label
            subset = dataset[(labels == unique_label).nonzero()]
            # Estimate entropy
            entropy_x_cond_y += self.estimate_entropy(self.distance_nearest_neighbour(subset))

        return entropy_x - entropy_x_cond_y

    def estimate_entropy(self, distances, dim=2):
        """Estimate entropy using distances to nearest neighbour."""
        n_examples = len(distances)
        return np.sum((np.log2((n_examples - 1) * (distances**dim))))

    def distance_nearest_neighbour(self, array, distance=np.linalg.norm):
        """Calculate distances to nearest neighbours for array of points."""
        distances = []
        for idx, _ in enumerate(array):
            # Do not take distance to self, append second nearest distance
            distances.append(np.partition(np.array([distance(x) for x in array-array[idx]]), 1)[1])
        return np.array(distances)
