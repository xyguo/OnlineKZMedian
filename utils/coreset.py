import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import k_means
from .kzmeans import kz_means


def coreset(X, size, n_seeds=20, n_outliers=0):
    """
    Given X, compute a coreset for X
    :param X: array of shape=(n_samples, n_features), data set
    :param size: int, size of the coreset
    :param n_seeds: int, number of pre-clustering centers used for computing the coreset
    :return C, sample_weights, sample_indices:
        C: array of shape=(size, n_features), the coreset
        sample_weights: array of shape=(size,), weight of each core in the coreset
        sample_indices: array of shape=(size,), index in X of each core
    """
    n_samples, _ = X.shape
    if size >= n_samples:
        return X.copy(), np.ones(n_samples), np.arange(n_samples)

    if n_outliers > 0:
        init_centers = kz_means(X, n_clusters=n_seeds, n_outliers=n_outliers)
    else:
        init_centers, _, _ = k_means(X, n_clusters=n_seeds, return_n_iter=False)
    _, dists = pairwise_distances_argmin_min(X, init_centers, axis=1)

    # Truncate the distance of outliers to 0
    if n_outliers > 0:
        ignored = np.argpartition(-dists, kth=n_outliers)[:n_outliers]
        dists[ignored] = 0

    # calculate sensitivity of each point (actually an upper bound)
    sensitivity = dists
    total_sensitivity = sensitivity.sum()
    sampling_prob = sensitivity / total_sensitivity

    indices = np.arange(n_samples)
    coreset_idxs = np.random.choice(indices, size=size, replace=True,
                                    p=sampling_prob)
    samples = X[coreset_idxs]
    weights = np.ones(sampling_prob.shape) * np.inf
    weights[coreset_idxs] = 1 / sampling_prob[coreset_idxs]
    sample_weights = weights[coreset_idxs]
    sample_indices = coreset_idxs

    return samples, sample_weights, sample_indices
