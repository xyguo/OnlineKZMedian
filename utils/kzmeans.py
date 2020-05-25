# -*- coding: utf-8 -*-
"""Algorithm for (k,z)-means and k-means"""


import numpy as np

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances_argmin_min
from sklearn.utils import check_array
from utils import compute_cost


def kzmeans_cost_(X, C, sample_weights=None, n_outliers=0, L=None, element_wise=False):
    """
    :param X: array of shape=(n_samples, n_features), data set
    :param C: array of shape=(n_centers, n_features), centers
    :param sample_weights: array of shape=(n_samples,), sample weights
    :param n_outliers: int, number of outliers
    :param L: None or float. if not None then all distances larger than L will be truncated
    :param element_wise: bool, whether to return the cost for each element in X
    :return:
    """
    X = check_array(X, ensure_2d=True)
    C = check_array(C, ensure_2d=True)
    _, dists = pairwise_distances_argmin_min(X, C, axis=1, metric='euclidean',
                                             metric_kwargs={'squared': True})
    if L is None:
        if sample_weights is not None:
            dists *= sample_weights
        if element_wise:
            return dists
        dists.sort()
        return sum(dists[:-n_outliers]) if n_outliers > 0 else np.sum(dists)
    else:
        np.minimum(dists, L, out=dists)
        if sample_weights is not None:
            dists *= sample_weights
        if element_wise:
            return dists
        dist = dists.sum() - n_outliers * L
        return dist


def pairwise_squared_dist_(X, C):
    return euclidean_distances(X=X, Y=C, squared=True)


def kmeans_pp_(X, sample_weights, n_clusters):
    """
    (weighted) k-means++ initialization

    :param X: array of shape=(n_samples, n_features)
    :param sample_weights: array of shape=(n_samples,)
    :param n_clusters:
    :return centers: array of shape=(n_clusters, n_features)
    """
    n_samples, _ = X.shape
    first_idx = np.random.randint(0, n_samples)
    centers = [X[first_idx]]
    for i in range(n_clusters - 1):
        _, dist = pairwise_distances_argmin_min(X, centers)
        probs = normalize(((dist ** 2) * sample_weights).reshape(1, -1), norm='l1')[0]
        next_idx = np.random.choice(n_samples, 1, replace=True, p=probs)[0]
        centers.append(X[next_idx])
    return np.array(centers)


def update_clusters_(X, centers, return_dist=False):
    """

    :param X:
    :param centers:
    :param return_dist: whether to return the distances of each point to its nearest center
    :return clusters: list of arrays, each array consists of the indices
        for data in the same cluster. If some cluster has size less than 2 then
        it is ignored.

    """
    n_centers = len(centers)
    idxs, dists = pairwise_distances_argmin_min(X, centers)
    clusters = []
    # dists_collected = []
    for i in range(n_centers):
        clusters.append(np.where(idxs == i)[0])
        # dists_collected.append(dists[clusters[-1]])
    clusters = [c for c in clusters if len(c) >= 1]
    return (clusters, dists) if return_dist else clusters


def update_centers_(X, sample_weights, clusters, outliers=None):
    """

    :param X:
    :param sample_weights:
    :param clusters:
    :return centers:
    """
    centers = []
    for c in clusters:
        if outliers is not None:
            c = list(set(c).difference(outliers))
        if len(c) > 0:
            centers.append(np.average(X[c], axis=0, weights=sample_weights[c]))
    return np.array(centers)


def kmeans_(X, sample_weights, n_clusters, init='kmeans++', max_iter=300):
    """
    Weighted K-Means implementation (Lloyd's Algorithm).
    :param X:
    :param sample_weights:
    :param n_clusters:
    :param init: string in {'random', 'kmeans++'}, default 'kmeans++'
    :param max_iter: maximum number of iterations
    :return cluster_centers_:
    """
    n_samples, n_features = X.shape
    # TODO: find a better way to handle negtive weights

    cluster_centers_ = None
    if init == 'kmeans++':
        cluster_centers_ = kmeans_pp_(X, np.clip(sample_weights, 0, np.inf), n_clusters)
    elif init == 'random':
        centers_idxs = np.random.choice(n_samples, n_clusters, replace=False)
        cluster_centers_ = X[centers_idxs]
    elif isinstance(init, np.ndarray):
        cluster_centers_ = init

    diff = np.inf
    i = 0
    while diff > 1e-3 and i < max_iter:
        clusters = update_clusters_(X, cluster_centers_)
        new_centers = update_centers_(X, sample_weights, clusters)
        if len(new_centers) == len(cluster_centers_):
            diff = np.linalg.norm(new_centers - cluster_centers_)
        cluster_centers_ = new_centers
        i += 1

    # if the program finishes before finding k'<k centers, we use the FarthestNeighbor
    # method to produce the remained k-k' centers
    if len(cluster_centers_) < n_clusters:
        centers = [c for c in cluster_centers_]
        _, dists_to_centers = pairwise_distances_argmin_min(X, np.atleast_2d(centers))

        for i in range(0, n_clusters - len(cluster_centers_)):
            next_idx = np.argmax(dists_to_centers)
            centers.append(X[next_idx])
            _, next_dist = pairwise_distances_argmin_min(X, np.atleast_2d(centers[-1]))
            dists_to_centers = np.minimum(dists_to_centers, next_dist)
        cluster_centers_ = np.array(centers)

    return cluster_centers_


def k_means_lloyd(X, n_clusters, sample_weights=None):
    """K-Means by Lloyd's Algorithm"""
    n_samples, _ = X.shape
    if sample_weights is None:
        sample_weights = np.ones(n_samples)
    return kmeans_(X=X, sample_weights=sample_weights,
                   n_clusters=n_clusters)


def kmeans_mm_(X, sample_weights, n_clusters, n_outliers, init='kmeans++', max_iter=300):
    """
    Weighted K-Means-- implementation.

    Sanjay Chawla and Aristides Gionis.
    k-means−−: A unified approach to clustering and outlier detection.
    In Proceedings of the 13th SIAM International Conference on Data Mining, 2013.
    :param X:
    :param sample_weights:
    :param n_clusters:
    :param n_outliers:
    :param init: string in {'random', 'kmeans++'}, default 'kmeans++'
    :param max_iter: maximum number of iterations
    :return cluster_centers_:
    """
    n_samples, n_features = X.shape
    # TODO: find a better way to handle negtive weights
    cluster_centers_ = None
    if init == 'kmeans++':
        cluster_centers_ = kmeans_pp_(X, np.clip(sample_weights, 0, np.inf), n_clusters)
    elif init == 'random':
        centers_idxs = np.random.choice(n_samples, n_clusters, replace=False)
        cluster_centers_ = np.atleast_2d(X[centers_idxs])
    elif isinstance(init, np.ndarray):
        cluster_centers_ = init

    # centers_idxs = np.random.choice(n_samples, n_clusters, replace=False)
    # cluster_centers_ = X[centers_idxs]

    diff = np.inf
    i = 0
    while diff > 1e-3 and i < max_iter:
        clusters, dists = update_clusters_(X, cluster_centers_, return_dist=True)

        # ignore the outliers when updating centers
        if n_outliers > 0:
            outliers = np.argsort(dists)[-n_outliers:]
        else:
            outliers = None
        new_centers = update_centers_(X, sample_weights, clusters, outliers=outliers)
        if len(new_centers) == len(cluster_centers_):
            diff = np.linalg.norm(new_centers - cluster_centers_)
        cluster_centers_ = new_centers
        i += 1

    # if the program finishes before finding k'<k centers, we use the FarthestNeighbor
    # method to produce the remained k-k' centers
    if len(cluster_centers_) < n_clusters:
        centers = [c for c in cluster_centers_]
        _, dists_to_centers = pairwise_distances_argmin_min(X, np.atleast_2d(centers))

        for i in range(0, n_clusters - len(cluster_centers_)):
            # next_idx = np.argmax(dists_to_centers)
            # Pick the (n_outliers + 1)-th farthest point as the new center
            far_to_nearest = np.argsort(dists_to_centers)[::-1]
            cum_weights = np.cumsum(sample_weights[far_to_nearest])
            next_idx = far_to_nearest[np.searchsorted(cum_weights, n_outliers)+1]
            ###
            centers.append(X[next_idx])
            _, next_dist = pairwise_distances_argmin_min(X, np.atleast_2d(centers[-1]))
            dists_to_centers = np.minimum(dists_to_centers, next_dist)
        cluster_centers_ = np.array(centers)

    return cluster_centers_


def kz_means(X, n_clusters, n_outliers, sample_weights=None):
    """ K-Means-- """
    n_samples, _ = X.shape
    if sample_weights is None:
        sample_weights = np.ones(n_samples)
    return kmeans_mm_(X=X, sample_weights=sample_weights, init='random',
                      n_clusters=n_clusters, n_outliers=n_outliers)


def kmeans_cost_no_outlier_(X, C, sample_weights=None, n_outliers=0, L=None, element_wise=False):
    return kzmeans_cost_(X, C, sample_weights=sample_weights,
                         n_outliers=0, L=None, element_wise=element_wise)


class KZMeans(object):
    """ A wrapper for the kz_means function to support sklearn interface """
    def __init__(self, n_clusters, n_outliers=0):
        self.n_clusters_ = n_clusters
        self.n_outliers_ = n_outliers
        self.cluster_centers_ = None
        self.cost_func_ = kzmeans_cost_

    def cost(self, X, remove_outliers=True):
        """

        :param X: array of shape=(n_samples, n_features),
            data set
        :param remove_outliers: None or int, default None
            whether to remove outliers when computing the cost on X
        :return: float,
            actual cost
        """
        return compute_cost(X, cluster_centers=self.cluster_centers_,
                            cost_func=kzmeans_cost_,
                            remove_outliers=remove_outliers)

    def fit(self, X):
        self.cluster_centers_ = kz_means(X, self.n_clusters_, self.n_outliers_)
        return self

    def predict(self, X):
        nearest, _ = pairwise_distances_argmin_min(X, self.cluster_centers_)
        return nearest

