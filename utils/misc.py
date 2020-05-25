import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array
from sklearn.metrics import pairwise_distances_argmin_min


def debug_print(s, debug=True):
    if debug:
        print(s)


def compute_cost(X, cluster_centers, cost_func, remove_outliers=None):
    """
    :param X: array,
        data set
    :param remove_outliers: None or int, default None
        whether to remove outliers when computing the cost on X
    :return: float,
        actual cost
    """
    if cluster_centers is None:
        raise NotFittedError("Model hasn't been fitted yet\n")
    X = check_array(X, ensure_2d=True)
    _, dists = pairwise_distances_argmin_min(X, cluster_centers, axis=1)
    dist_idxs = np.argsort(dists)
    if remove_outliers is not None:
        assert remove_outliers >= 0
        dist_idxs = dist_idxs if remove_outliers == 0 else dist_idxs[:-int(remove_outliers)]

        return cost_func(X[dist_idxs], cluster_centers)
    else:
        return cost_func(X, cluster_centers)



