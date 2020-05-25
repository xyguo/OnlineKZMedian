import unittest
import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from okzm.okzm import OnlineKZMed, _trivial_k_median
from utils import debug_print


def check_okzm(okzm, C, F, inlier_val, outlier_val, debugging):
    # permute C and F
    np.random.shuffle(F)
    np.random.shuffle(C)

    # obtain outlier indices and cluster center indices
    center_indices = np.where(np.abs(F).max(axis=1) < inlier_val)[0]
    is_outlier = np.abs(C).max(axis=1) > outlier_val
    _, optimal_cost = pairwise_distances_argmin_min(C[np.logical_not(is_outlier)], F[center_indices])
    optimal_cost = optimal_cost.sum()
    outlier_indices = np.where(is_outlier)[0]
    debug_print("True outliers: {}".format(outlier_indices), debugging)
    debug_print("Optimal centers: {}".format(center_indices), debugging)
    debug_print("Optimal cost (removing outliers): {}".format(optimal_cost), debugging)

    dist_mat = pairwise_distances(C, F)
    okzm.fit(C, F, distances=dist_mat)

    debug_print("Identified outliers: {}".format(okzm.outlier_indices), debugging)
    debug_print("Cluster centers: {}".format(okzm.cluster_center_indices), debugging)
    debug_print("Clustering cost: {}".format(okzm.cost), debugging)

    # check centers
    assert set(okzm.cluster_center_indices) == set(center_indices)

    # check outliers
    assert set(okzm.outlier_indices).issuperset(outlier_indices)


class TestOKZM(unittest.TestCase):

    def test_trivial_k_median(self):
        # TODO: finish this
        pass

    def test_toy_data(self):
        debugging = False
        C = np.array([[10.1, 10.1],  # 0: cluster 1
                      [-9.9, -10],  # 1: cluster 4
                      [-1000, 0],  # 2: outlier 1
                      [-10.1, 9.9],  # 3: cluster 3
                      [10.1, 9.9],  # 4: cluster 1
                      [10.1, -10.1],  # 5: cluster 2
                      [-10.1, 10.1],  # 6: cluster 3
                      [9.9, 10],  # 7: cluster 1
                      [-1, 1000],  # 8: outlier 2
                      [-9.9, 10],  # 9: cluster 3
                      [-10.1, -10.1],  # 10: cluster 4
                      [9.9, -10],  # 11: cluster 2
                      [-10.1, -9.9],  # 12: cluster 4
                      [10.1, -9.9]])  # 13: cluster 2

        F = np.array([[9.9, 10],  # 0: cluster 1
                      [10, -10],  # 1: cluster 2
                      [-9.9, 10],  # 2: cluster 3
                      [-10, -10],  # 3: cluster 4
                      [500, 0],  # 4
                      [-500, 0],  # 5
                      [0, 500],  # 6
                      [0, -500]])  # 7

        n_clusters = 4
        n_outliers = 2
        epsilon = 0.1
        gamma = 0
        ell = 2
        okzm1 = OnlineKZMed(n_clusters, n_outliers=n_outliers,
                            epsilon=epsilon, gamma=gamma, ell=ell,
                            debugging=debugging)

        check_okzm(okzm1, C, F, inlier_val=20, outlier_val=800, debugging=debugging)

        okzm2 = OnlineKZMed(n_clusters, n_outliers=n_outliers,
                            epsilon=epsilon, gamma=gamma, ell=ell,
                            random_swap_out=5, random_swap_in=20,
                            debugging=debugging)
        check_okzm(okzm2, C, F, inlier_val=20, outlier_val=800, debugging=debugging)

    def test_random_data(self):
        debugging = False
        F = np.array([[9.9, 10],  # 0: cluster 1
                      [10, -10],  # 1: cluster 2
                      [-9.9, 10],  # 2: cluster 3
                      [-10, -10],  # 3: cluster 4
                      [500, 0],  # 4
                      [-500, 0],  # 5
                      [0, 500],  # 6
                      [0, -500]])  # 7
        cov = np.identity(2) * 0.5
        clusters = [np.random.multivariate_normal(mean=F[i], cov=cov, size=50)
                    for i in range(4)]
        C = np.vstack(clusters)
        outliers = np.array([[-1000, 0],
                             [0, 1000],
                             [1000, 100],
                             [100, -1000]])
        C = np.vstack((C, outliers))

        n_clusters = 4
        n_outliers = 4
        epsilon = 0.1
        gamma = 0
        ell = 1

        okzm1 = OnlineKZMed(n_clusters, n_outliers=n_outliers,
                            epsilon=epsilon, gamma=gamma, ell=ell,
                            debugging=debugging)
        check_okzm(okzm1, C, F, inlier_val=20, outlier_val=800, debugging=debugging)

        okzm2 = OnlineKZMed(n_clusters, n_outliers=n_outliers,
                            epsilon=epsilon, gamma=gamma, ell=ell,
                            random_swap_out=5, random_swap_in=20,
                            debugging=debugging)
        check_okzm(okzm2, C, F, inlier_val=20, outlier_val=800, debugging=debugging)


if __name__ == '__main__':
    unittest.main()
