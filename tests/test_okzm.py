import unittest
import numpy as np
from sklearn.metrics import pairwise_distances
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_almost_equal
from okzm.okzm import OnlineKZMed, _trivial_k_median


class TestOKZM(unittest.TestCase):
    def test_toy_data(self):
        C = np.array([[0, 0],  # client 0
                      [0, 1],  # client 1
                      [1, 1],  # client 2
                      [1, 0],  # client 3
                      [1, -1],  # client 4
                      [0, -1],  # client 5
                      [-1, -1],  # client 6
                      [-1, 0],  # client 7
                      [-1, 1]])  # client 8

        F = np.array([[-0.1, 0],  # facility 0
                      [0, 1],  # facility 1
                      [1.1, 0],  # facility 2
                      [1, 1],  # facility 3
                      [-1.2, 0],  # facility 4
                      [0, 1.3]])  # facility 5
        n_clusters = 2
        n_outliers = 2
        epsilon = 0.1
        gamma = 1
        ell = 1
        dist_mat = pairwise_distances(C, F)
        okzm = OnlineKZMed(n_clusters, n_outliers=n_outliers,
                           epsilon=epsilon, gamma=gamma, ell=ell)

        okzm.fit(C, F, distances=dist_mat)


    def test_random_data(self):
        pass


if __name__ == '__main__':
    unittest.main()
