import unittest
import numpy as np
from utils import gaussian_mixture, add_outliers


class MyTestCase(unittest.TestCase):

    def test_gaussian_mixture(self):
        X = gaussian_mixture(n_samples=100, n_clusters=4,
                             n_outliers=10, n_features=2,
                             means=np.array([[1, 1],
                                             [1, -1],
                                             [-1, 1],
                                             [-1, -1]]),
                             outliers_dist_factor=100)
        assert X.shape == (100, 2)
        norms = np.sort(np.linalg.norm(X, axis=1))
        assert norms[-10] > 10 * norms[-11]

    def test_add_outliers(self):
        X = gaussian_mixture(n_samples=100, n_clusters=4,
                             n_outliers=0, n_features=2,
                             means=np.array([[1, 1],
                                             [1, -1],
                                             [-1, 1],
                                             [-1, -1]]))
        assert X.shape == (100, 2)

        X, outlier_idxs = add_outliers(X, n_outliers=10, dist_factor=100, return_index=True)
        assert X.shape == (110, 2)
        norms = np.sort(np.linalg.norm(X, axis=1))
        assert norms[-10] > 10 * norms[-11]


if __name__ == '__main__':
    unittest.main()
