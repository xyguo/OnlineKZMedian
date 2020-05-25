import unittest
import numpy as np
from utils import coreset


class MyTestCase(unittest.TestCase):

    def test_coreset(self):
        X = np.random.multivariate_normal(np.zeros(2), np.identity(2), size=1000)

        C, sw, sidxs = coreset(X, size=50, n_seeds=20, n_outliers=0)
        assert C.shape == (50, 2)
        assert np.all(sw > 0)
        assert sidxs.shape == (50,)

        C, sw, sidxs = coreset(X, size=50, n_seeds=20, n_outliers=5)
        assert C.shape == (50, 2)
        assert sidxs.shape == (50,)

