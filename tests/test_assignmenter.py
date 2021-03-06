import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_almost_equal
from sklearn.metrics import pairwise_distances
from okzm.assignmenter import DistanceMatrix, Assignmenter


class TestAssignmenter(unittest.TestCase):

    def test_DistanceMatrix(self):
        C = np.array([[0, 0],   # client 0
                      [0, 1],  # client 1
                      [1, 1]]) # client 2
        F = np.array([[-0.1, 0],  # facility 0
                      [0, 1],  # facility 1
                      [1.1, 0],  # facility 2
                      [1, 1],  # facility 3
                      [-1.2, 0],  # facility 4
                      [0, 1.3]])  # facility 5
        cs = np.array([0, 2])
        fs_long = np.array([0, 2, 3, 5])
        fs_short = np.array([2, 3])
        dm = pairwise_distances(C, F)

        # check when distances are computed realtime
        dmat = DistanceMatrix(C, F)
        assert_array_almost_equal(dmat.distances(cs, fs_long, pairwise=True), dm[np.ix_(cs, fs_long)], decimal=5)
        assert_array_almost_equal(dmat.distances(cs, fs_short, pairwise=False), dm[(cs, fs_short)], decimal=5)

        fi1, mdist1 = dmat.pairwise_dist_argmin_min(cs, fs_long)
        fi2 = fs_long[np.argmin(dm[np.ix_(cs, fs_long)], axis=1)]
        mdist2 = dm[(cs, fi2)]
        assert_array_equal(fi1, fi2)
        assert_array_almost_equal(mdist1, mdist2, decimal=5)

        # check when distances are computed with cached distance matrix
        dmat = DistanceMatrix(C, F, dist_mat=dm)
        assert_array_almost_equal(dmat.distances(cs, fs_long, pairwise=True), dm[np.ix_(cs, fs_long)], decimal=5)
        assert_array_almost_equal(dmat.distances(cs, fs_short, pairwise=False), dm[(cs, fs_short)], decimal=5)

        fi1, mdist1 = dmat.pairwise_dist_argmin_min(cs, fs_long)
        fi2 = fs_long[np.argmin(dm[np.ix_(cs, fs_long)], axis=1)]
        mdist2 = dm[(cs, fi2)]
        assert_array_equal(fi1, fi2)
        assert_array_almost_equal(mdist1, mdist2, decimal=5)

    def test_Assignmenter_static(self):
        C = np.array([[0, 0],   # client 0
                      [0, 1],  # client 1
                      [1, 1]]) # client 2
        F = np.array([[-0.1, 0],  # facility 0
                      [0, 1],  # facility 1
                      [1.1, 0],  # facility 2
                      [1, 1],  # facility 3
                      [-1.2, 0],  # facility 4
                      [0, 1.3]])  # facility 5
        opened = {2, 4}
        next_j = 2
        conn = np.array([2, 2, -1])

        asgntr = Assignmenter(C=C, F=F,
                              opened_facilities_idxs=opened,
                              next_client=next_j)

        # check properties
        assert_array_equal(asgntr.clients, C)
        assert_array_equal(asgntr.facilities, F)
        assert asgntr.opened_idxs == {2, 4}
        assert asgntr.closed_idxs == {0, 1, 3, 5}
        assert_array_equal(asgntr.active_client_idxs, np.arange(next_j))
        assert_array_equal(asgntr.assignment, conn)
        assert_almost_equal(asgntr.cost, 1.1 + np.sqrt(1 + 1.21), decimal=5)
        assert_almost_equal(asgntr.cost_p(p=1, remove=False), 1 + 1, decimal=5)
        assert_almost_equal(asgntr.cost_p(p=1.2, remove=True), 1.1, decimal=5)
        assert_almost_equal(asgntr.cost_z(0), asgntr.cost, decimal=5)
        assert_almost_equal(asgntr.cost_z(1), 1.1, decimal=5)
        assert_almost_equal(asgntr.cost_z(2), 0, decimal=5)
        assert_almost_equal(asgntr.cost_z(4), 0, decimal=5)

        # check the info of nearest facilities matches
        idxs, dists = asgntr.nearest_facility([0, 1])
        assert_array_equal(idxs, [2, 2])
        assert_array_almost_equal(dists, [1.1, np.sqrt(1.21 + 1)], decimal=5)

        # check that we can swap facility 2 with facility 1
        swap_out, reassigned, new_assignment, new_dist = asgntr.can_swap(swap_in=1, swap_out=[2],
                                                                         cost_thresh=0.5, avg_cost_thresh=0.5)
        assert swap_out == 2
        assert_array_equal(reassigned, [0, 1])  # client 1 will be reassigned
        assert_array_equal(new_assignment, [1, 1])  # The client will be reassigned to facility 1
        assert_array_equal(new_dist, [1, 0])

        # check that we can't swap facility 2 with facility 1 given too high cost thresh
        swap_out, reassigned, new_assignment, new_dist = asgntr.can_swap(swap_in=1, swap_out=[2],
                                                                         cost_thresh=2, avg_cost_thresh=0.5)
        assert swap_out is None
        assert reassigned is None
        assert new_assignment is None
        assert new_dist is None

        # check that we can't swap facility 2 with facility 1 given too high cost-per-recourse thresh
        swap_out, reassigned, new_assignment, new_dist = asgntr.can_swap(swap_in=1, swap_out=[2],
                                                                         cost_thresh=0.5, avg_cost_thresh=1)
        assert swap_out is None
        assert reassigned is None
        assert new_assignment is None
        assert new_dist is None

    def test_Assignmenter_dynamic(self):
        C = np.array([[0, 0],   # client 0
                      [0, 1],  # client 1
                      [1, 1]]) # client 2
        F = np.array([[-0.1, 0],  # facility 0
                      [0, 1],  # facility 1
                      [1.1, 0],  # facility 2
                      [1, 1],  # facility 3
                      [-1.2, 0],  # facility 4
                      [0, 1.3]])  # facility 5
        opened = {2, 4}
        next = 2

        asgntr = Assignmenter(C=C, F=F,
                              opened_facilities_idxs=opened,
                              next_client=next)

        asgntr.arrive(next - 1)
        assert len(asgntr.active_client_idxs) == next
        self.assertRaises(ValueError, asgntr.arrive, next + 1)
        asgntr.arrive(next)
        assert len(asgntr.active_client_idxs) == next + 1

        assert_array_equal(asgntr.active_client_idxs, [0, 1, 2])
        assert_array_equal(asgntr.assignment, [2, 2, 2])  # the nearest open facility of client 2 is facility 2
        assert_almost_equal(asgntr.cost, 1.1 + np.sqrt(1 + 1.21) + np.sqrt(1 + 0.01), decimal=5)
        assert_almost_equal(asgntr.cost_p(p=1, remove=False), 1 + 1 + 1, decimal=5)
        assert_almost_equal(asgntr.cost_p(p=1.2, remove=True), 1.1 + np.sqrt(1 + 0.01), decimal=5)
        assert_almost_equal(asgntr.cost_z(0), asgntr.cost, decimal=5)
        assert_almost_equal(asgntr.cost_z(1), 1.1 + np.sqrt(1 + 0.01), decimal=5)
        assert_almost_equal(asgntr.cost_z(2), np.sqrt(1 + 0.01), decimal=5)
        assert_almost_equal(asgntr.cost_z(4), 0, decimal=5)

        # check cached info of nearest facilities is updated accordingly
        idxs, dists = asgntr.nearest_facility([0, 1, 2])
        assert_array_equal(idxs, [2, 2, 2])
        assert_array_almost_equal(dists, [1.1, np.sqrt(1.21 + 1), np.sqrt(1 + 0.01)], decimal=5)

        # check that we can swap facility 1 with facility [2]
        swap_out, reassigned, new_assignment, new_dist = asgntr.can_swap(swap_out=[2], swap_in=1,
                                                                         cost_thresh=0.5, avg_cost_thresh=0.25)
        asgntr.swap(swap_out=swap_out, swap_in=1, reassigned=reassigned,
                    new_assignment=new_assignment, new_cost_vec=new_dist)
        assert_array_equal(asgntr.assignment, [1, 1, 1])
        assert asgntr.opened_idxs == {4, 1}
        assert asgntr.closed_idxs == {2, 3, 0, 5}
        assert_almost_equal(asgntr.cost, 1 + 0 + 1, decimal=5)
        assert_almost_equal(asgntr.cost_p(p=0.8, remove=False), 0.8 + 0 + 0.8, decimal=5)
        assert_almost_equal(asgntr.cost_p(p=0.8, remove=True), 0, decimal=5)
        assert_almost_equal(asgntr.cost_z(0), asgntr.cost, decimal=5)
        assert_almost_equal(asgntr.cost_z(1), 1 + 0, decimal=5)
        assert_almost_equal(asgntr.cost_z(2), 0, decimal=5)
        assert_almost_equal(asgntr.cost_z(4), 0, decimal=5)

    def test_Assignmenter_F_eq_C(self):
        C = np.array([[0, 0],   # client 0
                      [0, 1],  # client 1
                      [1, 1],  # client 2
                      [-0.1, 0],  # client 0
                      [0, 1],  # client 1
                      [1.1, 0],  # facility 2
                      [1, 1],  # facility 3
                      [-1.2, 0],  # facility 4
                      [0, 1.3]])  # facility 5
        opened = {0, 1}
        next = 2

        asgntr = Assignmenter(C=C, F=C,
                              F_is_C=True,
                              opened_facilities_idxs=opened,
                              next_client=next)
        assert asgntr.opened_idxs == {0, 1}
        assert len(asgntr.closed_idxs) == 0
        assert_array_equal(asgntr.assignment[:next], [0,1])
        assert asgntr.cost == 0
        assert_array_equal(asgntr.cost_vec[:next], [0, 0])

        asgntr.arrive(next)

        assert_array_equal(asgntr.active_client_idxs, [0, 1, 2])
        assert_array_equal(asgntr.assignment[:next+1], [0, 1, 1])  # the nearest open facility of client 2 is facility 2
        assert_almost_equal(asgntr.cost, 0 + 0 + 1, decimal=5)
        assert_almost_equal(asgntr.cost_p(p=1, remove=False), 0 + 0 + 1, decimal=5)
        assert_almost_equal(asgntr.cost_p(p=1.2, remove=True), 0 + 0 + 1, decimal=5)
        assert_almost_equal(asgntr.cost_z(0), asgntr.cost, decimal=5)
        assert_almost_equal(asgntr.cost_z(1), 0 + 0, decimal=5)
        assert_almost_equal(asgntr.cost_z(2), 0, decimal=5)
        assert_almost_equal(asgntr.cost_z(4), 0, decimal=5)

        # check cached info of nearest facilities is updated accordingly
        idxs, dists = asgntr.nearest_facility([0, 1, 2])
        assert_array_equal(idxs, [0, 1, 1])
        assert_array_almost_equal(dists, [0, 0, 1], decimal=5)

        # check that we can swap facility 1 with facility [2]
        swap_out, reassigned, new_assignment, new_dist = asgntr.can_swap(swap_out=[2], swap_in=1,
                                                                         cost_thresh=0.5, avg_cost_thresh=0.25)
        assert swap_out is None


if __name__ == '__main__':
    unittest.main()
