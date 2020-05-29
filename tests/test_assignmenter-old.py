import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_almost_equal
from sklearn.metrics import pairwise_distances
from okzm.assignmenter import DistanceMatrix, Assignmenter, _best_client_set, SimpleAssignmenter


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

        # check that the clip parameter works
        dmat = DistanceMatrix(C, F, p=0.1)
        assert np.all(dmat.distances(np.arange(len(C)), np.arange(len(F)), pairwise=True) <= 0.1)
        dmat.set_clip(0.2)
        assert np.all(dmat.distances(np.arange(len(C)), np.arange(len(F)), pairwise=True) <= 0.2)
        assert np.max(dmat.distances(np.arange(len(C)), np.arange(len(F)), pairwise=True)) >= 0.2
        dmat.set_clip(None)
        assert np.max(dmat.distances(np.arange(len(C)), np.arange(len(F)), pairwise=True)) >= 1

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

    def test_best_client_set(self):
        # TODO: finish this
        pass

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
        active = [0, 1]
        conn = np.array([4, 2, -1])

        #TODO: add test for initialization param assignment=None

        asgntr = Assignmenter(C=C, F=F,
                              opened_facilities_idxs=opened,
                              arrived_clients_idxs=active,
                              assignment=conn)

        # check properties
        assert_array_equal(asgntr.clients, C)
        assert_array_equal(asgntr.facilities, F)
        assert asgntr.opened_idxs == {2, 4}
        assert asgntr.closed_idxs == {0, 1, 3, 5}
        assert_array_equal(asgntr.active_client_idxs, active)
        assert_array_equal(asgntr.assignment, conn)
        assert_almost_equal(asgntr.cost, 1.2 + np.sqrt(1 + 1.21), decimal=5)

        # check cached info of nearest facilities
        idxs, dists = asgntr.nearest_facility([0, 1])
        assert_array_equal(idxs, [2, 2])
        assert_array_almost_equal(dists, [1.1, np.sqrt(1.21 + 1)], decimal=5)

        # check that we can swap facility 2 with facility 1 (lazily)
        reassigned, new_assignment, new_dist = asgntr.can_swap(swap_out=[2], swap_in=[1],
                                                               cost_thresh=0.5, avg_cost_thresh=0.5,
                                                               lazy=True)
        assert_array_equal(reassigned, [1])  # client 1 will be reassigned
        assert_array_equal(new_assignment, [1])  # The client will be reassigned to facility 1
        assert_array_equal(new_dist, [0])

        # check that we can swap facility 2 with facility 1 (non-lazily)
        reassigned, new_assignment, new_dist = asgntr.can_swap(swap_out=[2], swap_in=[1],
                                                               cost_thresh=0.5, avg_cost_thresh=0.5,
                                                               lazy=False)
        assert_array_equal(reassigned, [0, 1])  # client 1 will be reassigned
        assert_array_equal(new_assignment, [1, 1])  # The client will be reassigned to facility 1
        assert_array_equal(new_dist, [1, 0])

        # check that we can swap facility [2, 4] with facility [0, 1]
        reassigned, new_assignment, new_dist = asgntr.can_swap(swap_out=[2, 4], swap_in=[0, 1],
                                                               cost_thresh=0.5, avg_cost_thresh=0.5)
        assert_array_equal(reassigned, [0, 1])  # client 0 and 1 will be reassigned
        assert_array_equal(new_assignment, [0, 1])  # client 0 assigned to facility 0, and 1 to 1
        assert_array_equal(new_dist, [0.1, 0])

        # check that we can swap facility 2 with facility 1 (non-lazily)
        reassigned, new_assignment, new_dist = asgntr.can_swap(swap_out=[2, 4], swap_in=[0, 1],
                                                               cost_thresh=0.5, avg_cost_thresh=0.5,
                                                               lazy=False)
        assert_array_equal(reassigned, [0, 1])  # client 1 will be reassigned
        assert_array_equal(new_assignment, [0, 1])  # The client will be reassigned to facility 1
        assert_array_equal(new_dist, [0.1, 0])

        # check that we can't swap facility 2 with facility 1 given too high cost thresh
        reassigned, new_assignment, new_dist = asgntr.can_swap(swap_out=[2], swap_in=[1],
                                                               cost_thresh=2, avg_cost_thresh=0.5)
        assert reassigned is None
        assert new_assignment is None
        assert new_dist is None

        # check that we can't swap facility 2 with facility 1 given too high cost-per-recourse thresh
        reassigned, new_assignment, new_dist = asgntr.can_swap(swap_out=[2], swap_in=[1],
                                                               cost_thresh=0.5, avg_cost_thresh=1)
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
        active = [0, 1]
        conn = np.array([4, 2, -1])

        asgntr = Assignmenter(C=C, F=F,
                              opened_facilities_idxs=opened,
                              arrived_clients_idxs=active,
                              assignment=conn)

        # new client assigned to nearest open facility
        asgntr.arrive(2)
        assert_array_equal(asgntr.active_client_idxs, [0, 1, 2])
        assert_array_equal(asgntr.assignment, [4, 2, 2])  # the nearest open facility of client 2 is facility 2
        assert_almost_equal(asgntr.cost, 1.2 + np.sqrt(1 + 1.21) + np.sqrt(1 + 0.01), decimal=5)

        # check cached info of nearest facilities is updated accordingly
        idxs, dists = asgntr.nearest_facility([0, 1, 2])
        assert_array_equal(idxs, [2, 2, 2])
        assert_array_almost_equal(dists, [1.1, np.sqrt(1.21 + 1), np.sqrt(1 + 0.01)], decimal=5)

        # check that we can swap facility [2, 4] with facility [0, 1]
        reassigned, new_assignment, new_dist = asgntr.can_swap(swap_out=[2, 4], swap_in=[0, 1],
                                                               cost_thresh=0.5, avg_cost_thresh=0.5)
        asgntr.swap(swap_out=[2, 4], swap_in=[0, 1], reassigned=reassigned,
                    new_assignment=new_assignment, new_cost_vec=new_dist)
        assert_array_equal(asgntr.assignment, [0, 1, 1])
        assert asgntr.opened_idxs == {0, 1}
        assert asgntr.closed_idxs == {2, 3, 4, 5}
        assert_almost_equal(asgntr.cost, 0.1 + 0 + 1, decimal=5)

        # check cached info of nearest facilities is updated accordingly
        idxs, dists = asgntr.nearest_facility([0, 1, 2])
        assert_array_equal(idxs, [0, 1, 1])
        assert_array_almost_equal(dists, [0.1, 0, 1], decimal=5)

    def test_SimpleAssignmenter_static(self):
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

        asgntr = SimpleAssignmenter(C=C, F=F,
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

    def test_SimpleAssignmenter_dynamic(self):
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

        asgntr = SimpleAssignmenter(C=C, F=F,
                                    opened_facilities_idxs=opened,
                                    next_client=next)

        asgntr.arrive(next)
        assert_array_equal(asgntr.active_client_idxs, [0, 1, 2])
        assert_array_equal(asgntr.assignment, [2, 2, 2])  # the nearest open facility of client 2 is facility 2
        assert_almost_equal(asgntr.cost, 1.1 + np.sqrt(1 + 1.21) + np.sqrt(1 + 0.01), decimal=5)

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


if __name__ == '__main__':
    unittest.main()
