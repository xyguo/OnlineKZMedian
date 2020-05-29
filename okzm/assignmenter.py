# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import pairwise_distances


# TODO: rewrite the `[]` (__getitem__) operator to replace the cumbersome `distances` function
class DistanceMatrix(object):
    def __init__(self, C, F, p=None, dist_mat=None):
        """
        Mimic a distance matrix: it may compute the distance on-the-fly, but can be queried
        as a static distance matrix.
        :param C: array of shape=(n_clients, n_features)
        :param F: array of shape=(n_facilities, n_features)
        :param p: float, whether to clip distance below this threshold.
            If None, means no clipping.
        :param dist_mat: array of shape=(n_clients, n_facilities), precomputed distance matrix.
            If None, the distance will be computed on-the-fly.
        """
        self._C = C
        self._F = F
        self._dist_mat = dist_mat
        if p is not None and (p <= 0 or np.isinf(p)):
            p = None
        self._p = p

    def set_clip(self, p):
        """set new clipping threshold"""
        if p is not None and (p <= 0 or np.isinf(p)):
            p = None
        self._p = p

    def distances(self, c_idxs, f_idxs, pairwise=True):
        """
        Return pairwise distance or element-wise distance.
        :param c_idxs: array of queried client idxs in self._C
        :param f_idxs: array of queried facility idxs in self._F
        :param pairwise: if True then return an 2D array containing the pairwise distances between
            C[c_idxs] and F[f_idxs]; If False then return an 1D array containing the row-wise distances between
            C[c_idxs] and F[f_idxs] (this requires c_idxs and f_idxs has the same length)
        :return dists:
        """
        if self._dist_mat is None:
            if pairwise:
                dists = pairwise_distances(self._C[c_idxs], self._F[f_idxs])
            else:
                dists = np.linalg.norm(self._C[c_idxs] - self._F[f_idxs], axis=1)
        else:
            if pairwise:
                dists = self._dist_mat[np.ix_(c_idxs, f_idxs)]
            else:
                dists = self._dist_mat[(c_idxs, f_idxs)]
        return np.clip(dists, a_min=0, a_max=self._p) if self._p is not None else dists

    def pairwise_dist_argmin_min(self, c_idxs, f_idxs):
        """
        Return the nearest facility indices and the corresponding distance.
        :param c_idxs: array of queried client idxs in self._C
        :param f_idxs: array of queried facility idxs in self._F
        :return nearest, min_dist:
            nearest: nearest[p] = q means self._C[c_idxs[p]]'s nearest facility idx
                (among f_idxs) is q
            min_dist: min_dist[p] = d means self._C[c_idxs[p]]'s distance to its nearest facility
                (among self._F[f_idxs]) is d
        """
        # dists = self._pairwise_dist(client_idxs, facility_idxs)
        dists = self.distances(c_idxs, f_idxs, pairwise=True)
        nearest = np.argmin(dists, axis=1)
        min_dist = dists[(np.arange(dists.shape[0]), nearest)]

        return f_idxs[nearest], min_dist

    def pairwise_dist_kth_nearest(self, c_idxs, f_idxs, k):
        """
        Return the nearest facility indices and the corresponding distance.
        :param c_idxs: array of queried client idxs in self._C
        :param f_idxs: array of queried facility idxs in self._F
        :param k: int, must be nonnegative, k=0 means
        :return nearest, min_dist:
            nearest: nearest[p] = q means self._C[c_idxs[p]]'s nearest facility idx
                (among f_idxs) is q
            min_dist: min_dist[p] = d means self._C[c_idxs[p]]'s distance to its nearest facility
                (among self._F[f_idxs]) is d
        """
        # dists = self._pairwise_dist(client_idxs, facility_idxs)
        assert 0 <= k < len(f_idxs)
        dists = self.distances(c_idxs, f_idxs, pairwise=True)
        kth_nearest = np.argpartition(dists, k, axis=1)[:, k]
        d = dists[(np.arange(dists.shape[0]), kth_nearest)]

        return f_idxs[kth_nearest], d


class Assignmenter(object):

    def __init__(self, C, F, opened_facilities_idxs,
                 next_client=None,
                 dist_mat=None):
        """
        Object that store a facility solution and support efficient local search operations
        :param C: array of shape=(n_clients, n_features)
        :param F: array of shape=(n_facilities, n_features)
        :param opened_facilities_idxs: list or set of int, containing the indices of already opened facilities in F
        :param next_client: int, containing the index of the first not-yet-arrived client
        :param dist_mat: a DistanceMatrix object.
        """
        self._C = C
        self._F = F
        self._n_clients, self._n_facilities = len(C), len(F)
        self._opened_facilities_idxs = set(opened_facilities_idxs)
        self._closed_facilities_idxs = set(range(self._n_facilities)).difference(self._opened_facilities_idxs)
        #TODO: change active clients to a number instead of a list
        self._next_client = len(self._C) if next_client is None else next_client
        self._dist_mat = DistanceMatrix(C, F) if dist_mat is None else dist_mat

        # facility information maintained to support fast updates
        # cache for neighborhood info for each client
        # self._nearest_facility_vec[j] is the index of opened facility nearest to j
        self._2nd_nearest_facility_vec = np.full(len(self._C), -1, dtype=np.int)
        self._dist_to_2nd_nearest_facility_vec = np.full(len(self._C), 0, dtype=np.float)

        # TODO: do we really need heaps?
        # Every client by default is connected to the nearest facility
        # The heap contains the info of the 2nd-nearest facicility for every client
        self._open_facility_heaps = []
        self._close_facility_heaps = []
        self._update_2nd_nearest_facilities_info()

        # initialize cost
        self._conn_cost_vec = np.zeros(len(self._C))
        self._assignment = np.full(len(self._C), -1, dtype=np.int)
        self._cost = None
        self._init_assignment()

    @property
    def clients(self):
        """return the array containing all possible (arrived or not-yet arrived) clients"""
        return self._C

    @property
    def facilities(self):
        """return the array containing all possible facilities"""
        return self._F

    @property
    def opened_idxs(self):
        """return a set containing the indices of all opened facilities"""
        return self._opened_facilities_idxs

    @property
    def closed_idxs(self):
        """return a set containing the indices of all closed facilities"""
        return self._closed_facilities_idxs

    @property
    def active_client_idxs(self):
        """return a list containing the indices of all arrived clients"""
        return np.arange(self._next_client)

    @property
    def assignment(self):
        """return the array contain the assignment"""
        return self._assignment

    @property
    def cost(self):
        return self._cost

    @property
    def cost_vec(self):
        return self._conn_cost_vec

    def is_open(self, i):
        return i in self._opened_facilities_idxs

    def arrive(self, j):
        """assign new client j to nearest open facility"""
        if j < self._next_client:
            return self
        if j != self._next_client:
            raise ValueError
        self._next_client = j + 1

        indices = list(self._opened_facilities_idxs)
        dists = self._dist_mat.distances([j], indices, pairwise=True).ravel()
        if len(indices) >= 2:
            idxs = np.argpartition(dists, kth=1)
            self._2nd_nearest_facility_vec[j] = indices[idxs[1]]
            self._dist_to_2nd_nearest_facility_vec[j] = dists[idxs[1]]
        else:
            idxs = [np.argmin(dists)]

        self._assignment[j] = indices[idxs[0]]
        self._conn_cost_vec[j] = dists[idxs[0]]
        self._cost += dists[idxs[0]]

        return self

    def refresh_cache(self):
        self._init_assignment()
        self._update_2nd_nearest_facilities_info()

    def _update_2nd_nearest_facilities_info(self):
        if len(self._opened_facilities_idxs) >= 2:
            f_idxs = np.array(list(self._opened_facilities_idxs))
            idxs, dists = self._dist_mat.pairwise_dist_kth_nearest(self.active_client_idxs, f_idxs, k=1)
            self._2nd_nearest_facility_vec[:self._next_client] = idxs
            self._dist_to_2nd_nearest_facility_vec[:self._next_client] = dists
            # TODO: initialize heap: do we really nead that?
        return None

    def _init_assignment(self):
        """Assign every clients to its nearest facility"""
        indices = np.array(list(self._opened_facilities_idxs))

        closest, min_dist = self._dist_mat.pairwise_dist_argmin_min(self.active_client_idxs, indices)
        self._assignment[:self._next_client] = closest
        self._conn_cost_vec[:self._next_client] = min_dist

        self._cost = self._conn_cost_vec.sum()

        return None

    def nearest_facility(self, client_idxs):
        """
        Return the nearest open facilities and the corresponding distance for clients.

        NOTE: client_idxs must be a subset of active_client_idxs, otherwise the returned value is undefined.
        :param client_idxs: list of int, queried client indices
        :return (idxs, dist): idxs[j]=i means the nearest facility to client j is i.
            dist[j] is the dist from j to its nearest facility.
        """
        return self._assignment[client_idxs], self._conn_cost_vec[client_idxs]

    def swap(self, swap_out, swap_in, reassigned, new_assignment, new_cost_vec):
        """
        :param swap_out: int, index of facilities to be closed
        :param swap_in: int, index of facility to be open
        :param reassigned: list of int, indices of clients to be reassigned
        :param new_assignment: array of shape=(len(reassigned),), new_assignment[k] = i means
            client reassigned[k] should be assigned to facility i
        :param new_cost_vec: array of shape=(len(reassigned),), new_dist[j] = d means
            client reassigned[k] is of dist d to the new facility it's assigned to
        :return:
        """
        #TODO: test if using heaps improves efficiency
        self._opened_facilities_idxs.remove(swap_out)
        self._opened_facilities_idxs.add(swap_in)
        self._closed_facilities_idxs.remove(swap_in)
        self._closed_facilities_idxs.add(swap_out)
        self._assignment[reassigned] = new_assignment
        self._conn_cost_vec[reassigned] = new_cost_vec
        self._cost = self._conn_cost_vec[:self._next_client].sum()
        self._update_2nd_nearest_facilities_info()

        return self

    def can_swap(self, swap_in, swap_out=None, cost_thresh=0, avg_cost_thresh=0):
        """
        swap some opened facilities with new facilities
        :param swap_in: int, facility index to be open
        :param swap_out: None or list of int, facility indices to look at,
            if None then look at all current open facilities
        :param cost_thresh: float, threshold for total cost reduced
        :param avg_cost_thresh: float, threshold for cost-per-client-recourse reduced
        :return (i, reassigned, new_assignment, new_cost):
            i: index of the facility to be swapped out
            reassigned: list of indices of clients to be reassigned.
            new_assignment: array of shape=(n_reassigned,), indices of facilities to be assigned to
            new_cost_vec: array of shape=(n_reassigned,), new conn cost for the reassigned clients
        """
        if swap_out is None:
            swap_out = self._opened_facilities_idxs
        else:
            swap_out = [i for i in swap_out if self.is_open(i)]
        if self.is_open(swap_in) or len(swap_out) == 0:
            return None, None, None, None

        # calculate the cost saved via this swap operation, "_a_s" means "after swap"
        curr_conn_cost = self._conn_cost_vec[:self._next_client]
        dist_to_swap_in = self._dist_mat.distances(self.active_client_idxs,
                                                   [swap_in]).ravel()
        closer_to_swap_in = (curr_conn_cost > dist_to_swap_in)

        # check if the swapped-in facility can reduce cost for any clients
        if not np.any(closer_to_swap_in):
            return None, None, None, None

        curr_assignment = self._assignment[:self._next_client]
        dist_to_2nd_nearest = self._dist_to_2nd_nearest_facility_vec[:self._next_client]
        idx_of_2nd_nearest = self._2nd_nearest_facility_vec[:self._next_client]
        for i in swap_out:
            cluster_i = (curr_assignment == i)
            reassigned = np.where(np.logical_or(closer_to_swap_in,
                                                cluster_i))[0]

            # check if enough cost is reduced
            cost_a_s = np.minimum(dist_to_2nd_nearest[reassigned],
                                  dist_to_swap_in[reassigned])
            saved_cost = curr_conn_cost[reassigned].sum() - cost_a_s.sum()
            if saved_cost < cost_thresh or saved_cost / (len(reassigned) + len(swap_out)) < avg_cost_thresh:
                continue

            # Able to swap; Calculate the information needed for conduct swap operation
            new_assignment = np.ones(len(reassigned), dtype=np.int) * -1
            new_assignment[cost_a_s < dist_to_2nd_nearest[reassigned]] = swap_in
            new_assignment[new_assignment < 0] = idx_of_2nd_nearest[reassigned][new_assignment < 0]

            return i, reassigned, new_assignment, cost_a_s

        return None, None, None, None,
