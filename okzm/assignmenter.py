# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
import warnings
import heapq


# TODO: rewrite the `[]` (__getitem__) operator to replace the cumbersome distances function
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


def _best_client_set(cost_reduction, cost_thresh=0, avg_cost_thresh=0, offset=0):
    """
    Given a vector that corresponds to the (connection) cost reduction per client of some operation,
    decide the best subset of clients.
    :param cost_reduction: array of shape=(n_clients,), cost_reduction[i] is the amount of cost reduced on i.
        (If cost_reduction[i]<0 that means client i's conn cost increases)
    :param cost_thresh: float, the threshold for total cost reduced
    :param avg_cost_thresh: float, threshold for cost-per-recourse reduced
    :param offset: float, an additional shift that can represent a "virtual facility cost" when
        then operation involves opening/closing a facility. offset>0 means a facility with opening
        cost offset is opened, otherwise it's closed.
    :return: list of the indices of best clients subset that satisfies avg_cost_thresh while having maximum
        total reduced cost. Return None if no such subset exists.
    """
    n_clients = len(cost_reduction)
    sorted_idx = np.argsort(cost_reduction)[::-1]
    amortized_cost_reduction = (cost_reduction[sorted_idx].cumsum() - offset) / np.arange(1, n_clients + 1)

    if amortized_cost_reduction[0] > avg_cost_thresh:
        good_idxs, _ = np.where(amortized_cost_reduction > cost_thresh)
        good_idxs = good_idxs.max()
        total_reduced_cost = cost_reduction[sorted_idx][:good_idxs+1].sum() - offset
        if total_reduced_cost > cost_thresh:
            return sorted_idx[:good_idxs+1]

    return None


class Assignmenter(object):

    def __init__(self, C, F, opened_facilities_idxs,
                 assignment=None,
                 arrived_clients_idxs=None,
                 dist_mat=None):
        """
        Object that store a facility solution and support efficient local search operations
        :param C: array of shape=(n_clients, n_features)
        :param F: array of shape=(n_facilities, n_features)
        :param opened_facilities_idxs: list or set of int, containing the indices of already opened facilities in F
        :param assignment: array of shape=(n_clients,), assignment[j]=i means C[j] is assigned to F[i].
            Note if assignment[j]=i for nonnegative i, then i must be in opened_facilities_idxs.
        :param arrived_clients_idxs: list of int, containing the indices of arrived clients
        :param dist_mat: a DistanceMatrix object.
        """
        self._C = C
        self._F = F
        self._n_clients, self._n_facilities = len(C), len(F)
        self._opened_facilities_idxs = set(opened_facilities_idxs)
        self._closed_facilities_idxs = set(range(self._n_facilities)).difference(self._opened_facilities_idxs)
        #TODO: change active clients to a number instead of a list
        self._active_client_idxs = list(range(len(self._C))) if arrived_clients_idxs is None else arrived_clients_idxs
        self._clusters = None
        self._dist_mat = DistanceMatrix(C, F) if dist_mat is None else dist_mat

        self._assignment = assignment
        # check if assignment is consistent with opened_facilities_idx
        if assignment is not None:
            assigned_to = set(np.unique(assignment[self._active_client_idxs]))
            if not assigned_to.issubset(opened_facilities_idxs):
                raise ValueError("Invalid assignment: some client is assigned to closed facilities.")
            if len(assignment) != len(C):
                raise ValueError("Invalid assignment: some client is unassigned.")
            self._assignment = np.array(assignment)

        # facility information maintained to support fast updates
        # cache for neighborhood info for each client
        # self._nearest_facility_vec[j] is the index of opened facility nearest to j
        self._nearest_facility_vec = np.full(len(self._C), -1, dtype=np.int)
        self._dist_to_nearest_facility_vec = np.full(len(self._C), 0, dtype=np.float)

        # TODO: do we really need heaps?
        self._open_facility_heaps = []
        self._close_facility_heaps = []
        self._update_nearest_facilities_info()

        # initialize cost
        self._conn_cost_vec = None
        self._cost = None
        self._init_cost()

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
        return self._active_client_idxs

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
        self._active_client_idxs.append(j)
        i, d = self.nearest_facility([j])
        self._assignment[j] = i[0]
        self._conn_cost_vec[j] = d[0]
        self._cost += d[0]
        self._nearest_facility_vec[j] = i[0]
        self._dist_to_nearest_facility_vec[j] = d[0]
        return self

    def refresh_cache(self):
        self._init_cost()
        self._update_nearest_facilities_info()

    def _update_nearest_facilities_info(self):
        f_idxs = np.array(list(self._opened_facilities_idxs))
        nearest, min_dist = self._dist_mat.pairwise_dist_argmin_min(np.arange(len(self._C)), f_idxs)
        self._nearest_facility_vec[:] = nearest
        self._dist_to_nearest_facility_vec[:] = min_dist
        # TODO: initialize heap: do we really nead that?
        return None

    def _init_cost(self):
        if self._opened_facilities_idxs is None:
            # only happens during initialization
            self._conn_cost_vec = np.full(len(self._C), np.inf)
        elif self._assignment is None:
            indices = np.array(list(self._opened_facilities_idxs))
            self._conn_cost_vec = np.zeros(len(self._C))

            # This indicates every client is assigned to its nearest facility
            closest, min_dist = self._dist_mat.pairwise_dist_argmin_min(self.active_client_idxs, indices)
            self._assignment = np.full(len(self._C), -1, dtype=np.int)
            self._assignment[self._active_client_idxs] = closest
            self._conn_cost_vec[self._active_client_idxs] = min_dist
        else:
            self._conn_cost_vec = np.zeros(len(self._C))
            self._conn_cost_vec[self._active_client_idxs] = self._dist_mat.distances(
                self._active_client_idxs,
                self._assignment[self._active_client_idxs],
                pairwise=False
            )

        self._cost = self._conn_cost_vec.sum()
        return None

    def nearest_facility(self, cluster):
        """
        :param cluster: list of int, queried client indices
        :return (idxs, dist): idxs[j]=i means the nearest facility to client j is i.
            dist[j] is the dist from j to its nearest facility.
        """
        return self._nearest_facility_vec[cluster], self._dist_to_nearest_facility_vec[cluster]

    def swap(self, swap_out, swap_in, reassigned, new_assignment, new_cost_vec):
        """
        :param swap_out: list of int, indices of facilities to be closed
        :param swap_in: list of int, facility indices to be open
        :param reassigned: list of int, indices of clients to be reassigned
        :param new_assignment: array of shape=(len(reassigned),), new_assignment[k] = i means
            client reassigned[k] should be assigned to facility i
        :param new_cost_vec: array of shape=(len(reassigned),), new_dist[j] = d means
            client reassigned[k] is of dist d to the new facility it's assigned to
        :return:
        """
        #TODO: test if using heaps improves efficiency
        self._opened_facilities_idxs.difference_update(swap_out)
        self._opened_facilities_idxs.update(swap_in)
        self._closed_facilities_idxs.difference_update(swap_in)
        self._closed_facilities_idxs.update(swap_out)
        self._assignment[reassigned] = new_assignment
        self._conn_cost_vec[reassigned] = new_cost_vec
        self._cost = self._conn_cost_vec[self._active_client_idxs].sum()
        self._update_nearest_facilities_info()

        return self

    def can_swap(self, swap_out, swap_in, cost_thresh=0, avg_cost_thresh=0, lazy=True):
        """
        swap some opened facilities with new facilities
        :param swap_out: list of int, facility indices to be closed
        :param swap_in: list of int, facility indices to be open
        :param cost_thresh: float, threshold for total cost reduced
        :param avg_cost_thresh: float, threshold for cost-per-client-recourse reduced
        :param lazy: If True, only reassign clients that are previously assigned to facilities in swap_out, i.e.,
            if j is not originally assigned to swap_out, then it won't be reassigned even if some i in swap_in is
            closer to j that its current hosting facility.
        :return (reassigned, new_assignment):
            reassigned: list of indices of clients to be reassigned.
            new_assignment: array of shape=(n_reassigned,), indices of facilities to be assigned to
            new_cost_vec: array of shape=(n_reassigned,), new conn cost for the reassigned clients
        """
        swap_out = [i for i in swap_out if self.is_open(i)]
        swap_in = [i for i in swap_in if not self.is_open(i)]
        if len(swap_in) == 0 or len(swap_out) == 0 or len(swap_out) != len(swap_in):
            return None, None, None

        # calculate the cost saved via this swap operation, "_a_s" means "after swap"
        reassigned = np.where(np.isin(self._assignment, swap_out))[0]
        facilities_a_s = np.array(list(self._opened_facilities_idxs.difference(swap_out).union(swap_in)))
        if not lazy:
            _, min_d_a_s = self._dist_mat.pairwise_dist_argmin_min(self._active_client_idxs, facilities_a_s)
            _, min_ds_curr = self.nearest_facility(self._active_client_idxs)
            nearest_f_changed = np.array(self._active_client_idxs)[np.where(min_d_a_s < min_ds_curr)[0]]
            reassigned = set(nearest_f_changed).union(reassigned)
            if len(reassigned) == 0:
                return None, None, None
            reassigned = np.array(list(reassigned))
            reassigned.sort()

        nearest_f_a_s, cost_a_s = self._dist_mat.pairwise_dist_argmin_min(reassigned, facilities_a_s)

        saved_cost = self._conn_cost_vec[reassigned].sum() - cost_a_s.sum()
        if saved_cost > cost_thresh and saved_cost / (len(reassigned) + len(swap_out)) > avg_cost_thresh:
            return reassigned, nearest_f_a_s, cost_a_s
        else:
            return None, None, None
