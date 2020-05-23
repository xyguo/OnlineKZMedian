# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
import warnings
import heapq


def _best_client_set(cost_reduction, cost_thresh=0, avg_cost_thresh=0, offset=0):
    """
    Given a vector that corresponds to the (connection) cost reduction per client of some operation,
    decide the best subset of clients.
    :param cost_reduction: array of shape=(n_clients,), cost_reduction[i] is the amount of cost reduced on i.
        (If cost_reduction[i]<0 that means client i's conn cost increases)
    :param cost_thresh: float, the threshold for total cost reduced
    :param avg_cost_thresh: float, threshold for cost-per-recourse reduced
    :param offset: float, an additional shift that can represent a "virtual facility cost" when then operation involves opening/closing a facility. offset>0 means a
        facility with opening cost offset is opened, otherwise it's closed
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
    """
    Object that store a facility solution and support efficient local search operations
    :param C: array of shape=(n_clients, n_features)
    :param F: array of shape=(n_facilities, n_features)
    :param opened_facilities_idxs: list or set containing the indices of already opened facilities in F
    :param assignment: array of shape=(n_clients,), assignment[j]=i means C[j] is assigned to F[i].
        Note if assignment[j]=i for nonnegative i, then i must be in opened_facilities_idxs.
    :param arrived_clients_idxs:
    :param distances: a vectorized callable with two params, distances(j, i) returns the distance
        from client j to facility i.
    """
    def __init__(self, C, F, opened_facilities_idxs,
                 assignment=None,
                 arrived_clients_idxs=None,
                 distances=None):
        self._C = C
        self._F = F
        self._n_clients, self._n_facilities = len(C), len(F)
        self._opened_facilities_idxs = set(opened_facilities_idxs)
        self._closed_facilities_idxs = set(range(self._n_facilities)).difference(self._opened_facilities_idxs)
        #TODO: change active clients to a number instead of a list
        self._active_clients = list(range(len(self._C))) if arrived_clients_idxs is None else arrived_clients_idxs
        self._distances = distances
        self._clusters = None
        self._assignment = assignment

        # check if assignment is consistent with opened_facilities_idx
        if assignment is not None:
            assigned_to = set(np.unique(assignment[self._active_clients]))
            if assigned_to != opened_facilities_idxs or len(assignment) != len(C):
                raise ValueError("Invalid assignment: some client is assigned to closed facilities")
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

        #calculate cost
        self._conn_cost_vec = None
        self._cost = None
        self._calc_cost()

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
        return self._active_clients

    @property
    def assignment(self):
        """return the array contain the assignment"""
        return self._assignment

    @property
    def cost(self):
        return self._cost

    def is_open(self, i):
        return i in self._opened_facilities_idxs

    def arrive(self, j):
        """assign new client j to nearest open facility"""
        self._active_clients.append(j)
        i, d = self.nearest_facility([j])
        self._assignment[j] = i[0]
        self._conn_cost_vec[j] = d[0]
        self._cost += d[0]
        self._nearest_facility_vec[j] = i[0]
        self._dist_to_nearest_facility_vec[j] = d[0]
        return self

    def _get_dist(self, clients, facilities):
        """

        :param clients: array of queried client idxs
        :param facilities: array of queried client idxs
        :return dist: array of shape=(n_clients, n_facilities), pairwise dist between queried clients and facilities
        """
        if self._distances is not None:
            dist = self._distances(clients, facilities)
        else:
            dist = pairwise_distances(self._C[clients], self._F[facilities])
        return dist

    def _update_nearest_facilities_info(self):
        f_idxs = np.array(list(self._opened_facilities_idxs))
        dist = self._get_dist(np.arange(len(self._C)), f_idxs)

        nearest = np.argmin(dist, axis=1)
        self._nearest_facility_vec[:] = f_idxs[nearest]
        self._dist_to_nearest_facility_vec[:] = dist[(np.arange(len(self._C)), nearest)]
        # TODO: initialize heap: do we really nead that?
        return None

    def _calc_cost(self):
        if self._opened_facilities_idxs is None:
            # only happens during initialization
            self._conn_cost_vec = np.full(len(self._C), np.inf)
        elif self._assignment is None:
            indices = np.array(list(self._opened_facilities_idxs))
            F = self._F[indices]
            C = self._C[self._active_clients]
            self._conn_cost_vec = np.zeros(len(self._C))

            # This indicates every client is assigned to its nearest facility
            closest, dist = pairwise_distances_argmin_min(C, F)
            self._assignment = np.full(len(self._C), -1, dtype=np.int)
            self._assignment[self._active_clients] = indices[closest]
            self._conn_cost_vec[self._active_clients] = dist
        else:
            self._conn_cost_vec = np.zeros(len(self._C))
            C = self._C[self._active_clients]
            F = self._F[self._assignment[self._active_clients]]
            self._conn_cost_vec[self._active_clients] = np.linalg.norm(C - F, axis=1)
        self._cost = self._conn_cost_vec.sum()
        return None

    def nearest_facility(self, cluster):
        """
        :param cluster: list of client indices
        :return (idxs, dist): idxs[j]=i means the nearest facility to client j is i.
            dist[j] is the dist from j to its nearest facility.
        """
        return self._nearest_facility_vec[cluster], self._dist_to_nearest_facility_vec[cluster]

    def swap(self, swap_out, swap_in, reassigned, new_assignment, new_cost_vec):
        """
        :param swap_out: list of indices of facilities to be closed
        :param swap_in: list of facility indices to be open
        :param reassigned:
        :param new_assignment:
        :param new_cost_vec:
        :return:
        """
        #TODO: test if using heaps improves efficiency
        self._opened_facilities_idxs.difference_update(swap_out)
        self._opened_facilities_idxs.update(swap_in)
        self._closed_facilities_idxs.difference_update(swap_in)
        self._closed_facilities_idxs.update(swap_out)
        self._assignment[reassigned] = new_assignment
        self._conn_cost_vec[reassigned] = new_cost_vec
        self._cost = self._conn_cost_vec[self._active_clients].sum()
        self._update_nearest_facilities_info()

        return self

    def can_swap(self, swap_out, swap_in, cost_thresh=0, avg_cost_thresh=0):
        """
        swap some opened facilities with new
        :param swap_out: list of facility indices to be closed
        :param swap_in: list of facility indices to be open
        :param cost_thresh: float, threshold for total cost reduced
        :param avg_cost_thresh: float, threshold for cost-per-recourse reduced
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
        dist_a_s = self._get_dist(reassigned, facilities_a_s)
        nearest_f_a_s = np.argmin(dist_a_s, axis=1)
        cost_a_s = dist_a_s[(np.arange(dist_a_s.shape[0]), nearest_f_a_s)]
        nearest_f_a_s = facilities_a_s[nearest_f_a_s]

        saved_cost = self._conn_cost_vec[reassigned].sum() - cost_a_s.sum()
        if saved_cost > cost_thresh and saved_cost / (len(reassigned) + len(swap_out)) > avg_cost_thresh:
            return reassigned, nearest_f_a_s, cost_a_s
        else:
            return None, None, None
