# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from sklearn.neighbors import BallTree, KDTree
from sklearn.exceptions import NotFittedError
import warnings
import heapq


class _FacilityInfo(object):

    def __init__(self, C, F, opened_facilities,
                 active_clients,
                 distances):
        """

        :param C:
        :param F:
        :param opened_facilities:
        :param active_clients:
        :param distances:
        """
        self._C = C
        self._F = F
        self._active_clients = active_clients
        self._distances = distances
        self._opened_facilities = opened_facilities

        # cache for neighborhood info for each client
        # self._nearest_facility_vec[j] is the index of opened facility nearest to j
        self._nearest_facility_vec = np.full(len(self._C), -1, dtype=np.int)
        self._dist_to_nearest_facility_vec = np.full(len(self._C), 0, dtype=np.float)
        self._2nd_nearest_facility_vec = np.full(len(self._C), -1, dtype=np.int)
        self._dist_to_2nd_nearest_facility_vec = np.full(len(self._C), -1, dtype=np.float)

        # TODO: do we really need heaps?
        self._open_facility_heaps = []
        self._close_facility_heaps = []
        self._init_heap()

    def _init_heap(self):

        #TODO
        if self._distances is None:

            dist = pairwise_distances(self._C, self._F)
        pass

    def nearest_facility(self, cluster):
        """
        :param cluster: list of client indices
        :return (idxs, dist): idxs[j]=i means the nearest facility to client j is i.
            dist[j] is the dist from j to its nearest facility.
        """
        return self._nearest_facility_vec[cluster], self._dist_to_nearest_facility_vec[cluster]

    def second_nearest_facility(self, cluster):
        """
        :param cluster: list of client indices
        :return (idxs, dist): idxs[j]=i means the 2nd nearest facility to client j is i.
            dist[j] is the dist from j to its 2nd nearest facility.
        """
        return self._2nd_nearest_facility_vec[cluster], self._dist_to_2nd_nearest_facility_vec[cluster]

    def add(self, i, dist_to_i):
        """
        Add new opened facility.
        :param i: index of opened facility in self._F
        :param dist_to_i: array of shape=(n_clients,), contain the distance from each client to facility i.
        :return self:
        """
        change_nearest = self._dist_to_nearest_facility_vec > dist_to_i
        self._dist_to_nearest_facility_vec[change_nearest] = dist_to_i[change_nearest]
        self._nearest_facility_vec[change_nearest] = i

        change_2nd_nearest = np.logical_and(self._dist_to_2nd_nearest_facility_vec > dist_to_i,
                                            self._dist_to_2nd_nearest_facility_vec < dist_to_i)
        self._dist_to_2nd_nearest_facility_vec[change_2nd_nearest] = dist_to_i[change_2nd_nearest]
        self._2nd_nearest_facility_vec[change_2nd_nearest] = i

        #TODO: update heap
        return self

    def remove(self, i):
        """
        Remove a closed facility.
        :param i: index of the facility to be close in self._F
        :return self:
        """
        # TODO: add a heap implementation?
        # Right now the implementation is by bruteforce
        affected_clients, _ = np.where(self._nearest_facility_vec == i)

        return self


class Assignment(object):
    """
    Object that store a facility solution and support efficient local search operations
    :param C: array of shape=(n_clients, n_features)
    :param F: array of shape=(n_facilities, n_features)
    :param opened_facilities_idxs: list or set containing the indices of already opened facilities in F
    :param facility_cost_vec: array of shape=(n_facilities,),
        facility_cost_vec[i] is the opening cost of facility F[i]
    :param assignment: array of shape=(n_clients,), assignment[j]=i means C[j] is assigned to F[i].
        Note if assignment[j]=i for nonnegative i, then i must be in opened_facilities_idxs.
    :param arrived_clients_idxs:
    :param distances: a vectorized callable with two params, distances(j, i) returns the distance
        from client j to facility i.
    """
    def __init__(self, C, F, opened_facilities_idxs,
                 facility_cost_vec=None, assignment=None,
                 arrived_clients_idxs=None,
                 distances=None):
        self._C = C
        self._F = F
        self._opened_facilities_idxs = set(opened_facilities_idxs)
        self._facility_cost_vec = np.zeros(len(F)) if not facility_cost_vec else facility_cost_vec
        self._assignment = assignment
        self._active_clients = list(range(len(self._C))) if arrived_clients_idxs is None else arrived_clients_idxs
        self._distances = distances
        self._clusters = None

        # check if assignment is consistent with opened_facilities_idx
        if assignment is not None:
            assigned_to = np.unique(assignment)
            if not np.all(assigned_to == np.sort(opened_facilities_idxs)):
                raise ValueError("Invalid assignment: some client is assigned to closed facilities")

        # facility information maintained to support fast updates
        # cache for neighborhood info for each client
        # self._nearest_facility_vec[j] is the index of opened facility nearest to j
        self._nearest_facility_vec = np.full(len(self._C), -1, dtype=np.int)
        self._dist_to_nearest_facility_vec = np.full(len(self._C), 0, dtype=np.float)
        self._2nd_nearest_facility_vec = np.full(len(self._C), -1, dtype=np.int)
        self._dist_to_2nd_nearest_facility_vec = np.full(len(self._C), -1, dtype=np.float)
        # TODO: do we really need heaps?
        self._open_facility_heaps = []
        self._close_facility_heaps = []
        self._init_cache()

        #calculate cost
        self._conn_cost_vec = None
        self._fac_cost = None
        self._calc_cost()

    def is_open(self, i):
        return i in self._opened_facilities_idxs

    def arrive(self, j):
        self._active_clients.append(j)
        return self

    def assign(self, j, i):
        self._assignment[j] = i
        return self

    @property
    def clients(self):
        return self._C

    @property
    def all_facilities(self):
        return self._F

    @property
    def opened(self):
        return self._opened_facilities_idxs

    @property
    def arrived(self):
        return self._active_clients

    @property
    def connection(self):
        return self._assignment

    @property
    def cost(self):
        return self._conn_cost_vec.sum() + self._fac_cost

    @property
    def connection_cost(self):
        return self._conn_cost_vec.sum()

    @property
    def facility_cost(self):
        return self._fac_cost

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

    def _init_cache(self):
        f_idxs = list(self._opened_facilities_idxs)
        dist = self._get_dist(np.arange(len(self._C)), f_idxs)

        idx = np.argpartition(dist, 2, axis=1)
        sorted_dist = np.partition(dist, 2, axis=1)

        self._nearest_facility_vec[:] = f_idxs[idx[:, 0]]
        self._2nd_nearest_facility_vec[:] = f_idxs[idx[:, 1]]

        self._dist_to_nearest_facility_vec[:] = sorted_dist[:, 0]
        self._dist_to_2nd_nearest_facility_vec[:] = sorted_dist[:, 1]
        # TODO: initialize heap: do we really nead that?

    def _calc_cost(self):
        if self._opened_facilities_idxs is None:
            # only happens during initialization
            self._conn_cost_vec = np.full(len(self._C), np.inf)
            self._fac_cost = 0
        elif self._assignment is None:
            # This indicates every client is assigned to its nearest facility
            indices = list(self._opened_facilities_idxs)
            self._fac_cost = self._facility_cost_vec[indices].sum()
            F = self._F[indices]
            closest, dist = pairwise_distances_argmin_min(self._C, F)
            self._assignment = indices[closest]
            self._conn_cost_vec = dist
        else:
            indices = list(self._opened_facilities_idxs)
            self._fac_cost = self._facility_cost_vec[indices].sum()
            F = self._F[self._assignment]
            self._conn_cost_vec = np.linalg.norm(self._C - F, axis=1)
            self._cost = self._fac_cost

    def _update_cache_open(self, i, dist_to_i):
        change_nearest = self._dist_to_nearest_facility_vec > dist_to_i
        self._dist_to_nearest_facility_vec[change_nearest] = dist_to_i[change_nearest]
        self._nearest_facility_vec[change_nearest] = i

        change_2nd_nearest = np.logical_and(self._dist_to_2nd_nearest_facility_vec > dist_to_i,
                                            self._dist_to_nearest_facility_vec < dist_to_i)
        self._dist_to_2nd_nearest_facility_vec[change_2nd_nearest] = dist_to_i[change_2nd_nearest]
        self._2nd_nearest_facility_vec[change_2nd_nearest] = i
        # TODO: add a heap implementation?

    def _update_cache_close(self, i):
        # Right now the implementation is by bruteforce
        f_idxs = list(self._opened_facilities_idxs)
        affected_clients, _ = np.where(np.logical_or(self._nearest_facility_vec == i,
                                                     self._2nd_nearest_facility_vec == i))
        dist = self._get_dist(affected_clients, f_idxs)
        idx = np.argpartition(dist, 2, axis=1)
        sorted_dist = np.partition(dist, 2, axis=1)

        self._nearest_facility_vec[affected_clients] = f_idxs[idx[:, 0]]
        self._2nd_nearest_facility_vec[affected_clients] = f_idxs[idx[:, 1]]

        self._dist_to_nearest_facility_vec[affected_clients] = sorted_dist[:, 0]
        self._dist_to_2nd_nearest_facility_vec[affected_clients] = sorted_dist[:, 1]
        # TODO: add a heap implementation?

    def nearest_facility(self, cluster):
        """
        :param cluster: list of client indices
        :return (idxs, dist): idxs[j]=i means the nearest facility to client j is i.
            dist[j] is the dist from j to its nearest facility.
        """
        return self._nearest_facility_vec[cluster], self._dist_to_nearest_facility_vec[cluster]

    def second_nearest_facility(self, cluster):
        """
        :param cluster: list of client indices
        :return (idxs, dist): idxs[j]=i means the 2nd nearest facility to client j is i.
            dist[j] is the dist from j to its 2nd nearest facility.
        """
        return self._2nd_nearest_facility_vec[cluster], self._dist_to_2nd_nearest_facility_vec[cluster]

    def _best_client_set(self, cost_reduction, cost_thresh=0, avg_cost_thresh=0, f_cost=0):
        """
        Given a vector that corresponds to the (connection) cost reduction per client of some operation,
        decide the best subset of clients.
        :param cost_reduction: array of shape=(n_clients,), cost_reduction[i] is the amount of cost reduced on i.
            (If cost_reduction[i]<0 that means client i's conn cost increases)
        :param cost_thresh: float, the threshold for total cost reduced
        :param avg_cost_thresh: float, threshold for cost-per-recourse reduced
        :param f_cost: float, a shift when then operation involves opening/closing a facility. f_cost>0 means a
            facility with opening cost f_cost is opened, otherwise it's closed
        :return: list of the indices of best clients subset that satisfies avg_cost_thresh while having maximum
            total reduced cost. Return None if no such subset exists.
        """
        n_clients = len(cost_reduction)
        sorted_idx = np.argsort(cost_reduction)[::-1]
        amortized_cost_reduction = (cost_reduction[sorted_idx].cumsum() - f_cost) / np.arange(1, n_clients+1)

        if amortized_cost_reduction[0] > avg_cost_thresh:
            good_idxs, _ = np.where(amortized_cost_reduction > cost_thresh)
            good_idxs = good_idxs.max()
            total_reduced_cost = cost_reduction[sorted_idx][:good_idxs+1].sum() - f_cost
            if total_reduced_cost > cost_thresh:
                return sorted_idx[:good_idxs+1]

        return None

    def can_open(self, i, cost_thresh=0, avg_cost_thresh=0):
        """
        Return true if opening i reduces enough cost or cost-per-recourse
        :param i: index of the queried facility in F
        :param cost_thresh: float, threshold for total cost reduced
        :param avg_cost_thresh: float, threshold for cost-per-recourse reduced
        :return: True if if it's possible to open i respecting the two thresholds
        """
        if i in self._opened_facilities_idxs:
            return False
        _, dist_to_i = pairwise_distances_argmin_min(self._F[i][:, np.newaxis], self._C)
        closer_to_i = np.where(self._conn_cost_vec > dist_to_i)[0]
        saved_conn_cost = self._conn_cost_vec[closer_to_i] - dist_to_i[closer_to_i]
        f_cost = self._facility_cost_vec[i]

        clients_set = self._best_client_set(saved_conn_cost, cost_thresh=cost_thresh,
                                            avg_cost_thresh=avg_cost_thresh, f_cost=f_cost)
        return clients_set is not None

    def can_close(self, i, cost_thresh=0, avg_cost_thresh=0):
        """
        Return true if closing i reduces enough cost or cost-per-recourse
        :param i: index of the queried facility in F
        :param cost_thresh: float, threshold for total cost reduced
        :param avg_cost_thresh: float, threshold for cost-per-recourse reduced
        :return: True if if it's possible to close i respecting the two thresholds
        """
        if i not in self._opened_facilities_idxs:
            return False
        cluster_i, _ = np.where(self._assignment == i)
        nearest_f_idx, dist = self._nearest_facility_vec[cluster_i]
        sec_nearest_f_idx, dist_sec = self._2nd_nearest_facility_vec[cluster_i]

        # for clients previously assigned to i, re-assign it to its nearest facility or 2nd nearest one
        new_dist = dist.copy()
        new_dist[nearest_f_idx == i] = dist_sec[nearest_f_idx == i]
        reduced_cost = (self._conn_cost_vec[cluster_i] - new_dist).sum() + self._facility_cost_vec[i]

        return reduced_cost > cost_thresh and reduced_cost / len(cluster_i) > avg_cost_thresh

    def can_swap(self, i1, i2, cost_thresh=0, avg_recourse_thresh=0):
        #TODO
        return True

    def _open(self, i, reassigned, dist_to_i):
        """
        open facility i, and assign specified clients to it
        :param i: index of the facility to be open
        :param reassigned: index of clients to be reassigned to i
        :param dist_to_i: array of shape=(n_clients,), dist from each client to i
        :return self:
        """
        self._opened_facilities_idxs.add(i)
        self._assignment[reassigned] = i
        self._conn_cost_vec[reassigned] = dist_to_i[reassigned]
        self._fac_cost += self._facility_cost_vec[i]

        # update facility info cache
        self._update_cache_open(i, dist_to_i)

    def open(self, i, cost_thresh=0, avg_cost_thresh=0, reassigned=None):
        """
        Try to open i if it reduces enough cost and cost-per-recourse
        :param i: index of the queried facility in F
        :param cost_thresh: float, threshold for total cost reduced
        :param avg_cost_thresh: float, threshold for cost-per-recourse reduced
        :param reassigned: list of int, indices of clients to be reassigned to i
        :return self:
        """
        warnings.warn(DeprecationWarning)
        # _, dist_to_i = pairwise_distances_argmin_min(self._F[i][:, np.newaxis], self._C)
        if self.is_open(i):
            return self
        dist_to_i = self._get_dist(np.arange(len(self._C)), [i]).ravel()
        if not reassigned:
            closer_to_i = np.where(self._conn_cost_vec > dist_to_i)[0]
            saved_conn_cost = self._conn_cost_vec[closer_to_i] - dist_to_i[closer_to_i]
            f_cost = self._facility_cost_vec[i]

            reassigned = self._best_client_set(saved_conn_cost, cost_thresh=cost_thresh,
                                               avg_cost_thresh=avg_cost_thresh, f_cost=f_cost)

        # it's still possible that self._best_client_set returned None
        if reassigned is not None:
            self._open(i, reassigned, dist_to_i)
        return self

    def _close(self, i, cluster_i, new_assignment, new_dist):
        """
        close facility i and reassign all affected clients to nearest facilities
        :param i:
        :param cluster_i:
        :param new_assignment:
        :param new_dist:
        :return:
        """
        self._opened_facilities_idxs.remove(i)
        self._assignment[cluster_i] = new_assignment
        self._conn_cost_vec[cluster_i] = new_dist
        self._fac_cost -= self._facility_cost_vec[i]
        # update facility info cache
        self._update_cache_close(i)

    def close(self, i, cost_thresh=0, avg_cost_thresh=0):
        """
        Try to close i if it reduces enough total cost and cost-per-recourse
        :param i: index of the queried facility in F
        :param cost_thresh: float, threshold for total cost reduced
        :param avg_cost_thresh: float, threshold for cost-per-recourse reduced
        :return: True if it's possible to close i respecting the two thresholds
        """
        warnings.warn(DeprecationWarning)
        if not self.is_open(i):
            return self
        cluster_i, _ = np.where(self._assignment == i)
        nearest_f_idx, dist = self.nearest_facility(cluster_i)
        sec_nearest_f_idx, dist_sec = self.second_nearest_facility(cluster_i)

        # for clients previously assigned to i, re-assign it to its nearest facility or 2nd nearest one
        new_dist = dist.copy()
        new_dist[nearest_f_idx == i] = dist_sec[nearest_f_idx == i]
        new_assignment = nearest_f_idx.copy()
        new_assignment[nearest_f_idx == i] = sec_nearest_f_idx[nearest_f_idx == i]

        reduced_cost = (self._conn_cost_vec[cluster_i] - new_dist).sum() + self._facility_cost_vec[i]
        if reduced_cost > cost_thresh and reduced_cost / len(cluster_i) > avg_cost_thresh:
            self._close(i, cluster_i, new_assignment, new_dist)
        return self

    def _swap(self, i1, i2, new_assignment, new_dist):
        #TODO
        return None

    def swap(self, i1, i2, cost_thresh=0, avg_cost_thresh=0, reassigned_to_i2=None):
        """
        swap facility i1 and i2: open i2 and close i1
        :param i1: facility index to be closed
        :param i2: facility index to be open
        :param cost_thresh: float, threshold for total cost reduced
        :param avg_cost_thresh: float, threshold for cost-per-recourse reduced
        :param reassigned_to_i2: list of int, one can specify a set of clients to be assigned to i2
        :return self:
        """
        if self.is_open(i1) and self.is_open(i2):
            return self.close(i1, cost_thresh=cost_thresh, avg_cost_thresh=avg_cost_thresh)
        elif not self.is_open(i1):
            return self.open(i2, cost_thresh=cost_thresh, avg_cost_thresh=avg_cost_thresh)

        # These are clients affected by closing i1. All of them will be
        # assigned to nearest facility (if not specified in `reassigned_to_i2`)
        cluster_i1, _ = np.where(self._assignment == i1)
        nearest_f_idx, dist_to_nearest = self.nearest_facility(cluster_i1)
        sec_nearest_f_idx, dist_to_2nd_nearest = self.second_nearest_facility(cluster_i1)

        # compute the nearest (and 2nd nearest) facility after i2 is open
        dist_to_i2 = self._get_dist(np.arange(len(self._C)), [i2]).ravel()
        cluster_i1_dist_to_i2 = dist_to_i2[cluster_i1]
        i2_as_nearest, _ = np.where(dist_to_nearest > cluster_i1_dist_to_i2)
        nearest_f_idx[i2_as_nearest] = i2
        dist_to_nearest = np.minimum(cluster_i1_dist_to_i2, dist_to_nearest)
        i2_as_2nd_nearest, _ = np.where(np.logical_and(
            dist_to_nearest <= cluster_i1_dist_to_i2,
            dist_to_2nd_nearest > cluster_i1_dist_to_i2))
        sec_nearest_f_idx[i2_as_2nd_nearest] = i2
        dist_to_2nd_nearest[i2_as_2nd_nearest] = cluster_i1_dist_to_i2[i2_as_2nd_nearest]

        # for clients previously assigned to i, re-assign it to its nearest facility or 2nd nearest one
        new_dist = dist_to_nearest.copy()
        new_dist[nearest_f_idx == i1] = dist_to_2nd_nearest[nearest_f_idx == i1]
        new_assignment = nearest_f_idx.copy()
        new_assignment[nearest_f_idx == i1] = sec_nearest_f_idx[nearest_f_idx == i1]

        if not reassigned_to_i2:
            closer_to_i2, _ = np.where(self._conn_cost_vec > dist_to_i2)
            saved_conn_cost = self._conn_cost_vec[closer_to_i2] - dist_to_i2[closer_to_i2]
            f_cost = self._facility_cost_vec[i2]

            reassigned_to_i2 = self._best_client_set(saved_conn_cost, cost_thresh=cost_thresh,
                                                     avg_cost_thresh=avg_cost_thresh, f_cost=f_cost)

        reduced_cost = (self._conn_cost_vec[cluster_i1] - new_dist).sum() + self._facility_cost_vec[i2]
        if reduced_cost > cost_thresh and reduced_cost / len(cluster_i1) > avg_cost_thresh:
            self._swap(i1, i2, new_assignment, new_dist)
        return self
