# -*- coding: utf-8 -*-
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
from itertools import combinations
import warnings
from .assignmenter import Assignmenter


def _trivial_k_median(clients, facilities):
    """
    k-median when k = number of clients
    :param clients: array of shape=(n_clients, n_features)
    :param facilities: array of shape=(n_facilities, n_features)
    :return opened_facility_idxs: set of indices of facilities opened
    """
    n_clients, n_facilities = len(clients), len(facilities)
    if n_clients >= n_facilities:
        return set(range(n_facilities))
    idxs = pairwise_distances_argmin(clients, facilities)
    return set(np.unique(idxs))


class OnlineKZMed(object):
    def __init__(self, n_clusters, n_outliers=0, epsilon=0.1, gamma=1, ell=1):
        """

        :param n_clusters:
        :param n_outliers:
        :param epsilon:
        :param gamma:
        :param ell:
        """
        self._n_clusters = n_clusters
        self._n_outliers = n_outliers
        self._epsilon = epsilon
        self._ell = ell
        self._gamma = gamma

        self._assignmenter = None
        self._cluster_centers = None
        self._clusters = None
        self._facility_recourse = 0
        self._client_recourse = 0

    @property
    def cluster_centers(self):
        return self._cluster_centers

    @property
    def clusters(self):
        return self._clusters

    @property
    def assignment(self):
        if not self._assignmenter:
            raise NotFittedError
        return self._assignmenter.assignment

    def facility_recourse(self):
        return self._facility_recourse

    def client_recourse(self):
        return self._client_recourse

    def cost(self):
        if not self._assignmenter:
            raise NotFittedError
        return self._assignmenter.cost

    def fit(self, C, F=None, distances=None):
        """
        This method *simulates* the online (k,z)-median algorithm by first read in the whole
        data set C and iterate through it.
        :param C: array of shape=(n_clients, n_features)
        :param F: array of shape=(n_facilities, n_features)
        :param distances: array of shape=(n_clients, n_facilities), distance[j, i] is the distance between client j and facility i
        :return self:
        """
        # initialize parameters
        n_clients, n_features = C.shape
        if F is None:
            F = C
        n_facilities, _ = F.shape
        n_clusters = min(self._n_clusters, n_facilities, n_clients)
        n_outliers = min(self._n_outliers, n_clients)

        p = min(1 / (3 * n_outliers), 1 / (self._gamma * n_outliers))
        if distances is None:
            def dist(c_idxs, f_idxs):
                return np.clip(pairwise_distances(C[c_idxs], F[f_idxs]), a_min=0, a_max=p)
        else:
            def dist(c_idxs, f_idxs):
                return distances[np.ix_(c_idxs, f_idxs)]

        # initialization: assign the first k client to k nearest facilities
        arrived_clients = list(range(n_clusters))
        opened_facilities = _trivial_k_median(clients=C[arrived_clients],
                                              facilities=F)
        self._assignmenter = Assignmenter(C, F,
                                          opened_facilities_idxs=opened_facilities,
                                          arrived_clients_idxs=arrived_clients,
                                          distances=dist)
        prev_cost = self._assignmenter.cost

        # accepting subsequent clients
        n_outliers_thresh = (1 + 1/self._ell) * (1 + self._gamma) / (1 - self._epsilon)
        for j in range(n_clusters, n_clients):
            # Accommodate the new arrived client j by assigning it to its nearest facility.
            self._assignmenter.arrive(j)
            # TODO: should this 1-swap also be part of the accommodating step?
            # self._local_search(rho=0,
            #                    assignment=self._assignmenter,
            #                    ell=1)

            # If client j doesn't increase cost too much, then skip local search
            if self._assignmenter.cost < 2 * prev_cost:
                continue

            # update the solution via local search
            thresh = self._epsilon * self._assignmenter.cost / n_clusters
            while True:
                ok, f_rec, c_rec = self._local_search(rho=thresh,
                                                      assignmenter=self._assignmenter,
                                                      ell=self._ell)
                while ok:
                    self._facility_recourse += f_rec
                    self._client_recourse += c_rec
                    ok, f_rec, c_rec = self._local_search(rho=thresh,
                                                          assignmenter=self._assignmenter,
                                                          ell=self._ell)
                _, d = self._assignmenter.nearest_facility(list(range(j)))

                # control the number of outliers
                if np.count_nonzero(d >= p) >= n_outliers_thresh:
                    p = (1 + self._epsilon) * p
                else:
                    break

            prev_cost = self._assignmenter.cost

        opened_idxs_lst = list(self._assignmenter.opened_idxs)
        self._cluster_centers = F[opened_idxs_lst]
        self._clusters = [np.where(self._assignmenter == i)[0] for i in opened_idxs_lst]
        return self

    def _local_search(self, rho, assignmenter, ell=1):
        """conduct one step of rho-efficient ell-swap"""
        n_opened = len(assignmenter.opened_idxs)
        ell = min(ell, n_opened)

        for swap_out in np.random.permutation(list(combinations(assignmenter.opened_idxs, ell))):
            for swap_in in np.random.permutation(list(combinations(assignmenter.closed_idxs, ell))):
                reassigned, connection, cost = assignmenter.can_swap(swap_out, swap_in, cost_thresh=rho * ell,
                                                                     avg_cost_thresh=0)
                if reassigned is not None:
                    assignmenter.swap(swap_out, swap_in, reassigned, connection, cost)
                    return True, len(swap_out) + len(swap_in), len(reassigned)
        return False, None, None

    def update(self, clients):
        """ accept new clients """
        raise NotImplementedError("The `update` method has not been implemented yet.")
