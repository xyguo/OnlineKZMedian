# -*- coding: utf-8 -*-
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import pairwise_distances_argmin
from .assignmenter import Assignmenter, DistanceMatrix
from utils import debug_print


def _trivial_k_median(clients, facilities):
    """
    k-median when k = number of clients
    :param clients: array of shape=(n_clients, n_features)
    :param facilities: array of shape=(n_facilities, n_features)
    :return (opened_facility_idxs, assignment):
        opened_facility_idxs: set of indices of facilities opened
        assignment: array of shape=(n_clients,), assignment[j] = i means client j assigned to facility i
    """
    n_clusters = min(len(clients), len(facilities))
    idxs = pairwise_distances_argmin(clients, facilities)
    idxs_set = set(np.unique(idxs))

    # make sure we open n_clusters-many facilities
    if len(idxs_set) < n_clusters:
        n_extra = n_clusters - len(idxs_set)
        unused = set(range(len(facilities))).difference(idxs_set)
        for i in range(n_extra):
            idxs_set.add(unused.pop())

    return idxs_set, idxs


class OnlineKZMed(object):
    def __init__(self, n_clusters, n_outliers=0,
                 epsilon=0.1, gamma=1,
                 random_swap_in=None, random_swap_out=None,
                 debugging=False):
        """
        Implementation of the bi-criteria online (k,z)-median algorithm based on local search
        :param n_clusters: int, number of clusters (i.e, k).
            The algorithm will open at most n_clusters many facilities.
        :param n_outliers: int, number of outliers (i.e, z)
            The algorithm might discard more than n_outliers many data points
        :param epsilon: float, epsilon/n_clusters is the smallest fraction of cost that needs to be reduced
            per each local operation
        :param gamma: float, slackness on the number of outliers allowed, the algorithm will discard
            at most (1 + 1/self._ell) * (1 + gamma) / (1 - epsilon) * n_outliers many outliers.
        :param random_swap_in: int, number of swap-in choices sampled among opened facilities.
            If None, enumerate all swap-in choices among available facilities.
        :param random_swap_out: int, number of swap-out choices sampled among opened facilities.
            If None, enumerate all swap-out choices among opened facilities.
        :param debugging: boolean, whether to print extra information for debugging purpose
        """
        self._n_clusters = n_clusters
        self._n_outliers = n_outliers
        self._epsilon = epsilon
        self._gamma = gamma
        self._debugging = debugging
        self._random_swap_in = None if random_swap_in is None or random_swap_in <= 0 \
            else random_swap_in
        self._random_swap_out = None if random_swap_out is None or random_swap_out <= 0 \
            else random_swap_out

        self._assignmenter = None
        self._cluster_centers = None
        self._cluster_center_idxs = None
        self._clusters = None
        self._outlier_idxs = None
        self._inlier_idxs = None
        self._facility_recourse = 0
        self._client_recourse = 0

    @property
    def cluster_centers(self):
        return self._cluster_centers

    @property
    def clusters(self):
        return self._clusters

    @property
    def cluster_center_indices(self):
        return self._cluster_center_idxs

    @property
    def outlier_indices(self):
        return self._outlier_idxs

    @property
    def inlier_indices(self):
        return self._inlier_idxs

    @property
    def assignment(self):
        if not self._assignmenter:
            raise NotFittedError
        return self._assignmenter.assignment

    @property
    def facility_recourse(self):
        return self._facility_recourse

    @property
    def client_recourse(self):
        return self._client_recourse

    @property
    def cost(self):
        if not self._assignmenter:
            raise NotFittedError
        return self._assignmenter.cost_vec[self.inlier_indices].sum()

    def fit(self, C, F=None, distances=None, init_p=None):
        """
        This method *simulates* the online (k,z)-median algorithm by first read in the whole
        data set C and iterate through it.
        :param C: array of shape=(n_clients, n_features)
        :param F: array of shape=(n_facilities, n_features)
        :param distances: array of shape=(n_clients, n_facilities), distance[j, i] is the distance
            between client j and facility i. If None, then the distance will be computed on-the-fly.
        :param init_p: float, the init threshold parameter for the penalty distance
        :return self:
        """
        # initialize parameters
        n_clients, n_features = C.shape
        if F is None:
            F = C
        n_facilities, _ = F.shape
        n_clusters = min(self._n_clusters, n_facilities, n_clients)
        n_outliers = min(self._n_outliers, n_clients)

        # creat the distance matrix
        if init_p is None:
            if n_outliers == 0:
                p = np.inf
            elif self._gamma == 0:
                p = 1 / (3 * n_outliers)
            else:
                p = min(1 / (3 * n_outliers), 1 / (self._gamma * n_outliers))
        else:
            p = init_p
        dist = DistanceMatrix(C, F, dist_mat=distances, p=p)

        # initialization: assign the first k client to k nearest facilities
        debug_print("Initialization with p = {} ...".format(p), self._debugging)
        arrived_clients = list(range(n_clusters))
        opened_facilities, initial_asgn = _trivial_k_median(clients=C[arrived_clients],
                                                            facilities=F)
        self._assignmenter = Assignmenter(C, F,
                                          opened_facilities_idxs=opened_facilities,
                                          next_client=n_clusters,
                                          dist_mat=dist)
        prev_cost = self._assignmenter.cost

        # accepting subsequent clients
        n_outliers_thresh = 2 * (1 + self._gamma) / (1 - self._epsilon) * self._n_outliers
        debug_print("Tolerate at most {} outliers".format(n_outliers_thresh), self._debugging)
        for j in range(n_clusters, n_clients):
            # Accommodate the new arrived client j by assigning it to its nearest facility.
            self._assignmenter.arrive(j)

            # If client j doesn't increase cost too much, then skip local search
            if self._assignmenter.cost < 2 * prev_cost:
                debug_print("- Skip local search for client {}".format(j), self._debugging)
                continue

            # update the solution via local search
            debug_print("- Begin local search for client {} with beginning cost {} ...".format(j, self._assignmenter.cost),
                        self._debugging)
            rho = self._epsilon / n_clusters
            while True:
                ok, f_rec, c_rec = self._local_search(rho=rho,
                                                      assignmenter=self._assignmenter)
                while ok:
                    self._facility_recourse += f_rec
                    self._client_recourse += c_rec
                    ok, f_rec, c_rec = self._local_search(rho=rho,
                                                          assignmenter=self._assignmenter)
                _, d = self._assignmenter.nearest_facility(list(range(j)))

                # control the number of outliers
                if np.count_nonzero(d >= p) >= n_outliers_thresh:
                    p = (1 + self._epsilon) * p
                    dist.set_clip(p)
                    self._assignmenter.refresh_cache()
                    debug_print("-- Update p to {} ... ".format(p), self._debugging)
                else:
                    break

            prev_cost = self._assignmenter.cost
            debug_print("-- local search finished for client {} with ending cost {}"
                        .format(j, self._assignmenter.cost), self._debugging)

        # collect info of the final clustering
        opened_idxs_lst = list(self._assignmenter.opened_idxs)
        self._cluster_centers = F[opened_idxs_lst]
        self._cluster_center_idxs = np.array(opened_idxs_lst)
        self._outlier_idxs = np.where(self._assignmenter.cost_vec >= p)[0]
        self._inlier_idxs = np.where(self._assignmenter.cost_vec < p)[0]
        inliers = np.logical_not(np.isin(np.arange(len(C)), self._outlier_idxs))
        self._clusters = [np.where(np.logical_and(self._assignmenter == i, inliers))[0] for i in opened_idxs_lst]
        debug_print("Final cost = {}, #ouliers = {}".format(self.cost, len(self.outlier_indices)), self._debugging)

        return self

    def _local_search(self, rho, assignmenter):
        """conduct one step of (rho * cost)-efficient swap"""
        # sample or enumerate all possible swap-out choices
        swap_out_choices = np.array(list(assignmenter.opened_idxs))
        np.random.shuffle(swap_out_choices)
        if self._random_swap_out is not None:
            n_swaps = min(self._random_swap_out, len(swap_out_choices))
            swap_out_choices = swap_out_choices[:n_swaps]

        # sample or enumerate all swap-in choices
        swap_in_choices = np.array(list(assignmenter.closed_idxs))
        np.random.shuffle(swap_in_choices)
        if self._random_swap_in is not None:
            n_swaps = min(self._random_swap_in, len(swap_in_choices))
            swap_in_choices = swap_in_choices[:n_swaps]

        for swap_in in swap_in_choices:
            swap_out, reassigned, connection, cost = assignmenter.can_swap(
                swap_in, swap_out_choices,
                cost_thresh=rho * assignmenter.cost,
                avg_cost_thresh=0
            )
            prev_cost = assignmenter.cost
            if reassigned is not None:
                assignmenter.swap(swap_out, swap_in, reassigned, connection, cost)
                assert prev_cost > assignmenter.cost
                debug_print("-- Local search: swap out {} and swap in {}, cost decreases from {} to {}"
                            .format(swap_out, swap_in, prev_cost, assignmenter.cost),self._debugging)
                return True, 2, len(reassigned)
        return False, None, None

    def update(self, clients):
        """ accept new clients """
        raise NotImplementedError("The `update` method has not been implemented yet.")
