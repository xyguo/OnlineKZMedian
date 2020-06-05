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
                 n_outliers_func=None,
                 epsilon=0.1, gamma=1, alpha=1,
                 random_swap_in=None, random_swap_out=None,
                 debugging=False, record_stats=False):
        """
        Implementation of the bi-criteria online (k,z)-median algorithm based on local search
        :param n_clusters: int, number of clusters (i.e, k).
            The algorithm will open at most n_clusters many facilities.
        :param n_outliers: int, number of outliers (i.e, z)
            The algorithm might discard more than n_outliers many data points
        :param n_outliers_func: callable that given an integer argument, return another number. Here the
            integer argument is the number of arrived clients, and the returned value is the number of
            acceptable outliers. This can implement n_outliers evolving with time.
            If None then the number of outliers is fixed to n_outliers.
            NOTE: if this parameter is not None, then it will override the effect of n_outlier.
        :param epsilon: float, epsilon/n_clusters is the smallest fraction of cost that needs to be reduced
            per each local operation
        :param gamma: float, slackness on the number of outliers allowed, the algorithm will discard
            at most (1 + 1/self._ell) * (1 + gamma) / (1 - epsilon) * n_outliers many outliers.
        :param alpha: float, tolerance on cost increasing. When a new client arrives, the algorithm
            will conduct local search only when cost has increased to (1+alpha) * (the cost after last local search)
        :param random_swap_in: int, number of swap-in choices sampled among opened facilities.
            If None, enumerate all swap-in choices among available facilities.
        :param random_swap_out: int, number of swap-out choices sampled among opened facilities.
            If None, enumerate all swap-out choices among opened facilities.
        :param debugging: boolean, whether to print extra information for debugging purpose
        :param record_stats: boolean, if True then record the cost and recourse during the whole execution
        """
        assert n_clusters >= 1
        self._n_clusters = n_clusters
        self._n_outliers = n_outliers
        self._n_outliers_func = n_outliers_func
        self._epsilon = epsilon
        self._gamma = gamma
        self._alpha = alpha
        self._debugging = debugging
        self._random_swap_in = None if random_swap_in is None or random_swap_in <= 0 \
            else random_swap_in
        self._random_swap_out = None if random_swap_out is None or random_swap_out <= 0 \
            else random_swap_out

        # vbls maintaining the solution
        self._assignmenter = None
        self._cluster_centers = None
        self._cluster_center_idxs = None
        self._outlier_idxs = None
        self._inlier_idxs = None
        self._facility_recourse = 0
        self._client_recourse = 0

        # bookkeeping vbls
        self._record = record_stats
        self._cost_p_stats = []
        self._recourse_stats = []
        self._cost_z_stats = []

    @property
    def n_clusters(self):
        return self._n_clusters

    @property
    def cluster_centers(self):
        return self._cluster_centers

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

    def cost(self, cost_type='p'):
        """This is the cost with outliers removed
        :param cost_type: str in {'p', z'}. If 'p', return the cost that's provably good, but may remove
            more than 'n_outliers'-many outliers. If 'z', return the cost with exactly 'n_outliers'-many
            outliers removed.
        """
        if not self._assignmenter:
            raise NotFittedError
        if cost_type == 'p':
            return self._assignmenter.cost_vec[self.inlier_indices].sum()
        elif cost_type == 'z':
            n = len(self._assignmenter.cost_vec)
            return np.sort(self._assignmenter.cost_vec)[:(n-self._n_outliers)].sum()

    @property
    def cost_z_stats(self):
        """This is the statistics of cost with exactly 'n_outliers'-many outliers removed.
        This value doesn't have provable guarantee.
        """
        return self._cost_z_stats

    @property
    def cost_p_stats(self):
        """This is the statistics of cost with provable guarantee, but it can possibly remove
        more than 'n_outliers'-manny outliers.
        """
        return self._cost_p_stats

    @property
    def recourse_stats(self):
        return self._recourse_stats

    def fit(self, C, F=None, distances=None, init_p=None, F_is_C=False):
        """
        This method *simulates* the online (k,z)-median algorithm by first read in the whole
        data set C and iterate through it.
        :param C: array of shape=(n_clients, n_features)
        :param F: array of shape=(n_facilities, n_features)
        :param distances: array of shape=(n_clients, n_facilities), distance[j, i] is the distance
            between client j and facility i. If None, then the distance will be computed on-the-fly.
        :param init_p: float, the init threshold parameter for the penalty distance
        :param F_is_C: bool, if True then F is also online: i.e., changing with C,
            This parameter should be set to True only if F is indeed C. Otherwise the behavior is undefined.
        :return self:
        """
        # initialize parameters and bookkeeping vbls
        n_clients, n_features = C.shape
        if F is None:
            F = C.copy()
        n_facilities, _ = F.shape
        n_clusters = min(self._n_clusters, n_facilities, n_clients)
        if self._n_outliers_func is not None:
            # n_outliers is changing with the number of arrived outliers
            n_outliers = self._n_outliers_func
        else:
            n_outliers = lambda _: min(self._n_outliers, n_clients)
        self._cost_p_stats = np.zeros(n_clients)
        self._cost_z_stats = np.zeros(n_clients)
        self._recourse_stats = np.zeros(n_clients, dtype=np.int)

        # creat the distance matrix
        if init_p is None:
            if n_outliers(n_clients) == 0:
                p = np.inf
            elif self._gamma == 0:
                p = 1 / (3 * n_outliers(n_clients))
            else:
                p = min(0.1, 1 / (10 * self._gamma * n_outliers(n_clients)))
        else:
            p = init_p
        dist = DistanceMatrix(C, F, dist_mat=distances)

        # initialization: assign the first k client to k nearest facilities
        debug_print("Initialization with p = {} ...".format(p), self._debugging)
        arrived_clients = list(range(n_clusters))
        init_n_arrived = len(arrived_clients)
        if F_is_C:
            opened_facilities = set(arrived_clients)
        else:
            opened_facilities, _ = _trivial_k_median(clients=C[arrived_clients],
                                                     facilities=F)
        self._assignmenter = Assignmenter(C, F,
                                          opened_facilities_idxs=opened_facilities,
                                          F_is_C=F_is_C,
                                          next_client=n_clusters,
                                          dist_mat=dist)
        init_outliers_thresh = 2 * (1 + self._gamma) / (1 - self._epsilon) * n_outliers(init_n_arrived) + 0.01

        # initialization: update p to be large enough and remember the current threshold cost
        d = self._assignmenter.cost_vec[:init_n_arrived]
        while np.count_nonzero(d >= 0.99*p) > init_outliers_thresh:
            p = 2 * p
            debug_print("-- Update p to {} ... ".format(p), self._debugging)
        # This vbl 'cost_after_last_local_search' records the cost under threshold distance
        # (but without removing outliers)
        cost_after_last_local_search = self._assignmenter.cost_p(p, remove=False)

        # initialize bookkeeping vbls
        self._facility_recourse = len(opened_facilities)
        self._client_recourse = init_n_arrived
        self._recourse_stats[:init_n_arrived] = self._facility_recourse
        self._cost_p_stats[:init_n_arrived] = cost_after_last_local_search
        self._cost_z_stats[:init_n_arrived] = self._assignmenter.cost_z(n_outliers(init_n_arrived))

        # accepting subsequent clients
        debug_print("Tolerate at most {} outliers".format(n_outliers(n_clients)), self._debugging)
        for j in range(n_clusters, n_clients):
            curr_n_outliers = n_outliers(j)
            n_outliers_thresh = 2 * (1 + self._gamma) / (1 - self._epsilon) * curr_n_outliers + 0.01
            # Accommodate the new arrived client j by assigning it to its nearest facility.
            self._assignmenter.arrive(j)

            # If client j doesn't increase cost too much, then skip local search
            d = self._assignmenter.cost_vec[:j+1]
            while np.count_nonzero(d >= 0.99*p) > n_outliers_thresh:
                p = 2 * p
                debug_print("-- Update p to {} ... ".format(p), self._debugging)
            if self._assignmenter.cost_p(p, remove=False) < (1 + self._alpha) * cost_after_last_local_search:
                debug_print("- Skip local search for client {}".format(j), self._debugging)
                # bookkeeping
                if self._record:
                    self._recourse_stats[j] = self._facility_recourse
                    self._cost_p_stats[j] = self._assignmenter.cost_p(p, remove=True)
                    self._cost_z_stats[j] = self._assignmenter.cost_z(curr_n_outliers)
                continue

            # update the solution via local search
            debug_print("- Begin local search for client {} with beginning cost_p={} and cost_z={} ..."
                        .format(j, self._assignmenter.cost_p(p, remove=False),
                                self._assignmenter.cost_z(curr_n_outliers)),
                        self._debugging)
            rho = self._epsilon / n_clusters
            while True:
                ok, f_rec, c_rec = self._local_search(rho=rho,
                                                      assignmenter=self._assignmenter, p=p)
                while ok:
                    self._facility_recourse += f_rec
                    self._client_recourse += c_rec
                    ok, f_rec, c_rec = self._local_search(rho=rho,
                                                          assignmenter=self._assignmenter, p=p)

                # control the number of outliers
                d = self._assignmenter.cost_vec[:j+1]
                if np.count_nonzero(d >= 0.99*p) > n_outliers_thresh:
                    p = 2 * p
                    debug_print("-- Update p to {} ... ".format(p), self._debugging)
                else:
                    break

            cost_after_last_local_search = self._assignmenter.cost_p(p, remove=False)
            debug_print("-- local search finished for client {} with ending cost_p={}, cost_z={}"
                        .format(j, cost_after_last_local_search, self._assignmenter.cost_z(curr_n_outliers)),
                        self._debugging)
            # bookkeeping
            self._recourse_stats[j] = self._facility_recourse
            self._cost_p_stats[j] = self._assignmenter.cost_p(p, remove=True)
            self._cost_z_stats[j] = self._assignmenter.cost_z(curr_n_outliers)

        # collect info of the final clustering
        opened_idxs_lst = list(self._assignmenter.opened_idxs)
        self._cluster_centers = F[opened_idxs_lst]
        self._cluster_center_idxs = np.array(opened_idxs_lst)
        self._outlier_idxs = np.where(self._assignmenter.cost_vec >= p)[0]
        self._inlier_idxs = np.where(self._assignmenter.cost_vec < p)[0]
        inliers = np.logical_not(np.isin(np.arange(len(C)), self._outlier_idxs))
        self._clusters = [np.where(np.logical_and(self._assignmenter == i, inliers))[0] for i in opened_idxs_lst]
        debug_print("Final cost = {}, #ouliers = {}".format(self.cost('p'), len(self.outlier_indices)), self._debugging)

        return self

    def _local_search(self, rho, assignmenter, p=None):
        """conduct one step of (rho * cost)-efficient swap
        :param rho: float at most 1. fraction of the reduction of the total total cost.
            A local swap will be conducted only when it decreases current cost by at least rho fraction.
        :param assignmenter: object of Assignmenter. Maintain the current solution.
        :param p: float, threshold parameter for calculating cost. p=None means no threshold.
            If not None, all the distances will be clipped to at most p when computing the cost.
        """
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

        cost_thresh = rho * assignmenter.cost_p(p, remove=False)
        for swap_in in swap_in_choices:
            swap_out, reassigned, connection, cost = assignmenter.can_swap(
                swap_in, swap_out_choices,
                cost_thresh=cost_thresh,
                avg_cost_thresh=0,
                p=p
            )
            if reassigned is not None:
                assignmenter.swap(swap_out, swap_in, reassigned, connection, cost)
                return True, 2, len(reassigned)
        return False, None, None

    def update(self, clients):
        """ accept new clients """
        raise NotImplementedError("The `update` method has not been implemented yet.")
