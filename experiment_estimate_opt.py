# -*- coding: utf-8 -*-
"""
This code is for estimating the offline OPT on the data set.
"""
import numpy as np
from datetime import datetime
from sklearn.metrics import pairwise_distances
from time import time

from utils import compute_cost, kz_means, kzmedian_cost_, coreset
from utils import gaussian_mixture, add_outliers, get_realworld_data

# Parameters
n_clients = 10000
n_available_facilities = 5000
n_clusters_range = [10, 50, 100]
n_outliers = 200
F_status = 'dynamic'
# F_status = 'static'
z_status = 'static'
# z_status = 'dynamic'
if z_status == 'static':
    n_outliers_func = lambda _: n_outliers
else:
    n_outliers_func = lambda x: int(x / n_clients * n_outliers)
n_true_outliers = 0
random_state = None
# data_name = 'gmm'
# data_name = 'power'
data_name = 'shuttle'
# data_name = 'letter'
# data_name = 'covertype'
# data_name = 'skin'
now = datetime.now()
starting_time = now.strftime("%b-%d")

if data_name != 'gmm':
    print("Read in data set {} {} shuffle ... ".format(data_name, 'and' if random_state is not None else 'without'))
    X = get_realworld_data(data_name)
    if random_state is not None:
        np.random.seed(random_state)
        np.random.shuffle(X)

    # take the first 100000 clients
    print("Take the first {} data points (with dimension {}) as client set ... "
          .format(n_clients, X.shape[1]))
    C = X[:n_clients]
    C = add_outliers(C, n_outliers=n_true_outliers, dist_factor=150)
else:
    print("Create synthesized GMM data of size {} and shuffle it ... ".format(n_clients))
    n_clusters = 5
    C = gaussian_mixture(n_samples=n_clients, n_clusters=n_clusters, n_outliers=n_true_outliers, n_features=5,
                         outliers_dist_factor=50)
n_features = C.shape[1]
print("\nCreate {} potential facility locations via pre-clustering or coreset ... ".format(n_available_facilities))

t1_start = time()
if F_status == 'dynamic':
    F = C.copy()
else:
    F, _, _ = coreset(C, size=n_available_facilities, n_seeds=100, n_outliers=0)
pre_clustering_time = time() - t1_start
print("Pre-clustering takes {0:.2f} secs".format(pre_clustering_time))

if n_clients * n_available_facilities > 2e7:
    dist_mat = None
else:
    print("\nPre-compute distance matrix ... ")
    t2_start = time()
    dist_mat = pairwise_distances(C, C)
    dist_mat_time = time() - t2_start
    print("Computing distance matrix takes {0:.2f} secs".format(dist_mat_time))

# result
opt_window_size = 50
results_var_k = {
    'data_name': data_name,
    'opt_cost': [],
    'n_features': n_features,
    'n_clients': n_clients,
    'n_clusters_range': n_clusters_range,
    'n_outliers': n_outliers,
    'n_artificial_outliers': n_true_outliers,
    'rando_seed': random_state,
    'opt_window_size': opt_window_size,
    'opt_sample_point': []
}
print("\nOn dataset {}, try different k with n={}, {} z={}, z'={}"
      .format(data_name, n_clients, z_status, n_outliers, n_true_outliers))
for k in n_clusters_range:
    steps = np.arange(k + n_outliers, len(C), opt_window_size)
    results_var_k['opt_sample_point'].append(steps)
    print("\n===\n k={}, window size={}".format(k, opt_window_size))
    # Run!
    t3_start = time()
    estimated_opt = []
    for n in steps:
        kzm_costs = []
        arrived_C = C[:n]
        for i in range(5):
            z = n_outliers_func(n)
            offline_kzmeans_centers = kz_means(arrived_C, n_clusters=k, n_outliers=z)
            c = compute_cost(arrived_C, offline_kzmeans_centers,
                             cost_func=kzmedian_cost_, remove_outliers=z)
            kzm_costs.append(c)
        offline_kzmeans_cost = min(kzm_costs)
        print("\n--- estimated OPT[0:{}] (with z={}) = min({}) = {}"
              .format(n, z, kzm_costs, offline_kzmeans_cost))
        estimated_opt.append(offline_kzmeans_cost)
    results_var_k['opt_cost'].append(np.array(estimated_opt))
    kzmeans_time = time() - t3_start
    print("--- Done, takes {0:.2f} secs for k={1:d}".format(kzmeans_time, k))


filename = "kmeans--_for_dataset_{}_{}_z-star_{}_opt-wd_{}_rseed_{}_z_{}_{}"\
    .format(data_name, n_clients, n_true_outliers, opt_window_size, random_state, z_status, starting_time)
np.savez(filename, **results_var_k)
