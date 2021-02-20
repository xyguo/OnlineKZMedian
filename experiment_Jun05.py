# -*- coding: utf-8 -*-
"""
This code is for producing the experiment result shown in our paper
"""
import numpy as np
from datetime import datetime
from sklearn.metrics import pairwise_distances
from time import time

from okzm import OnlineKZMed
from utils import compute_cost, kzmedian_cost_, coreset
from utils import gaussian_mixture, add_outliers, get_realworld_data

# Parameters
n_clients = 20000
n_available_facilities = 5000
n_clusters_range = [10, 50, 100]
n_outliers = 200
n_true_outliers = 0
n_outliers_func = lambda x: int(x / n_clients * n_outliers)
# n_outliers_func = None
z_status = 'static' if n_outliers_func is None else 'dynamic'
F_status = 'dynamic'
# F_status = 'static'
random_state = None
# data_name = 'gmm'
# data_name = 'covertype'
# data_name = 'letter'
data_name = 'skin'
# data_name = 'power'
# data_name = 'shuttle'
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
if F_status == 'dynamic':
    F = C.copy()
    F_is_C = True
    F_status = 'dynamic'
    print("Facility set F=C, dynamic.\n")
else:
    F, _, _ = coreset(C, size=n_available_facilities, n_seeds=100, n_outliers=0)
    F_is_C = False
    print("Facility set F=core set of size {}, static.\n".format(n_available_facilities))

if n_clients * n_available_facilities > 2e7:
    dist_mat = None
else:
    print("\nPre-compute distance matrix ... ")
    t2_start = time()
    dist_mat = pairwise_distances(C, C)
    dist_mat_time = time() - t2_start
    print("Computing distance matrix takes {0:.2f} secs".format(dist_mat_time))

# okzm parameters
alpha = 0.2
epsilon = 0.05
gamma = 1
print("OKZM model parameter: alpha={}, epsilon={}, gamma={}".format(alpha, epsilon, gamma))
# result
results_var_k = {
    'data_name': data_name,
    'cost_p': [],
    'cost_z': [],
    'recourse': [],
    'n_features': n_features,
    'n_clients': n_clients,
    'n_clusters_range': n_clusters_range,
    'n_outliers': n_outliers,
    'n_available_facilities': len(F),
    'n_artificial_outliers': n_true_outliers,
    'alpha': alpha,
    'epsilon': epsilon,
    'gamma': gamma
}
print("\nOn dataset {}, try different k with n={}, ({}) z={}, z'={}"
      .format(data_name, n_clients, z_status, n_outliers, n_true_outliers))
for k in n_clusters_range:
    print("\n===\n k={}".format(k))
    # Run!
    t3_start = time()
    verbose = True
    okzm = OnlineKZMed(n_clusters=k, n_outliers=n_outliers,
                       n_outliers_func=n_outliers_func,
                       gamma=gamma, epsilon=epsilon, alpha=alpha,
                       random_swap_in=None,
                       random_swap_out=None,
                       debugging=verbose,
                       record_stats=True)
    print("\n--- Start fitting the OKZM model ... ")
    okzm.fit(C, F, distances=dist_mat, init_p=None, F_is_C=F_is_C)
    okzm_time = time() - t3_start
    okzm_cost = compute_cost(C, okzm.cluster_centers, cost_func=kzmedian_cost_, remove_outliers=n_outliers)
    print("\n--- Done, on data set {}, cost={} (removing {} outliers) or {} (removing {} outliers)"
          .format(data_name, okzm.cost('p'), len(okzm.outlier_indices), okzm_cost, n_outliers))
    print("--- Facility recourse={}, client recourse={}".format(okzm.facility_recourse, okzm.client_recourse))
    print("--- Fit OKZM takes {0:.2f} secs".format(okzm_time))
    results_var_k['cost_p'].append(okzm.cost_p_stats.copy())
    results_var_k['cost_z'].append(okzm.cost_z_stats.copy())
    results_var_k['recourse'].append(okzm.recourse_stats.copy())


filename = "okzm_results_for_dataset_{}_{}_z-star_{}_alpha_{}_rseed_{}_F_{}_z_{}_{}"\
    .format(data_name, n_clients, n_true_outliers, alpha, random_state, F_status, z_status, starting_time)
np.savez(filename, **results_var_k)
