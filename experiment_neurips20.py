import numpy as np
from sklearn.cluster import k_means
from sklearn.metrics import pairwise_distances
from time import time

from okzm import OnlineKZMed
from utils import compute_cost, kz_median, kz_means, kzmedian_cost_, coreset
from utils import gaussian_mixture, add_outliers, get_realworld_data

# Parameters
n_clients = 20000
n_available_facilities = 4000
n_clusters = 20
n_outliers = 100
n_true_outliers = 50
data_name = 'gmm'
# data_name = 'covertype'

if data_name != 'gmm':
    print("Read in data set {} and shuffle it ... ".format(data_name))
    X = get_realworld_data(data_name)
    n_features = X.shape[1]
    np.random.shuffle(X)

    # take the first 100000 clients
    print("Take the first {} data points as client set ... ".format(n_clients))
    C = X[:n_clients]
    C = add_outliers(C, n_outliers=n_true_outliers)
else:
    print("Create synthesized GMM data of size {} and shuffle it ... ".format(n_clients))
    C = gaussian_mixture(n_samples=n_clients, n_clusters=n_clusters, n_outliers=n_true_outliers, n_features=5)

# init facility positions
print("\nCreate {} potential facility locations via pre-clustering or coreset ... ".format(n_available_facilities))

# F = k_means_lloyd(C, n_clusters=n_facilities)
t1_start = time()
# km = KMeans(n_clusters=n_facilities, n_init=1, max_iter=100)
# km.fit(C)
# F = km.cluster_centers_
F, _, _ = coreset(C, size=n_available_facilities, n_seeds=100, n_outliers=0)
pre_clustering_time = time() - t1_start
print("Pre-clustering takes {0:.2f} secs".format(pre_clustering_time))

if n_clients * n_available_facilities > 2e7:
    dist_mat = None
else:
    print("\nPre-compute distance matrix ... ")
    t2_start = time()
    dist_mat = pairwise_distances(C, F)
    dist_mat_time = time() - t2_start
    print("Computing distance matrix takes {0:.2f} secs".format(dist_mat_time))

# Run!
t3_start = time()
verbose = False
okzm = OnlineKZMed(n_clusters=n_clusters, n_outliers=n_outliers,
                   gamma=0, ell=1, epsilon=0.1,
                   random_swap_in=400,
                   random_swap_out=1,
                   debugging=verbose)
print("\nStart fitting the OKZM model ... ")
okzm.fit(C, F, dist_mat, init_p=100)
okzm_time = time() - t3_start
okzm_cost = compute_cost(C, okzm.cluster_centers, cost_func=kzmedian_cost_, remove_outliers=n_outliers)
print("\nDone. cost={} (removing {} outliers) or {} (removing {} outliers)"
      .format(okzm.cost, len(okzm.outlier_indices), okzm_cost, n_outliers))
print("Facility recourse={}, client recourse={}".format(okzm.facility_recourse, okzm.client_recourse))
print("Fit OKZM takes {0:.2f} secs".format(okzm_time))

# Compared with offline k-median-- method
t4_start = time()
offline_kzmed_centers = kz_median(C, n_clusters=n_clusters, n_outliers=n_outliers)
kzmed_time = time() - t4_start
offline_kzmed_cost = compute_cost(C, offline_kzmed_centers, cost_func=kzmedian_cost_, remove_outliers=n_outliers)
print("\nOffline k-median-- method gives cost {} (removing {} outliers)".
      format(offline_kzmed_cost, n_outliers))
print("online cost / Offline kz-median cost = {} (removing {} outliers)".
      format(okzm_cost / offline_kzmed_cost, n_outliers))
print("Fit offline kz-median takes {0:.2f} secs".format(kzmed_time))

# Compared with offline k-means-- method
t5_start = time()
offline_kzmeans_centers = kz_means(C, n_clusters=n_clusters, n_outliers=n_outliers)
kzmeans_time = time() - t5_start
offline_kzmeans_cost = compute_cost(C, offline_kzmeans_centers, cost_func=kzmedian_cost_, remove_outliers=n_outliers)
print("\nOffline k-means-- method gives cost {} (removing {} outliers)".
      format(offline_kzmeans_cost, n_outliers))
print("online cost / Offline kz-means cost = {} (removing {} outliers)".
      format(okzm_cost / offline_kzmeans_cost, n_outliers))
print("Fit offline kz-means takes {0:.2f} secs".format(kzmeans_time))

# Compared with offline k-means method
t6_start = time()
offline_kmeans_centers, _, _ = k_means(C, n_clusters=n_clusters, return_n_iter=False)
kmeans_time = time() - t6_start
offline_kmeans_cost = compute_cost(C, offline_kmeans_centers, cost_func=kzmedian_cost_, remove_outliers=n_outliers)
print("\nOffline k-means method gives cost {} (removing {} outliers)".
      format(offline_kmeans_cost, n_outliers))
print("online cost / Offline k-means cost = {} (removing {} outliers)".
      format(okzm_cost / offline_kmeans_cost, n_outliers))
print("Fit offline k-means takes {0:.2f} secs".format(kmeans_time))
