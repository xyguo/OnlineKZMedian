import numpy as np
from okzm import OnlineKZMed, k_means_lloyd, k_median_lloyd
from utils.get_data import get_realworld_data
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

data_name = 'covertype'
print("Read in data set {} and shuffle it ... ".format(data_name))
X = get_realworld_data(data_name)
n_features = X.shape[1]
np.random.shuffle(X)

# take the first 100000 clients
n_clients = 5000
print("Take the first {} data points as client set ... ".format(n_clients))
C = X[:n_clients]

# init facility positions
n_facilities = 500
print("Create {} potential facility locations via pre-clustering done by k-means ... ".format(n_facilities))

# F = k_means_lloyd(C, n_clusters=n_facilities)
km = KMeans(n_clusters=n_facilities, n_init=1, max_iter=100)
km.fit(C)
F = km.cluster_centers_

print("Pre-compute distance matrix ... ")
dist_mat = pairwise_distances(C, F)

# Run!
verbose = True
n_clusters = 20
okzm = OnlineKZMed(n_clusters=n_clusters, n_outliers=100,
                   gamma=0, ell=1, epsilon=0.1, debugging=verbose)
print("Start fitting model ... ")
okzm.fit(C, F, dist_mat, init_p=10)

print("Done. cost={}, facility recourse={}, client recourse={}"
      .format(okzm.cost, okzm.facility_recourse, okzm.client_recourse))

