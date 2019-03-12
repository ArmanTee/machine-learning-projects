### 1 import blobs.csv

import pandas as pd
dataset_1 = pd.read_csv('Notes_from_unsupervised_learning/blobs.csv')[:80].values

%matplotlib inline
## Help in plotting
import Notes_from_unsupervised_learning.dbscan_lab_helper as helper

helper.plot_dataset(dataset_1)

#TODO: Import sklearn's cluster module
from sklearn import cluster

#TODO: create an instance of DBSCAN
dbscan = cluster.DBSCAN()
#TODO: use DBSCAN's fit_predict to return clustering labels for dataset_1
clustering_labels_1 = dbscan.fit_predict(dataset_1)


helper.plot_clustered_dataset(dataset_1, clustering_labels_1)

# Plot clustering with neighborhoods
helper.plot_clustered_dataset(dataset_1, clustering_labels_1, neighborhood=True)

### increasing epsilon###

# TODO: increase the value of epsilon to allow DBSCAN to find three clusters in the dataset
epsilon=1.5

# Cluster
dbscan = cluster.DBSCAN(eps=epsilon)
clustering_labels_2 = dbscan.fit_predict(dataset_1)

# Plot
helper.plot_clustered_dataset(dataset_1, clustering_labels_2, neighborhood=True, epsilon=epsilon)


#### DATA SET 2 ########

dataset_2 = pd.read_csv('Notes_from_unsupervised_learning/varied.csv')[:300].values

# Plot
helper.plot_dataset(dataset_2, xlim=(-14, 5), ylim=(-12, 7))

# Cluster with DBSCAN
# TODO: Create a new isntance of DBSCAN
dbscan = cluster.DBSCAN()
# TODO: use DBSCAN's fit_predict to return clustering labels for dataset_2
clustering_labels_3 = dbscan.fit_predict(dataset_2)


# Plot
helper.plot_clustered_dataset(dataset_2,
                              clustering_labels_3,
                              xlim=(-14, 5),
                              ylim=(-12, 7),
                              neighborhood=True,
                              epsilon=0.5)



# TODO: Experiment with different values for eps and min_samples to find a suitable clustering for the dataset
eps= 1
min_samples=3

# Cluster with DBSCAN
dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples)
clustering_labels_4 = dbscan.fit_predict(dataset_2)

# Plot
helper.plot_clustered_dataset(dataset_2,
                              clustering_labels_4,
                              xlim=(-14, 5),
                              ylim=(-12, 7),
                              neighborhood=True,
                              epsilon=0.5)


### Hueristics Experiment with DBSCAN parameter
eps_values = [0.3, 0.5, 1, 1.3, 1.5]
min_samples_values = [2, 5, 10, 20, 80]

helper.plot_dbscan_grid(dataset_2, eps_values, min_samples_values)
