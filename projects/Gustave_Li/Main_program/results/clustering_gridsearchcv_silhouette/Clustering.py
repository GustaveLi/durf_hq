# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 16:36:49 2021

@author: gqmli
"""

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hdbscan import HDBSCAN
from sklearn_extra.cluster import KMedoids
import argparse

# Add a parser to parse the clustering method and target provided in the shell 
parser = argparse.ArgumentParser()
parser.add_argument('method', type=str, help='Method used for clustering')
parser.add_argument('target', type=str, help='Target filename used for clustering')
args = parser.parse_args()
method_name = args.method
target_name = args.target

# File path to read files
read_dir = '/xspace/hl4212/durf_hq/projects/Gustave_Li/Main_program/results/dimensionality_reduction'

# File path to write files
write_dir = '/xspace/hl4212/durf_hq/projects/Gustave_Li/Main_program/results/clustering'

# Load array for clustering
file_path = f'{read_dir}/dimreduct_{target_name}.npy'
arr = np.load(file_path)
    
# Define the silhouette scorer for gridsearch
def sil_score(estimator, X, y=None):
    labels = estimator.fit_predict(X)
    score = silhouette_score(X, labels)
    return score
# Select estemators and hyperparameters    
if method_name.lower() == 'kmeans':
    param_grid = [
        {'n_clusters': list(range(2,21))}
        ]
    estimator = KMeans()
    
elif method_name.lower() == 'kmedoids':
    param_grid = [
        {'n_clusters': list(range(2,21))}
        ]
    estimator = KMedoids()
    
elif method_name.lower() == 'hdbscan':
    param_grid = [
        {'min_cluster_size': list(range(5, 106, 20)),
         'min_samples': [None, 1, 10, 50, 500],
         'cluster_selection_epsilon': [0, 0.5, 1, 100]
         }
        ]
    estimator = HDBSCAN()
    
elif method_name.lower() == 'gmm':
    param_grid = [
        {'n_components': list(range(2, 23, 5)),
         'n_init': [10]
         }
        ]
    estimator = GaussianMixture()

# Do grid search to determine best hyperparameters combination, print out
grid_search = GridSearchCV(estimator, param_grid, scoring=sil_score, \
                           cv=[(slice(None), slice(None))], n_jobs=(-1), return_train_score=False)
grid_search.fit(arr)
best_estimator = grid_search.best_estimator_
best_params = grid_search.best_params_
#cvres = grid_search.cv_results_
print(best_params)
#print(cvres)

# Perform clustering and obtain labels and cluster centers, HDBSCAN doesn't have cluster centers
best_labels = best_estimator.fit_predict(arr)

if method_name.lower() == 'kmeans' or method_name.lower() == 'kmedoids':
    cluster_centers = best_estimator.cluster_centers_
elif method_name.lower() == 'gmm':
    cluster_centers = best_estimator.means_

# Save labels and ccenters back to disk
labels_path = f'{write_dir}/{target_name}_{method_name}_labels.npy'
np.save(labels_path, best_labels)

if method_name.lower() != 'hdbscan':
    cluster_path = f'{write_dir}/{target_name}_{method_name}_ccenters.npy'
    np.save(cluster_path, cluster_centers)