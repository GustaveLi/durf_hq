# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 16:36:49 2021

@author: gqmli
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
#from hdbscan import HDBSCAN
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

# Select estemators   
if method_name.lower() == 'kmeans':
    estimator = KMeans()
    
elif method_name.lower() == 'kmedoids':
    estimator = KMedoids(init='k-medoids++')
    
elif method_name.lower() == 'gmm':
    estimator = GaussianMixture(n_init=10)
    
# Loop for different components, get evaluators (inertia or bic) and corresponding cluster number
cluster_nums = []
evaluators = []

for a in range(2, 21, 2): 
    cluster_nums.append(a)
    
    try: # for kmeans & kmedoids
        estimator.set_params(n_clusters=a)
        estimator.fit_predict(arr)
        evaluator = estimator.inertia_
    except: # for gmm
        estimator.set_params(n_components=a)
        estimator.fit_predict(arr)
        evaluator = estimator.bic(arr)
    
    evaluators.append(evaluator)

# Convert the lists to numpy array & save to disk

cluster_num_arr = np.array(cluster_nums)
cluster_num_arr_path = f'{write_dir}/{target_name}_{method_name}_cluster_num_arr.npy'
np.save(cluster_num_arr_path, cluster_num_arr)

evaluator_arr = np.array(evaluators)
evaluator_arr_path =  f'{write_dir}/{target_name}_{method_name}_evaluator_arr.npy'
np.save(evaluator_arr_path, evaluator_arr)

    

