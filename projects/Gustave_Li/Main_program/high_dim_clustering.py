import argparse
import numpy as np
from tools.clustering import Cluster_inspection, High_dim_clustering

parser = argparse.ArgumentParser()
parser.add_argument('clustering_method', type=str, help='Method used for clustering')
parser.add_argument('dimred_method', type=str, help='Method used for dimensionality reduction')
parser.add_argument('index', type=int, help='Index of parameters')
parser.add_argument('threshold', type=float, help='The threshold value for RMSD')

args = parser.parse_args()
clustering_method = args.clustering_method
dimred_method = args.dimred_method
index = args.index
threshold = args.threshold

high_dim = High_dim_clustering(100000, dimred_method, clustering_method, index)
center_idx, labels = high_dim.high_dim_labels(threshold)

results_dir = '/xspace/hl4212/results/high_dim_clustering'
center_idx_path = f'{results_dir}/{dimred_method}_{clustering_method}_{index}_{threshold}_CenterIdx'
labels_path = f'{results_dir}/{dimred_method}_{clustering_method}_{index}_{threshold}_Labels'

np.save(center_idx_path, center_idx)
np.save(labels_path, labels)