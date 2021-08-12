import argparse
from tools.clustering import Cluster_inspection
from tools import plotting

parser = argparse.ArgumentParser()
parser.add_argument('clustering_method', type=str, help='Method used for clustering')
parser.add_argument('dimred_method', type=str, help='Method used for dimensionality reduction')
parser.add_argument('index', type=int, help='Index of parameters')
args = parser.parse_args()
clustering_method = args.clustering_method
dimred_method = args.dimred_method
index = args.index

inspector = Cluster_inspection(100000, dimred_method, clustering_method, index)
core_rmsd = inspector.core_rmsd(1000)
title = f'{dimred_method}_{clustering_method}_{index}'
plotting.rmsd_heatmap(core_rmsd, title, 1000, save=True)


    