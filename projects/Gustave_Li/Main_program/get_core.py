import mdtraj as md
import numpy as np
import argparse
from tools.clustering import Cluster_inspection

parser = argparse.ArgumentParser()
parser.add_argument('clustering_method', type=str, help='Method used for clustering')
parser.add_argument('dimred_method', type=str, help='Method used for dimensionality reduction')
parser.add_argument('index', type=int, help='Index of parameters')
args = parser.parse_args()
clustering_method = args.clustering_method
dimred_method = args.dimred_method
index = args.index

file_dir = '/xspace/hl4212/DURF_datasets/triad_molecule'
traj_path_aligned = f'{file_dir}/triad_dataset_aligned.nc'
top_path = f'{file_dir}/triad_forcefield_ground.prmtop'
traj = md.load(traj_path_aligned, top=top_path)

inspector = Cluster_inspection(100000, dimred_method, clustering_method, index)
core_index = inspector.find_nearest_five()
core_index_arr = np.array(core_index).reshape(len(core_index)*5,)
del core_index

for i in core_index_arr:
    traj[i].save(f'{file_dir}/core/{i}.nc')
    