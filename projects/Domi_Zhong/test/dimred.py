#!/usr/bin/env python3

from sklearn.decomposition import KernelPCA
import numpy as np

results_dir = '/xspace/hl4212/durf_hq/projects/Gustave_Li/Main_program/results'
descriptor_path = f'{results_dir}/descriptors_arr.npy'
result_path = f'{results_dir}/kpca.npy'


d_array = np.load(descriptor_path)
kpca = KernelPCA(n_components=2, kernel='rbf')
dimreduct_kpca = kpca.fit_transform(d_array)
np.save(result_path, dimreduct_kpca)
