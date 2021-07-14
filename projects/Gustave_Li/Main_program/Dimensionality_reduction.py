from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS, TSNE
import numpy as np
import argparse

# Add a parser to parse the dimreduct method provided in the shell 
parser = argparse.ArgumentParser()
parser.add_argument('method', type=str, help='Method used for dim_reduct')
args = parser.parse_args()
method_name=args.method

# Specify target directory for file reading and writing
results_dir = '/xspace/hl4212/durf_hq/projects/Gustave_Li/Main_program/results'

# Load discriptors for dimensionality reduction
descriptor_path = f'{results_dir}/descriptors_arr.npy'
d_array = np.load(descriptor_path)

if method_name.lower() == 'pca':
    pca = PCA(n_components=2)
    result_arr = pca.fit_transform(d_array)
elif method_name.lower() == 'kpca':
    kpca = KernelPCA(n_components=2, kernel='poly')
    result_arr = kpca.fit_transform(d_array)
elif method_name.lower() == 'mds':
    mds = MDS(n_components=2)
    result_arr = mds.fit_transform(d_array)
elif method_name.lower() == 'tsne_standard':
    tsne_std = TSNE()
    result_arr = tsne_std.fit_transform(d_array)
elif method_name.lower() == 'tsne_optimized':
    tsne_optm = TSNE(n_components=2, perplexity=30, learning_rate=(100000//12), init='pca')
    result_arr = tsne_optm.fit_transform(d_array)

# Save the data back to disk
result_path = f'{results_dir}/dimreduct_{method_name}.npy'
np.save(result_path, result_arr)


