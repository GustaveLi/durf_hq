from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
import numpy as np
import argparse

# Add a parser to parse the dimreduct method provided in the shell 
parser = argparse.ArgumentParser()
parser.add_argument('method', type=str, help='Method used for dim_reduct')
args = parser.parse_args()
method_name = args.method

# Specify target directory for file reading and writing
results_dir = '/xspace/hl4212/durf_hq/projects/Gustave_Li/Main_program/results/dimensionality_reduction'

# Load arrays for dimensionality reduction
if 'xyz' in method_name:
    xyz_dir = '/gpfsnyu/scratch/hl4212'
    xyz_path = f'{xyz_dir}/xyz_aligned.npy'
    arr = np.load(xyz_path)

else:
    descriptor_path = f'{results_dir}/descriptors_arr.npy'
    arr = np.load(descriptor_path)
    
# Start dimreduct
if method_name.lower() == 'pca' or method_name.lower() == 'pca_xyz':
    pca = PCA(n_components=2)
    result_arr = pca.fit_transform(arr)
    
elif method_name.lower() == 'kpca_poly' or method_name.lower() == 'kpca_poly_xyz':
    kpca = KernelPCA(n_components=2, kernel='poly')
    result_arr = kpca.fit_transform(arr)
    
elif method_name.lower() == 'kpca_rbf' or method_name.lower() == 'kpca_rbf_xyz':
    kpca = KernelPCA(n_components=2, kernel='rbf')
    result_arr = kpca.fit_transform(arr)
    
elif method_name.lower() == 'tsne_standard' or method_name.lower() == 'tsne_standard_xyz':
    tsne_std = TSNE()
    result_arr = tsne_std.fit_transform(arr)
    
elif method_name.lower() == 'tsne_optimized' or method_name.lower() == 'tsne_optimized_xyz':
    tsne_optm = TSNE(n_components=2, perplexity=50, learning_rate=(len(arr)//12), \
                     init='pca')
    result_arr = tsne_optm.fit_transform(arr)
     
# Save the data back to disk
#result_path = f'{results_dir}/dimreduct_{method_name}.npy'
#np.save(result_path, result_arr)


