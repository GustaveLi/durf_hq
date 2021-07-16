from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
import numpy as np
import argparse

# Add a parser to parse the dimreduct method provided in the shell 
parser = argparse.ArgumentParser()
parser.add_argument('method', type=str, help='Method used for dim_reduct')
args = parser.parse_args()
method_name=args.method

# Specify target directory for file reading and writing
results_dir = '/xspace/hl4212/durf_hq/projects/Gustave_Li/Main_program/results/Newer_version'

# Load discriptors for dimensionality reduction
descriptor_path = f'{results_dir}/descriptors_arr.npy'
d_array = np.load(descriptor_path)

if method_name.lower() == 'pca':
    pca = PCA(n_components=2)
    result_arr = pca.fit_transform(d_array)
    
elif method_name.lower() == 'kpca_poly':
    kpca = KernelPCA(n_components=2, kernel='poly')
    result_arr = kpca.fit_transform(d_array)
    
elif method_name.lower() == 'kpca_rbf':
    kpca = KernelPCA(n_components=2, kernel='rbf')
    result_arr = kpca.fit_transform(d_array)
    
elif method_name.lower() == 'tsne_standard':
    tsne_std = TSNE()
    result_arr = tsne_std.fit_transform(d_array)
    
elif method_name.lower() == 'tsne_optimized':
    tsne_optm = TSNE(n_components=2, perplexity=30, learning_rate=(len(d_array)//12), \
                     init='pca')
    result_arr = tsne_optm.fit_transform(d_array)
    
elif method_name.lower() == 'tsne_xyz':
    # Load the xyz file on disk
    xyz_dir = '/gpfsnyu/scratch/hl4212'
    xyz_path = f'{xyz_dir}/xyz_aligned.npy'
    xyz = np.load(xyz_path)
    
    xyz_pipeline = Pipeline([
        #('pca', PCA(n_components=50)),
        ('tSNE_opt', TSNE(n_components=2,
                          learning_rate=(len(xyz)//12), 
                          perplexity=100)),
        ])
    
    result_arr = xyz_pipeline.fit_transform(xyz)
    
# Save the data back to disk
result_path = f'{results_dir}/dimreduct_{method_name}.npy'
np.save(result_path, result_arr)


