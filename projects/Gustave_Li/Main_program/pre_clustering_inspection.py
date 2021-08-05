import numpy as np
from tools.clustering import Clustering
from sklearn.preprocessing import MinMaxScaler
import argparse

# Add a parser to parse the clustering method and target provided in the shell 
parser = argparse.ArgumentParser()
parser.add_argument('method', type=str, help='Method used for clustering')
parser.add_argument('target', type=str, help='Target filename used for clustering')
parser.add_argument('instances_num', type=int, help='Instance number for clustering')
args = parser.parse_args()
method_name = args.method
target_name = args.target
instances_num = args.instances_num

# File path to read files
read_dir = '/xspace/hl4212/results/dimensionality_reduction'

# File path to write files
write_dir = '/xspace/hl4212/results/clustering'

# Load array for clustering
file_path = f'{read_dir}/dimreduct_{target_name}.npy'
arr = np.load(file_path)

# Load modules
evaluator = Clustering(arr, instances_num)
dispatcher = {'kmeans':evaluator.kmeans,
              'kmedoids':evaluator.kmedoids,
              'gmm':evaluator.gmm,
              'hdbscan':evaluator.hdbscan
              }

# Pre-clustering and calculate several benchmarks
results = []
labels = []

if method_name != 'hdbscan':
    ccenters_list = []
    for cluster_num in range(6,11):
        result, ccenters, label = dispatcher[method_name](cluster_num)
        results.append(result)
        ccenters_list.append(ccenters)
        labels.append(list(label))
        del result, ccenters, label
        
else:
    tags = []
    for eps in [1,10,100,500]:
        for min_clus_size in range(10,100,20):
            for min_sample in [1,10,100]:
                tag = f'{eps}_{min_clus_size}_{min_sample}_{instances_num}'
                result, label = dispatcher[method_name](min_clus_size, min_sample, eps)
                tags.append(tag)
                results.append(result)
                labels.append(list(label))
                del result, label, tag

# Save the results (and tags) back to disk
scaler = MinMaxScaler(feature_range=(0,10))

results_arr = scaler.fit_transform(results)
results_path = f'{write_dir}/{target_name}_{method_name}_{instances_num}_benchmarks.npy'
np.save(results_path,results_arr)

labels_arr = np.array(labels)
labels_path = f'{write_dir}/{target_name}_{method_name}_{instances_num}_labels.npy'
np.save(labels_path,labels_arr)

try:
    ccenters_arr = np.array(ccenters_list)
    ccenters_path = f'{write_dir}/{target_name}_{method_name}_{instances_num}_ccenters.npy'
    np.save(ccenters_path,ccenters_arr)
except:
    pass

try:
    tags_arr = np.array(tags)
    tags_path = f'{write_dir}/{target_name}_{method_name}_{instances_num}_tags.npy'
    np.save(tags_path,tags_arr)
except:
    pass
                
    
   


    

