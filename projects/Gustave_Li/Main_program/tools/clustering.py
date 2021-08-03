from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd


class Clustering():
    def __init__(self, dataset, num_instances):
        self.dataset = dataset[:num_instances, :]
        
    def kmeans(self, n_clusters):
        estimator = KMeans(n_clusters=n_clusters)
        label = estimator.fit_predict(self.dataset)
        ccenters = estimator.cluster_centers_
        inertia = estimator.inertia_
        sil_score = silhouette_score(self.dataset, label)
        return [inertia, sil_score], ccenters, label

    def kmedoids(self, n_clusters):
        estimator = KMedoids(n_clusters=n_clusters, init='k-medoids++')
        label = estimator.fit_predict(self.dataset)
        ccenters = estimator.cluster_centers_
        inertia = estimator.inertia_
        sil_score = silhouette_score(self.dataset, label)
        return [inertia, sil_score], ccenters, label
    
    def gmm(self, n_components):
        estimator = GaussianMixture(n_components=n_components, n_init=10)
        label = estimator.fit_predict(self.dataset)
        ccenters = estimator.means_
        aic = estimator.aic(self.dataset)
        bic = estimator.bic(self.dataset)
        sil_score = silhouette_score(self.dataset, label)
        return [aic, bic, sil_score], ccenters, label
    
    def hdbscan(self, min_clus_size, min_sam, clus_sel_eps):
        estimator = HDBSCAN(min_cluster_size=min_clus_size,
                            min_samples = min_sam,
                            cluster_selection_epsilon = clus_sel_eps)
        label = estimator.fit_predict(self.dataset)
        sil_score = silhouette_score(self.dataset, label)
        return [sil_score], label


class Cluster_inspection():
    def __init__(self, num_instances, dimred_method, clustering_method, index):
        """
        Initialize module

        Parameters
        ----------
        num_instances : TYPE Integer
            DESCRIPTION. Number of instances in the dataset used for clustering,
            (10000, 40000, 70000, 100000)
        dimred_method : TYPE String
            DESCRIPTION. Method used for dimensionality reduction
        clustering_method : TYPE String
            DESCRIPTION. Method used for clustering
        index : TYPE Integer
            DESCRIPTION. The hyperparameter index from the preclustering loop,
            can be obtained by inspecting the benchmark plot

        Returns
        -------
        None.

        """
        results_dir = '/xspace/hl4212/results'
        instance_path = f'{results_dir}/dimensionality_reduction/dimreduct_{dimred_method}.npy'
        self.instance_array = np.load(instance_path)[:num_instances, :]
        
        label_path = f'{results_dir}/clustering/{dimred_method}_{clustering_method}_{num_instances}_labels.npy'
        self.label_array = np.load(label_path)[index]
        
        center_path = f'{results_dir}/clustering/{dimred_method}_{clustering_method}_{num_instances}_ccenters.npy'
        self.center_array = np.load(center_path, allow_pickle=True)[index]
        
        self.num_instances = num_instances
        
    def cluster_population(self):
        cluster_idx, population = np.unique(self.label_array, return_counts=True)
        percentage = np.array(population/self.num_instances*100, dtype=int)
        pop_dict = dict(cluster_idx=cluster_idx, population=population,
                        percentage=percentage)
        
        pop_df = pd.DataFrame(pop_dict)
        return pop_df
    
    def find_nearest_five(self):
        x_coord = self.instance_array[:, 0]
        y_coord = self.instance_array[:, 1]
        nearest_five_all = [] # Nearst five instance of all cluster centers, compound list
        
        for i in np.unique(self.label_array):
            nearest_five = [] # Nearst five instance of the current cluster center
            dist = 0
            ccenter_x = self.center_array[i, 0]
            ccenter_y = self.center_array[i, 1]
            
            while len(nearest_five)<5:
                x_candidate = np.where((x_coord > ccenter_x-dist) & (x_coord < ccenter_x+dist))[0]
                y_candidate = np.where((y_coord > ccenter_y-dist) & (y_coord < ccenter_y+dist))[0]
                coord_candidate = list(np.intersect1d(x_candidate, y_candidate))
                nearest_five += coord_candidate
                nearest_five = list(np.unique(nearest_five))
                dist += 0.001
            
            nearest_five_all.append(nearest_five[:5])
        return nearest_five_all