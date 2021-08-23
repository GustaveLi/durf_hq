from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score, accuracy_score
from operator import itemgetter 
import numpy as np
import pandas as pd
import mdtraj as md


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
        self.dimred_method = dimred_method
        self.clustering_method = clustering_method
        self.index = index
        
        
    def cluster_population(self):
        """
        Count the number of instances in each cluster, and calculates the instance number 
        in percentage. We expect a meaningful cluster to contain at least 5%
        of the total data points.

        Returns
        -------
        pop_df : TYPE Pandas DataFrame
            DESCRIPTION. A DataFrame that contains the index of cluster, number
            of instances, and percentage of instances.

        """
        cluster_idx, population = np.unique(self.label_array, return_counts=True)
        percentage = np.array(population/self.num_instances*100, dtype=int)
        pop_dict = dict(cluster_idx=cluster_idx, population=population,
                        percentage=percentage)
        
        pop_df = pd.DataFrame(pop_dict)
        return pop_df
    
    def find_nearest_n(self, n):
        """
        Find the index of n instance nearest to each cluster center.

        Parameters
        ----------
        n : TYPE Integer
            DESCRIPTION. Number of core instances in each cluster.

        Returns
        -------
        TYPE Numpy Array, shape=(num_of_clusters, n)
            DESCRIPTION. The result index array

        """
        x_coord = self.instance_array[:, 0]
        y_coord = self.instance_array[:, 1]
        nearest_n_all = [] # Nearst five instance of all cluster centers, compound list
        
        for i in np.unique(self.label_array):
            nearest_n = [] # Nearst five instance of the current cluster center
            dist = 0
            ccenter_x = self.center_array[i, 0]
            ccenter_y = self.center_array[i, 1]
            
            while len(nearest_n)<n:
                x_candidate = np.where((x_coord > ccenter_x-dist) & (x_coord < ccenter_x+dist))[0]
                y_candidate = np.where((y_coord > ccenter_y-dist) & (y_coord < ccenter_y+dist))[0]
                coord_candidate = list(np.intersect1d(x_candidate, y_candidate))
                nearest_n += coord_candidate
                nearest_n = list(np.unique(nearest_n))
                dist += 0.001
            
            nearest_n_all.append(nearest_n[:n])
        return np.array(nearest_n_all)
    
    def core_rmsd(self, n):
        """
        Generate index array from the find_nearest_n function, calculate the 
        pairwise-rmsd between the core instances

        Parameters
        ----------
        n : TYPE Integer
            DESCRIPTION. Number of core instances in each cluster.

        Returns
        -------
        pairwise_rmsd : TYPE Numpy array, shape=(n*cluster_num, n*cluster_num)
            DESCRIPTION. The Numpy array storing all the pairwise-rmsd. The unit
            is Angstrom

        """
        #top_dir = '/xspace/hl4212/DURF_datasets/triad_molecule'
        #top_path = f'{top_dir}/triad_forcefield_ground.prmtop'
        
        core_dir = '/xspace1/projects/CT_Landscape/data/triad_durf_dataset'
        nearest_n = Cluster_inspection.find_nearest_n(self, n).reshape(-1,)
        nearest_n_path = [f'{core_dir}/triad_durf_dataset_{int(np.floor(i/1000))}\
/triad_{i%1000}.pdb' for i in nearest_n]
        
        traj_core = md.load(nearest_n_path)
        
        pairwise_rmsd = np.empty((traj_core.n_frames, traj_core.n_frames))
        for i in range(traj_core.n_frames):
            pairwise_rmsd[i] = md.rmsd(traj_core, traj_core, i)
        
        return pairwise_rmsd
    
    def get_sorted_FeatureSpaceData(self):
        """
        Sort the dimensioanlity reduction data in feature space based on clustering result (labels)

        Returns
        -------
        sorted_lst_x : TYPE (Nested) List
            DESCRIPTION. The sorted list containing feature space data from the 1st dimension.
        sorted_lst_y : TYPE (Nested) List
            DESCRIPTION. The sorted list containing feature space data from the 2nd dimension.

        """
        # Load the feature space & the clustering labels data
        dimreduct_dir = '/xspace/hl4212/results/dimensionality_reduction'
        dimreduct_path = f'{dimreduct_dir}/dimreduct_{self.dimred_method}.npy'
        dimreduct_data = np.load(dimreduct_path)[:, :self.num_instances]
        
        label_data = self.label_array
        
        cluster_num = self.index+6
        sorted_lst_x = [[] for _ in range(cluster_num)]
        sorted_lst_y = [[] for _ in range(cluster_num)]
        
        # Loop through instances and put them in different lists according to the labels
        for i in range(self.num_instances):
            sorted_lst_x[label_data[i]].append(list(dimreduct_data[i])[0])
            sorted_lst_y[label_data[i]].append(list(dimreduct_data[i])[1])
        
        return sorted_lst_x, sorted_lst_y
    
    def PearsonR(self, n=None, dim='dim1'):
        """
        Take n instances equally spaced to each other, from the sorted data 
        generated by Cluster_inspection.get_sorted_FeatureSpaceData method. 
        Calculate the Pearson Correlation Coefficient between different clusters
        on dim 1 or dim 2.

        Parameters
        ----------
        n : TYPE Integer, optional
            DESCRIPTION Number of instances in each cluster chosen for Pearson
            R calculation. This number should be less than the smallest cluster
            size. The default is None. When this parameter is set as None, the 
            number of the smallest cluster size will be used.
            
        dim : TYPE String ('dim1' or 'dim2'), optional
            DESCRIPTION Specify the dimensionality. 'dim1' represents PC1,
            'dim2' represents PC2. The default is 'dim1'.

        Returns
        -------
        pearson : TYPE Numpy array, shape=(cluster_num, cluster_num)
            DESCRIPTION. The cluster-wise pearson correlation coefficient array 
            on 1st or 2nd dimension of the feature space

        """
        sorted_x, sorted_y = Cluster_inspection.get_sorted_FeatureSpaceData(self)
        
        # Pick the number of n when it is set to None
        if n == None:
            pop_df = Cluster_inspection.cluster_population(self)
            n = pop_df['population'].min()

        cluster_num = self.index+6
        truncated_x = np.empty([cluster_num, n], dtype = 'int')
        truncated_y = np.empty([cluster_num, n], dtype = 'int')
        
        # Pearson r calculation needs equal size of data
        for i in range(cluster_num):
            idx = np.linspace(0, len(sorted_x[i]), n, endpoint=False, dtype='int')
            truncated_x[i] = np.array(itemgetter(*idx)(sorted_x[i]))
            truncated_y[i] = np.array(itemgetter(*idx)(sorted_y[i]))
        del sorted_x, sorted_y
        pearson_x = np.corrcoef(truncated_x)
        pearson_y = np.corrcoef(truncated_y)
        if dim=='dim1':
            pearson = pearson_x
        elif dim=='dim2':
            pearson = pearson_y
        return pearson
    
    def cluster_quality_sequence(self):
        """
        Define cluster quality as the sum of absolute value of Pearson R between 
        the given cluster and other clusters (both on dim1 and dim2). Smaller the
        number greater the quality.

        Returns
        -------
        idx_list : TYPE List, len=number_of_clusters
            DESCRIPTION. The list of cluster labels sequenced by cluster quality.
            Cluster with better quality comes first.

        """
        PearsonR_dim1 = Cluster_inspection.PearsonR(self, dim='dim1')
        PearsonR_dim2 = Cluster_inspection.PearsonR(self, dim='dim2')
        
        scoring_dim1 = np.sum(np.abs(PearsonR_dim1), axis=1)
        scoring_dim2 = np.sum(np.abs(PearsonR_dim2), axis=1)
        scoring = scoring_dim1+scoring_dim2
        
        idx_list = []
        for i in np.sort(scoring):
            idx = int(np.where(scoring==i)[0])
            idx_list.append(idx)
            
        return idx_list
    
class High_dim_clustering(Cluster_inspection):
    
    def __init__(self, num_instances, dimred_method, clustering_method, index, threshold):
        super().__init__(num_instances, dimred_method, clustering_method, index)
        self.threshold = threshold
        
        # If the labeling process is finished, load the relative files
        try:
            highD_results_dir = '/xspace/hl4212/results/high_dim_clustering'        
            highD_labels_path = f'{highD_results_dir}/{dimred_method}_{clustering_method}_{index}_{threshold}_Labels.npy'
            self.highD_labels_arr = np.load(highD_labels_path)
            highD_ccenter_idx_path = f'{highD_results_dir}/{dimred_method}_{clustering_method}_{index}_{threshold}_CenterIdx.npy'
            self.highD_ccenter_idx_arr = np.load(highD_ccenter_idx_path)
        except:
            pass
        
        
    def rmsd(self, ref_frame):
        """
        Calculate the RMSD of the triad trajectory with respect to a reference
        frame

        Parameters
        ----------
        ref_frame : TYPE Integer
            DESCRIPTION. The reference frame number

        Returns
        -------
        rmsd : TYPE Numpy array, shape=(num_of_instances, )
            DESCRIPTION. The RMSD array

        """
        # Load triad data
        triad_dir = '/xspace/hl4212/DURF_datasets/triad_molecule'
        triad_path = f'{triad_dir}/triad_dataset.nc'
        top_path = f'{triad_dir}/triad_forcefield_ground.prmtop'
        traj = md.load(triad_path, top=top_path)
        
        # Calculate rmsd with respect to the given reference frame
        rmsd = md.rmsd(traj, traj, frame=ref_frame)
        return rmsd
    
    def high_dim_labels(self):
        """
        Try to assign instances to different clusters in high dimensional space
        (the original xyz space). We say the two instances are in the same 
        cluster when their RMSD is less than a given threshold. Clusters with better 
        quality (see Cluster_inspection.cluster_quality_sequence( ) for more information)
        will be assigned members first.

        Returns
        -------
        center_index : TYPE Numpy array, shape=(num_of_clusters, )
            DESCRIPTION. The triad index of each cluster center (which is the 
            instance closest to the cluster center in 2D feature space)
        high_dim_labels : TYPE TYPE Numpy array, shape=(num_of_instances, )
            DESCRIPTION.luster label of each instance in higher dimensional space.
            -1 means the instance belongs to no cluster under the current threshold

        """
        # Initialize the labels and set to -1. If it belongs to no cluster after
        # calculation, the label will remain -1
        high_dim_labels = np.ones((self.num_instances, ), dtype=int)*(-1)
        
        # Obtain index of the instance closest to each cluster center
        # Cluster of better quality (in 2D space) will be assigned members first
        center_index = High_dim_clustering.find_nearest_n(self, 1).reshape(-1,)
        cluster_sequence = High_dim_clustering.cluster_quality_sequence(self)
        
        for i in cluster_sequence:
            ref_frame = center_index[i]
            rmsd = High_dim_clustering.rmsd(self, ref_frame)
            high_dim_labels[np.where((rmsd < self.threshold) & (high_dim_labels == -1))] = i
            del rmsd
            
        return center_index, high_dim_labels
    
    def cluster_population(self):
        """
        Count the number of instances in each cluster, and calculates the instance number 
        in percentage. We expect a meaningful cluster to contain at least 5%
        of the total data points.

        Returns
        -------
        pop_df : TYPE Pandas DataFrame
            DESCRIPTION. A DataFrame that contains the index of cluster, number
            of instances, and percentage of instances.

        """
        cluster_idx, population = np.unique(self.highD_labels_arr, return_counts=True)
        percentage = np.array(population/self.num_instances*100, dtype=int)
        pop_dict = dict(cluster_idx=cluster_idx, population=population,
                        percentage=percentage)
        
        pop_df = pd.DataFrame(pop_dict)
        return pop_df
    
    def sorted_idx(self):
        """
        Group the index of triad molecule based on their labels

        Returns
        -------
        sorted_lst : TYPE List (of numpy arrays), len=num_of_clusters
            DESCRIPTION. The index of instances that belongs to the same cluster are sorted 
            in the same numpy array. Indices with label -1 is not included

        """
        sorted_lst = []
        for i in np.unique(self.highD_labels_arr):
            idx = np.where(self.highD_labels_arr==i)
            sorted_lst.append(idx)
        del sorted_lst[0]
        
        return sorted_lst
    
    def similarity_score(self, absolute=False):
        """
        We want to compare the labels generated by high-dimensional space clustering
        and feature space clustering. sklearn.metrics.accuracy_score is
        implemented in this function to inspect the fraction of instance labels 
        that are the same by two clustering methods
        
        Parameters
        ----------
        absolute : TYPE Bool, optional
            DESCRIPTION. If True, all the instances are considered. If False,
            will ignore the instances that has -1 label in high-dimensional clustering.
            The default is False.

        Returns
        -------
        score : TYPE Float
            DESCRIPTION. The fraction of instance that has the same label generated 
            by the two methods. The number ranges from 0 to 1.
            
        """
        if absolute==True:
            highD_labels = self.highD_labels_arr
            FeatureSpace_labels = self.label_array
            
        elif absolute==False:
            highD_labels = self.highD_labels_arr[np.where(self.highD_labels_arr\
                                                          != (-1))]
            FeatureSpace_labels = self.label_array[np.where(self.highD_labels_arr\
                                                          != (-1))]
        score = accuracy_score(highD_labels, FeatureSpace_labels)
        
        return score
          
        