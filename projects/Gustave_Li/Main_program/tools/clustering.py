from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score


class Preclustering():
    def __init__(self, dataset, num_instances=100000):
        self.dataset = dataset[:num_instances, :]
        
    def kmeans(self, n_clusters):
        estimator = KMeans(n_clusters=n_clusters)
        estimator.fit_predict(self.dataset)
        ccenters = estimator.cluster_centers_
        inertia = estimator.inertia_
        sil_score = silhouette_score(self.dataset, estimator.labels_)
        return [inertia, sil_score], ccenters

    def kmedoids(self, n_clusters):
        estimator = KMedoids(n_clusters=n_clusters, init='k-medoids++')
        estimator.fit_predict(self.dataset)
        ccenters = estimator.cluster_centers_
        inertia = estimator.inertia_
        sil_score = silhouette_score(self.dataset, estimator.labels_)
        return [inertia, sil_score], ccenters
    
    def gmm(self, n_components):
        estimator = GaussianMixture(n_components=n_components, n_init=10)
        labels = estimator.fit_predict(self.dataset)
        ccenters = estimator.means_
        aic = estimator.aic(self.dataset)
        bic = estimator.bic(self.dataset)
        sil_score = silhouette_score(self.dataset, labels)
        return [aic, bic, sil_score], ccenters
    
    def hdbscan(self, min_clus_size, min_sam, clus_sel_eps):
        estimator = HDBSCAN(min_cluster_size=min_clus_size,
                            min_samples = min_sam,
                            cluster_selection_epsilon = clus_sel_eps)
        estimator.fit_predict(self.dataset)
        sil_score = silhouette_score(self.dataset, estimator.labels_)
        return [sil_score]
    
    