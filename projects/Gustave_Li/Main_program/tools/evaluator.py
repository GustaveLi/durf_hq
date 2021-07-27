# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 12:38:35 2021

@author: gqmli
"""

import numpy as np
from sklearn.metrics import euclidean_distances

class Evaluators():
# Code reference from: https://github.com/mynameisfiber/pyxmeans/blob/c7d0edbc1a4e983c043a340f2dba9e911049c0b2/pyxmeans/xmeans.py
# BIC & AIC mechanism reference from: http://www.cs.cmu.edu/~dpelleg/download/xmeans.pdf
    def __init__(self, original_data, labels, ccenters):
        # original_data: dataset used for clustering
        self.original_data = original_data
        self.labels = labels
        self.ccenters = ccenters
        self.cluster_num = labels.max()+1 #Label index starts from 0
        self.total_num = len(labels)
        self.dims = original_data[0].size
        
    def get_sorted_data(self):
        # Sort the datapoints in original_data by cluster num
        
        # Initialize a dictionary to store datapoints from different cluster num
        sorted_dict = {}
        for num in range(self.cluster_num):
            sorted_dict[num]=np.ones((1,2))
        
        # Add datapoint to different arrays according to their label
        for i in range(self.total_num):
            new = np.append(sorted_dict[self.labels[i]], [self.original_data[i]], 
                            axis=0)
            sorted_dict[self.labels[i]] = new
            del new
            
        # Clean up dictionary (remove initialization)
        for num in range(self.cluster_num): 
            a = np.delete(sorted_dict[num], 0, axis=0)
            sorted_dict[num]=a
            del a
            
        self.sorted_data = sorted_dict
    
    def cluster_varience(self, cluster, centroid):
        distances = euclidean_distances(cluster, [centroid])
        v = (distances**2).sum()
        denom = float(self.total_num - self.cluster_num)
        return v / denom
    
    def loglikelihood(self):
        lllh=0
        for cluster, centroid in zip(self.sorted_data.values(),self.ccenters):
            varience = Evaluators.cluster_varience(self, cluster, centroid)
            t1 = len(cluster)*np.log(len(cluster))
            t2 = len(cluster)*np.log(self.total_num)
            t3 = len(cluster)*np.log(2*np.pi)/2
            t4 = len(cluster)*self.dims*np.log(varience)/2
            t5 = (len(cluster)-self.cluster_num)/2
            lllh += t1 - t2 - t3 - t4 - t5
        return lllh
    
    def free_params(self):
        return self.cluster_num * (self.dims+1)
    
    def bic(self):
        lllh = Evaluators.loglikelihood(self)
        num_params = Evaluators.free_params(self)
        bic = num_params*np.log(self.total_num) - 2*lllh
        return bic    
    
    def aic(self):
        lllh = Evaluators.loglikelihood(self)
        num_params = Evaluators.free_params(self)
        aic = 2*num_params - 2*lllh
        return aic