# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:55:20 2020

@author: vijet

use best cluster models to cluster datasets
"""

import sys
sys.path.append("../utils")

from datetime import datetime
from numpy import inf
from time import sleep
from dtaidistance import dtw,clustering
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
from sklearn.metrics.pairwise import euclidean_distances

from scipy.spatial import distance
from sklearn.neighbors.nearest_centroid import NearestCentroid

import pickle 
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from pathlib import Path, PurePath
import os
import scipy.io
import itertools

from  util_funcs_kinematic_cluster import *

from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from jqmcvi import base 
from sklearn.metrics import silhouette_score as Sil
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.cluster import adjusted_rand_score


data_dir = '../../../data/vision_pretraining/sampled_classified'  
expt_dir=''
timestamp= datetime.now().strftime("%m_%d_%H_%M")


 
#%%        
if __name__ == "__main__":  
    
    k=3

    
    expt_dir=create_expt('results/','dataset_k'+str(k))
#    expt_dir=set_expt_dir('results/','05_30_21_55HAC_K_medoids')
    
    vars_to_load=['frames_cycles', 'gait_inits', 'input_joints_cycles', 'knee_cycles', 'knee_cycles_mat', 'matlab_dtw_dist', 'matlab_intrasub_clusters', 'matlab_medoids']
    vars_to_load_test=['frames_cycles', 'gait_inits', 'input_joints_cycles', 'knee_cycles', 'knee_cycles_mat', 'matlab_dtw_dist', 'matlab_intrasub_clusters', 'matlab_medoids','gait_cycle_labels_GT','gait_cycle_labels_GT_fine']
    
    
    subs=['xUD002','xUD004','xUD006','xUD007','xUD008','xUD009','xUD011','xUD012']
#    subs=['xUD002']

    train_dirs = get_all_trials_in(subs)
    test_dir=['xUD015/unstructured/003']
    val_dir=['xUD015/unstructured/004']
    

    train_data_dict=get_data_from_dir(train_dirs,vars_to_load=vars_to_load)
    test_data_dict=get_data_from_dir(test_dir,vars_to_load=vars_to_load_test)
    val_data_dict=get_data_from_dir(val_dir,vars_to_load=vars_to_load)

      

    dataset_dict={}
    dataset_dict['train']=train_data_dict
    dataset_dict['val']=val_data_dict
    dataset_dict['test']=test_data_dict
    
#%% HAC 
       
    evaluate_key_list=['train']
    for key in   evaluate_key_list:
        
        print ("Evaluating Key", key)
        
        data_to_cluster=dataset_dict[key]['knee_cycles_mat']
        Z_ward = linkage(data_to_cluster, 'ward',metric='euclidean')
    
        
        print ("\n Evaluating K=",k)
    
        HAC_labels=fcluster(Z_ward, k, criterion='maxclust')
        print ("HAC min label", min(HAC_labels))
    
        if min(HAC_labels) ==1:
            HAC_labels=HAC_labels-1 # zero as label
            
        #if needed for  K=3 exchange SA to be 1 and SD=2    
            
#        HAC_labels = xchange_SA_SD_labels(HAC_labels)
    
        plt.plot(HAC_labels)
    
            
        this_k_centroid_dict,centroids,total_WCSS=get_cluster_centroids_wcss_dict(data_to_cluster,HAC_labels)

          
        dataset_dict[key]['HAC_labels'+'K'+str(k)]=HAC_labels
        dataset_dict[key]['HAC_centroids'+'K'+str(k)]=centroids

  
        mat_this_dict(expt_dir,dataset_dict[key],key+'_K'+str(k)+'_dict')
        pickle_this_dict(expt_dir,dataset_dict[key],key+'_K'+str(k)+'_dict')


    
    
        
 
        
