# -*- coding: utf-8 -*-
"""
Created on Mon May 25 09:15:23 2020

@author: vijet
"""
from datetime import datetime
from numpy import inf

from dtaidistance import dtw,clustering
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
from sklearn.metrics.pairwise import euclidean_distances
from scipy import spatial
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics.pairwise import euclidean_distances

from scipy.spatial import distance

import pickle 
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from pathlib import Path, PurePath
import os
import scipy.io
import itertools
import pandas as pd
expt_dir=''
timestamp= datetime.now().strftime("%m_%d_%H_%M")

data_dir = '../../../data/vision_pretraining/sampled_classified'  


def create_expt(base_dir,anno='_test'):
    
    global expt_dir
    expt_dir = os.path.join(base_dir, str(timestamp) + anno) 

    os.mkdir( expt_dir );
    
    return expt_dir



def set_expt_dir(base_dir,rel_expt_dir):
    global expt_dir
    
    expt_dir = os.path.join(base_dir, rel_expt_dir) 

    return expt_dir


def pickle_this_dict(expt_dir,dict_to_pickle,save_as):
    
    with open(Path(expt_dir,save_as+'.pkl'), 'wb') as f:  
        pickle.dump(dict_to_pickle, f)

def mat_this_dict(expt_dir,dict_to_mat,save_as):
    

    scipy.io.savemat(Path(expt_dir,save_as).with_suffix('.mat'),dict_to_mat)
    
def dict_this_mat(expt_dir,save_as):
    
    mat_dict = scipy.io.loadmat(Path(expt_dir,save_as).with_suffix('.mat'))
    
    return mat_dict
        
def dict_this_pickle(expt_dir,read_as):

    with open(Path(expt_dir,read_as+'.pkl'), 'rb') as f:  
        pickled_dict= pickle.load(f)
        
    return pickled_dict

def  pickle_save_dataframe(expt_dir,data_df,save_as):
    
    data_df.to_pickle(Path(expt_dir,save_as).with_suffix('.pkl'))
        
    return 
    

def pickle_read_dataframe(expt_dir,saved_pickle) :
    
    pickle_file=Path(expt_dir,saved_pickle).with_suffix('.pkl')
    print ("Reading pickle file:",pickle_file)
    data_df=pd.read_pickle(pickle_file)
    
    return data_df

"""" loading datat """""

def get_data_from_dir(sub_dirs,file_name='normalizedT_gait_cycles_data.mat',vars_to_load=['frames_cycles', 'gait_inits', 'input_joints_cycles', 'knee_cycles', 'knee_cycles_mat', 'matlab_dtw_dist', 'matlab_intrasub_clusters', 'matlab_medoids']):
    
    all_data_dict={}
    
    for this_var_name in vars_to_load: 
        
        all_data_var=[]
        for sub_dir in sub_dirs:
            full_sub_dir=Path(data_dir,sub_dir)
            full_filename=Path(full_sub_dir,file_name)
            trial_id=str(full_sub_dir.relative_to(data_dir))

            mat_dict = scipy.io.loadmat(full_filename)
            this_trial_var_data=mat_dict[this_var_name]
            all_data_var.append(this_trial_var_data)
            

        # concatenate all trials into one np array
        all_data_var=np.array(list(itertools.chain.from_iterable(all_data_var)) ) 

        all_data_dict[this_var_name]=all_data_var  
        
    # repeat for generating all trial ids repeated for every cycle in trial    
    all_trialids=[]    
    for sub_dir in sub_dirs:
            full_sub_dir=Path(data_dir,sub_dir)
            full_filename=Path(full_sub_dir,file_name)
            trial_id=str(full_sub_dir.relative_to(data_dir))

            mat_dict = scipy.io.loadmat(full_filename)
            this_trial_num_cycles=mat_dict['knee_cycles_mat'].shape[0]
            this_trial_ids=[trial_id]*this_trial_num_cycles

            all_trialids.append(this_trial_ids)
    all_trialids=list(itertools.chain.from_iterable(all_trialids)) 
    all_data_dict['trial_id']=all_trialids  

    return all_data_dict   

def get_all_trials_in(subjects):
    
    train_dirs=[]
    
    for sub_dir in subjects:
        full_sub_dir=Path(data_dir,sub_dir,'unstructured')
        for x in full_sub_dir.iterdir():
            trial_dir=x.relative_to(data_dir)

            train_dirs.append(trial_dir)
     
    return train_dirs

        
def dtw_dist(x_mat):

   ds = dtw.distance_matrix_fast(x_mat,compact=False)

   ds[ds==inf]=0

   ds_full=ds.T+ds
   
   return ds_full


def get_cluster_centroids_wcss_dict(data,data_clusters_labels):
    
#    data_clusters_labels_list= list(data_clusters_labels)
#    indices = [i for i, x in enumerate(my_list) if x == "whatever"]

    
    clusters_dict={} # dict of dicts
    
    
    for lbl in range(data_clusters_labels.min(), data_clusters_labels.max()+1):
        this_lbl_dict={}
        

        
        this_lbl_indices=np.where(data_clusters_labels == lbl)
        this_lbl_data=data[this_lbl_indices]
#        codebook.append(this_lbl_data.mean(0))
        
        this_lbl_dict['abs_indices']=np.asarray(this_lbl_indices).reshape(-1,)
        this_lbl_dict['mean']=this_lbl_data.mean(0)
        this_lbl_dict['data']=data[this_lbl_indices]
        
        clusters_dict[lbl]=this_lbl_dict
        
#    cluster_means= np.vstack(codebook)
    
    
    clf = NearestCentroid(metric='euclidean')
    clf.fit(data, data_clusters_labels)
    cluster_centroids=clf.centroids_
    
    
    total_sse=0
    
    for key in clusters_dict: #0,1,2
#        print ("key",key)
        this_cluster_sse=0
        
        this_lbl_dict=clusters_dict[key]
        this_lbl_dict['centroid']=cluster_centroids[key]# 0 index
        this_cluster_members_idx=this_lbl_dict['abs_indices']
        
        for idx in this_cluster_members_idx:

            this_cluster_sse=this_cluster_sse+distance.euclidean(this_lbl_dict['centroid'],data[idx])
        
        this_lbl_dict['WCSS']=this_cluster_sse
        total_sse= total_sse+ this_cluster_sse  
        

    return clusters_dict,cluster_centroids,total_sse

def get_r_ratio(cluster_analysis_dict):
    
    k_list=[]
    for key in cluster_analysis_dict: # get all cluster
        k_list.append(key)

    for k_idx,K in enumerate(k_list): # all except last cluster    
        
        if k_idx == len(k_list) -1: break
        N=cluster_analysis_dict[K]['cluster_labels'].shape[0]
        this_k_total_sse=cluster_analysis_dict[K]['total_wcss']
        kplusone_total_sse=cluster_analysis_dict[K+1]['total_wcss']
        
        cluster_analysis_dict[K]['R_ratio']=(this_k_total_sse/kplusone_total_sse -1)*(N-K+1)
        
    return cluster_analysis_dict  


"""" plotting """""

def plot_save_centroids(expt_dr,this_centroids):
    
    num_centroids=this_centroids.shape[0]
    
    print("Num of centroids",num_centroids)
    
    fig,ax =  plt.subplots()
#    plt.figure(1)
    for i in range(num_centroids):
        
#        a[i][0].plot(this_centroids[i,:])
#        a[i][0].set_title('Centroid:'+str(i))
#        plt.show()


#        plt.subplot(num_centroids,1,i+1)
#        plt.plot(this_centroids[i,:])
        
        
        ax.plot(this_centroids[i,:], '-b', label='cluster_'+str(i))
        
    ax.legend(frameon=False, loc='lower center', ncol=num_centroids)
#    plt.show()     


    
    
    
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)',fontsize=16, fontweight='bold')
        plt.xlabel('Cluster Size)',fontsize=13, fontweight='bold')
        plt.ylabel('Linkage Distance',fontsize=13, fontweight='bold')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata




def plotly_tsne_df(img_repr_df,clr_labels):
    
    fig=px.scatter_3d(img_repr_df, x='TSNE Dim 1', y='TSNE Dim 2', z='TSNE Dim 3',color=clr_labels,
                     width=1600, height=800) 
    fig.update_layout(
    margin=dict(l=5, r=5, t=5, b=40),
    )
    fig.show()
    
    return fig

###
    
def xchange_SA_SD_labels(labels_SA1):
    
    labels_SA2=np.copy(labels_SA1)
    
    labels_SA2[labels_SA1==1]=2
    labels_SA2[labels_SA1==2]=1
    
    return labels_SA2


def xchange_other_labels(labels_other1):
    
    labels_other3=np.copy(labels_other1)
    
#    labels_other3[labels_other1==2]=1
    labels_other3[labels_other1==3]=1
    labels_other3[labels_other1==1]=3

    
    return labels_other3

def xchange_specific_labels(labels_other1,label1,label2):
    
    labels_other3=np.copy(labels_other1)
    
#    labels_other3[labels_other1==2]=1
    labels_other3[labels_other1==label2]=label1
    labels_other3[labels_other1==label1]=label2

    
    return labels_other3

def xchange_specific_labels(labels_other1,label1,label2):
    
    labels_other3=np.copy(labels_other1)
    
#    labels_other3[labels_other1==2]=1
    labels_other3[labels_other1==label2]=label1
    labels_other3[labels_other1==label1]=label2

    
    return labels_other3