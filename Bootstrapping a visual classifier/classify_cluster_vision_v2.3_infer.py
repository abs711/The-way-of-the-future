# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:33:20 2020

@author: vijet
"""


import os, sys
from pathlib import Path, PurePath

from fastai.vision import *
import pandas as pd
import numpy as np
import time
from PIL import Image as pil_img
from datetime import datetime
from scipy.spatial.distance import cosine

timestamp= datetime.now().strftime("%m_%d_%H_%M")
from sklearn.manifold import TSNE
import plotly_express as px


from sklearn.cluster import KMeans 
  
import matplotlib.pyplot as plt
from util_funcs import *
data_dir = '../../../data/vision_pretraining/sampled_classified'  
tmfs = get_transforms()

import pickle

# obj0, obj1, obj2 are created here...



#%%
         
if __name__ == "__main__":
    
    

    bImageRepsNow=True
    

    batch_size=50
    anno='all_train'
    expt_dir=set_expt_dir('05_17_09_15')


    if anno=='all_train':
        train_data_csvfile=  ['train_all_data.csv']
        saved_model='resnet18-224-unfreezed_all_train'
 
    else :
        train_data_csvfile=  ['train_steady_data.csv']
        saved_model='resnet18-224-unfreezed_steady_train'

     
    train_df,train_databunch,train_databunch_unsplit=get_train_databunch_from_csv(train_data_csvfile,batch_size)

    val_df,val_databunch=get_unsplit_databunch_from_csv(['val_all_data.csv'],batch_size)
    inf_df,inf_databunch=get_unsplit_databunch_from_csv(['test_all_data.csv'],batch_size)

    
    learner_train=get_pretrained_model_db(train_databunch_unsplit,saved_model)   
    
    model_train = learner_train.model   
    linear_output_layer_train = get_named_module_from_model(model_train, '1.4')
    
    
     
    
    #read classified 
#    train_df=pickle_read_dataframe(expt_dir,'classified_train.pkl')
    
    # read img reps
    
    train_df=pickle_read_dataframe(expt_dir,'img_reps_tsne_train.pkl')
    train_img_repr_matrix=read_img_rep_matrix_from_df(train_df)
    
    val_df=pickle_read_dataframe(expt_dir,'img_reps_tsne_val.pkl')
    val_img_repr_matrix=read_img_rep_matrix_from_df(val_df)

    inf_df=pickle_read_dataframe(expt_dir,'img_reps_tsne_inf.pkl')
    inf_img_repr_matrix=read_img_rep_matrix_from_df(inf_df)
    
    
    



    """ ######Inference #####################"""
    
    # infer based on model fit on train set
    
    #trained train, test inf
    cond_str='_inter_sub'
    inf_df =predict_clusters_all_models(expt_dir,inf_df,train_img_repr_matrix,cond_str)
    
    ## trained val, test inf   
    cond_str='_intra_sub'
    inf_df =predict_clusters_all_models(expt_dir,inf_df,val_img_repr_matrix,cond_str)
    
    ## save all
    
    inf_df=save_results_df(expt_dir,inf_df,save_as='results_inf_df')  
#    inf_intra_df=save_results_df(expt_dir,inf_intra_df,save_as='results_intra_df')  


   
# 