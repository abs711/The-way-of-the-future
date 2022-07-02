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
from util_funcs_v1 import *
frames_data_dir = '../../../data/vision_pretraining/processed'  
tmfs = get_transforms()


#%%
         
if __name__ == "__main__":
    
    
    bTrainNow= True
    bImageRepsNow=True
    
    batch_size=50
    anno='_all_train'
    expt_dir=create_expt(anno)
    
    if anno=='_all_train':
        train_data_csvfile=  ['train_all_data.csv']
        saved_model='resnet18-224-unfreezed_all_train'
 
    else :
        train_data_csvfile=  ['train_steady_data.csv']
        saved_model='resnet18-224-unfreezed_steady_train'
        
    train_df,train_databunch,train_databunch_unsplit=get_train_databunch_from_csv(train_data_csvfile,batch_size)

    val_df,val_databunch=get_unsplit_databunch_from_csv(['val_all_data.csv'],batch_size)
    inf_df,inf_databunch=get_unsplit_databunch_from_csv(['test_all_data.csv'],batch_size)

#    
    
    #%%
    if bTrainNow:
      # train new model 
        train_model(train_databunch,n_iters=50,lr=1e-06,save_anno=anno)   
        
  
    # load it with unsplit train data   
    learner_train=get_pretrained_model_db(train_databunch_unsplit,saved_model)   

    model_train = learner_train.model   
    linear_output_layer_train = get_named_module_from_model(model_train, '1.4')
    
    
    ######## Generate labels, img reps and save
    # train, val and test
    
    """ ###### Train #####"""
        # classifiy and save train set
    print ("Train Set Data Generation")    
    train_df=classify_add_df(learner_train,train_databunch_unsplit,train_df)  
    pickle_save_dataframe(expt_dir,train_df,'classified_train.pkl')
    
 #%%
    # get train img reps and tsne and save
    train_df,train_img_repr_matrix=get_img_reps_mat_add_df(bImageRepsNow,expt_dir,train_df,train_databunch_unsplit,linear_output_layer_train,model_train)
    train_df= get_tsne_data_add_df(train_df,train_img_repr_matrix)    
    pickle_save_dataframe(expt_dir,train_df,'img_reps_tsne_train.pkl')


    # save  combined results train
    train_df=save_results_df(expt_dir,train_df,save_as='results_train') 
    
    
    """###### Validation set ######## """
    
    print ("Val Set Data Generation")    

        #    Classify Val set
    val_df=classify_add_df(learner_train,val_databunch,val_df)       
    pickle_save_dataframe(expt_dir,val_df,'classified_val.pkl')

    
    # get inf img reps and tnse save
    val_df,val_img_repr_matrix=get_img_reps_mat_add_df(bImageRepsNow,expt_dir,val_df,val_databunch,linear_output_layer_train,model_train)       
    val_df= get_tsne_data_add_df(val_df,val_img_repr_matrix)   
    pickle_save_dataframe(expt_dir,val_df,'img_reps_tsne_val.pkl')

    # save val combined results
    val_df=save_results_df(expt_dir,val_df,save_as='results_val') 
#    


    """ ######Inference #####################"""
    
    print ("Test Set Data Generation")    

    #    classify inf 
    inf_df=classify_add_df(learner_train,inf_databunch,inf_df)       
    pickle_save_dataframe(expt_dir,inf_df,'classified_inf.pkl')

    
    # get inf img reps and tnse
    inf_df,inf_img_repr_matrix=get_img_reps_mat_add_df(bImageRepsNow,expt_dir,inf_df,inf_databunch,linear_output_layer_train,model_train)       
    inf_df= get_tsne_data_add_df(inf_df,inf_img_repr_matrix)   
    pickle_save_dataframe(expt_dir,inf_df,'img_reps_tsne_inf.pkl')


    inf_df=save_results_df(expt_dir,inf_df,save_as=anno+'results_inf')  
    
   


# 