#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:40:08 2019

@author: raiv
"""

import torch
import torch.nn as nn
import datamanager_v7 as dm

from datetime import datetime
import os

from torch_utils import *
from log_utils import *
from train_eval_functions import evaluate
from loss_modules import RMSELoss

from KinematicsDataset import *

import numpy as np
from itertools import product
from collections import defaultdict
from collections import OrderedDict
from math import sqrt 

timestamp= datetime.now().strftime("%m_%d_%H_%M")


    

if __name__ == "__main__":
    
    
    
    data_dir = "D:/GDrive_UW/DeepLearning/data/xSens_Phase1/" 
    #data_dir= 'C:/Users/vijet/Documents/Google Drive/DeepLearning/data/xSens_Phase1/'

    config_file_name = "ConfigFiles/SeqToSeq/v6_cf_Obs_(Ankle)-test.csv"
    
    
#    
#    expt_path= "D:/GDrive_UW/DeepLearning/data/xSens_Phase1/results/SeqToOne/obstacle/baseline_10trials_02_03_15_01/model"
#    model_path=expt_path +'/param_1_trial_2.pt'
#    model_type='SeqToOne'
    
    
    expt_path= "D:/GDrive_UW/DeepLearning/code/torchKinematics_LSTM/SeqtoSeq_tuts/TCN-master/TCN/adding_problem/saved"
    model_path=expt_path +'/model_conv.pt'
    model_type='TCN_Ankle'
    
    
    results_path= expt_path
    
    dm.set_results_dir_manual(results_path)
    
    
    print ("model path",model_path)
    param_vals = {'learning_rate': (1e-3), 'num_epochs': [500], 'batch_size':(100),'num_trials':[10],
                   'seq_length':[30],'target_len':[1], 'window_step':[3],'sub_sample_factor':[2], 'frame_subsample_factor':[0],
                   'num_layers':(2),'num_hid_units':(30), 'kernel_size':(7),'tcn_levels':(6),
                   'reg_parameter':[0.0],'noise_std':[0.02],'clip':[-1],'dropout':(0.0),
                   'num_features':(60)
                   }
    
    param_vals=OrderedDict(param_vals)
     
    
    print(param_vals.keys())
    
    train_dict, test_dict, val_dict = dm.get_and_reformat_all_data(data_dir, config_file_name,param_vals)

    
    print ('test shape',test_dict['y'].shape)
    #assert shape
    
#    num_features_data=train_dict['x'].shape[2]
#    assert(param_vals['num_features'] == num_features_data),"num features mismatch in data"
     
    

    model=getModule(model_type,param_vals) 
    criterion = RMSELoss()

    dataloaders=getDataloader(train_dict,test_dict,val_dict,model_type,param_vals)


    
    log_predictions=getLogger() 
    
    thisTrial_model_dict = torch.load(model_path)
    model.load_state_dict(thisTrial_model_dict['model_state_dict'])
       #optimiser.load_state_dict(bestModel_dict['optimiser_state_dict'])
    
    yTests,yPreds=evaluate(model,model_type,dataloaders['test'])
    
    this_trial_rmse=criterion(yPreds, yTests).item()
    
    print ("RMSE",this_trial_rmse)
       
    log_predictions['y_preds'].append(yPreds.cpu().detach().numpy());
    log_predictions['y_tests'].append(yTests.cpu().detach().numpy())
    log_predictions['trial_info'].append(param_vals)
    log_predictions['test_RMSE'].append(this_trial_rmse)
       #log_predictions['Epoch vs Loss'].append(hist)
         
    dm.save_as_mat([model_type + '/preds','predictions'],log_predictions) 
    #dm.save_as_mat(['hyperparams','predictions_1testtrial'],log_predictions) 
   
       