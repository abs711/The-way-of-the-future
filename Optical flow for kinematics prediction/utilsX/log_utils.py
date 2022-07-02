# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:52:15 2020

@author: vijet
"""
import torch
 
import datamanager_v7 as dm
 
import os
import copy

import numpy as np

from collections import defaultdict
from collections import OrderedDict

import sys

from pytorchtools import EarlyStopping


logger={"y_preds":[],"y_tests":[],"x_frames":[],"trial_info":[],"test_Loss":[], 'val_Loss':[], "Epoch_vs_TrainLoss": [], "Epoch_vs_ValLoss": [],"test_full_samples":[], "pred_full_samples":[], "Train_Report": [], "Test_Report": [], 'Train_Confusion': [], 'Test_Confusion': [], 'LR_Schedule': []}    


def append_to_log(temp_dict):
   #''' Desc:  appends to loG_dictionary the items in temp dict. Log MUST have
   #          same and all keys
   #     Arguments:  
   #     returns:    appended log dictioneay
   #     @todo :           
   #'''
 
     for key in temp_dict.keys():         
          #print ("Key",log_dict[key])
          logger[key].append(temp_dict[key])                       
     return logger

def getLogger():

    global logger 
    
    return logger


def scriptable_params(params):
    
    for key, value in params.items():
          
           if type(params[key])!=list:
#              print (key, value)               
              try:
                  params[key]=list(value)
              except :
                  params[key]=[value]
              
    
    return params

def get_auto_anno(param_vals):
    
   return 'seq'+str(param_vals['seq_length'][0]) + '_tar'+str(param_vals['target_len'][0])+'_pred'+str(param_vals['pred_horizon'][0])
    
    
     
