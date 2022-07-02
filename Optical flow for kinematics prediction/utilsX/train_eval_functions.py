# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:59:15 2020

@author: vijet
"""

import torch

import datamanager_v7 as dm

import os
import copy
from Lstm_Modules import *
from SeqToSeqModels import *
import numpy as np

import time

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
import device_utils
device = device_utils.getCudaDevice()
USE_CUDA = torch.cuda.is_available()

model_type_dicts={'history_models': ['HistorySeqToOne','HistorySeqToTwo','HistorySeqToSeq_Ankle','NARX_Ankle'],
                  'no_history_models':['SeqToTwo','TCN_Ankle'] ,
                  'has_inbuilt_iter':['SeqToOne','UserStatSeqToOne']
                  }

print ("device for training",device)
def gaussian(ins, is_training, mean, stddev):
    if is_training:
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        return ins + noise
    return ins


""" iters for models
"""    
def history_iters(batch,model,phase):
     
     seq_len=model.seq_len 
     target_len=model.target_len
     with torch.no_grad():
        batchX,batchUserStat,batchYSeq=batch

#        batchX=batchX[:,0:-1,:]
        batchY_history=batchYSeq[:,0:seq_len-1,:]; batchY=batchYSeq[:,-target_len:,:]
        if USE_CUDA:
                              
              batchX=batchX.to(device)
              batchY=batchY.to(device)
              batchY_history=batchY_history.to(device)
                       
        y_preds=model(batchX,batchY_history,batchY, phase)
        return batchY,y_preds

def no_history_iters(batch,model,phase):
    
    with torch.no_grad():
        batchX,batchUserStat,batchY=batch

        if USE_CUDA:
                              
              batchX=batchX.to(device)
              batchY=batchY.to(device)
            
        y_preds=model(batchX)
        return batchY,y_preds



""" general evaluate and train functions for all classes of models

"""


def kinematics_evaluate(model,phase,model_type,dataloader)  : 
    
    yTests_temp = []; yPred_temp = []
    model.set_phase(phase)   
        
    for batch_idx,batch in enumerate(dataloader):   
       torch.cuda.empty_cache() 
       tic_batch=time.clock() 
      
       if model_type in  model_type_dicts['history_models']:
               
           batchY,batchY_pred= history_iters(batch,model,phase)
           
       elif  model_type in  model_type_dicts['no_history_models']:

             batchY,batchY_pred= no_history_iters(batch,model,phase)
             
       elif  model_type in  model_type_dicts['has_inbuilt_iter']  :
           
             batchY,batchY_pred = model.execute_model(batch,phase)
       
       yTests_temp.append(batchY); yPred_temp.append(batchY_pred)
        
       toc_batch=time.clock()
       
    yPreds =torch.cat(yPred_temp)
    yTests = torch.cat(yTests_temp)
    
    return yTests,yPreds
   

def kinematics_train(model,phase,model_type,dataloader)  : 
    
    yTests_temp = []; yPred_temp = []
    model.set_phase(phase)   
        
    for batch_idx,batch in enumerate(dataloader):   
       torch.cuda.empty_cache() 
       tic_batch=time.clock() 
       
       if model_type=='NARX_Ankle' or model_type=='NoHistoryNARX_Ankle':  
           batchY,batchY_pred= NARX_train_iters(batch,model,phase)
           
       elif model_type=='TCN_Ankle':    
           batchY,batchY_pred= TCN_train_iters(batch,model,phase)
       
       elif  model_type=='HistorySeqToOne':  
           batchY,batchY_pred= history_iters(batch,model,phase)
       
       elif model_type=='SeqToOne':
             batchY,batchY_pred= TCN_train_iters(batch,model,phase)
             
             
       yTests_temp.append(batchY); yPred_temp.append(batchY_pred)
        
       toc_batch=time.clock()
       
    yPreds =torch.cat(yPred_temp)
    yTests = torch.cat(yTests_temp)
    
    return yTests,yPreds
 

 
def evaluate(model,phase,model_type,dataloader) :
    
#       if model_type =='SeqToOne' or model_type=='HistorySeqToOne' or model_type =='TCN_Ankle' or model_type =='NARX_Ankle' or  model_type=='NoHistoryNARX_Ankle':
        print ("Model type", model_type )
        yTests,yPreds = kinematics_evaluate(model,phase,model_type,dataloader)

        return yTests,yPreds  
    


#####
#        
#        def vision_eval_iters( batch_idx, batch,model):
#    
#    model.eval()
#    
#    with torch.no_grad():
#       batchX,batchUserStat,batchFrames,batchY=batch 
#       if USE_CUDA:
#
#          batchX=batchX.to(device)
#          batchY=batchY.to(device)
#          batchUserStat=batchUserStat.to(device)
#          batchFrames=batchFrames.to(device)
#       
#       probs_outs,preds=model(batchX,batchUserStat,batchFrames)
#       torch.cuda.empty_cache() 
#       
#       return  probs_outs.detach().cpu(),preds.detach().cpu(),batchFrames.detach().cpu() 
#   
#    
#def vision_evaluate()  : 
#    
#    all_probs=[];all_frames=[];all_preds=[];
#                   
#    for batch_idx,batch in enumerate(dataloaders['test']):   
#       torch.cuda.empty_cache() 
#
#       tic_batch=time.clock() 
#       if debugPrint : print ("Pre iters allocated and cached ", batch_idx,torch.cuda.memory_allocated()/1e+9,torch.cuda.memory_cached()/1e+9)
#
#       probs_outs,preds,batchFrames= eval_iters(batch_idx,batch,model)
#       
#       toc_batch=time.clock()
#       
#       #print ("shapes",probs_outs.shape,batchFrames.shape,preds.shape)
#       all_probs.append(probs_outs);
#       all_frames.append(batchFrames)
#       all_preds.append(preds)
##                           toc_batch=time.clock()
#       if batch_idx == 300: break  
