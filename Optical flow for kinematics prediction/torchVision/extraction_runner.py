# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 11:50:48 2020

@author: vijet
"""

import torch
import sys
sys.path.append("../../utilsX")
 
import datamanager_v7 as dm
 
import os
import copy
import numpy as np
from itertools import product
from collections import defaultdict
from collections import OrderedDict
import time
import torch.utils.data as utils 
from torch.autograd import Variable
import sys
from pytorchtools import EarlyStopping

from torch_utils import *
from log_utils import *
from train_eval_functions import evaluate
from loss_modules import * #RMSELoss
from metrics import *
from random import randrange
debugPrint=0


import device_utils
device = device_utils.getCudaDevice() 
#sanity_check=True
#single_batch= False
#clipGrads=False
#indie_ckpt = False
substring_list = ['Conv','Vision','Scene','Optical','Spatiotemporal']
##def SetTrainingOpts(sanityflag,use_single_batch,overlap_datatransfers,model_name,clipGradients=False,indie_checkpoints=False,vision_substrings =['Conv','Vision','Scene','Optical','Spatiotemporal'],numscenesamples=1):
##    global sanity_check,single_batch,non_blocking,unnecessary_model_name,clipGrads,indie_ckpt,substring_list,num_scenesamples
##    sanity_check,single_batch,non_blocking,unnecessary_model_name,clipGrads,indie_ckpt,substring_list,num_scenesamples= sanityflag,use_single_batch,overlap_datatransfers,model_name,clipGradients,indie_checkpoints,vision_substrings,numscenesamples
##    return 
##
##def GetTrainingOpts():
##    global sanity_check,substring_list,num_scenesamples,unnecessary_model_name
##    return sanity_check,substring_list,num_scenesamples,unnecessary_model_name
##
##
#sanityflag,single_batch,clipGrads,indie_ckpt,substring_list = SetTrainingOpts()
##validation_patience = 1
# we will use CUDA if it is available
USE_CUDA = torch.cuda.is_available()
#os.environ['CUDA_VISIBLE_DEVICES'] = device_utils.getCudaDeviceidx()



def get_batches4models(batch,model_type):

       if any(substring in model_type for substring in substring_list):
               batchX,batchUserStat,batchFrames,batchYSeq,frame_Names=batch
               return batchX,batchUserStat,batchFrames,batchYSeq,frame_Names
       else:
               batchX,batchUserStat,batchYSeq=batch
               return batchX,batchUserStat,None,batchYSeq


def load_frames4models(batchFrames,model_type):
       
          if 'Scene' in model_type:
                  #frame_id = 4#randrange(seq_len)
                          #print('Frame selected: ', frame_id)
                  #batchFrames=batchFrames[:,frame_id,:,:,:]
                          
                  batchFrames=batchFrames.contiguous().to(device,non_blocking=True)
##                  print(batchFrames.size())
          else:
                  batchFrames=batchFrames.contiguous().to(device,non_blocking=True)
##                  print(batchFrames.size())
          return batchFrames


def forwardpass4models(batchX,batchFrames,batchY_history,batchY, batchUserStat,phase,model,model_type):

       if any(substring in model_type for substring in substring_list):
               #tic_fp = time.clock()
               #y_preds=model(phase,batchX,batchFrames,y_history=y_history,batchY=batchY,batchUserStat=batchUserStat) 
               y_preds=model(phase,batchX,batchFrames) 
               #toc_fp = time.clock()
               #print('Forward Pass Time',toc_fp-tic_fp)
       else:
               y_preds=model(batchX,batchY_history,batchY, batchUserStat,phase)
       return y_preds


def extract_iters(batch_idx, batch,model,model_type,push2gpu_flag):
        
       phase='val'
       seq_len=model.seq_len 
       target_len=model.target_len
       #print("IN TRAIN ITERS")
##       print(push2gpu_flag)
       batchX,batchUserStat,batchFrames,batchYSeq,frame_Names = get_batches4models(batch,model_type) 
       # first seq len -1 indices are history
       batchY_history=batchYSeq[:,0:seq_len-1,:]; batchY=batchYSeq[:,-target_len:,:]
##       print('Frame Names',frame_Names)
##       print('batchY.type(): ',batchY.type())
       if USE_CUDA and push2gpu_flag:

          #tic_batch_gpuload=time.clock()
          if batch_idx == 0:
                 print('ON GPU DEVICE '+str(device)+' '+str(torch.cuda.current_device()))
          batchX=batchX.contiguous().to(device,non_blocking=True)
          batchY=batchY.contiguous().to(device,non_blocking=True)
          #batchY_history=batchY_history.to(device,non_blocking=non_blocking)          
          #batchUserStat=batchUserStat.to(device,non_blocking=non_blocking)
          if any(substring in model_type for substring in substring_list):
                  batchFrames=load_frames4models(batchFrames,model_type)

       
       feats = forwardpass4models(batchX,batchFrames,batchY_history,batchY, batchUserStat,phase,model,model_type)
       #torch.cuda.empty_cache() 
       batchsize = batchFrames.size(0)
       return  feats,frame_Names,batchsize                   



             
def extract(train_dict,test_dict,val_dict,model_type,param_vals, uid_manager):
 
    torch.cuda.empty_cache() 
    param_vals=scriptable_params(param_vals) 

    param_vals = OrderedDict(sorted(param_vals.items(), key=lambda t: t[0]))
    model_params=param_vals.fromkeys(param_vals.keys())
    print(model_params.keys())

    #stores final results
    log_avg_dict=defaultdict(list)
    
    cnt = 0; 
  
    for p_tups in product(*param_vals.values()):
        print ("\n New Param set \n") 

        trial_predictions=getLogger() 

        cnt = cnt + 1; 
        temp_params_dict=dict(zip(param_vals.keys(),p_tups) )
        model_params.update(temp_params_dict) 
        print(model_params)

        print ("runner xtrain y train shapes",train_dict['x'].shape, train_dict['y'].shape)
        print ("runner xtest y test shapes",test_dict['x'].shape, test_dict['y'].shape)
        print ("runner xVal y Val shapes",val_dict['x'].shape,val_dict['y'].shape)

        # training set loaded into TensorDataset, 
        # only batches will be loaded onto GPU to save GPU memory
        dataloaders= getDataloader(train_dict,test_dict,val_dict,model_type,model_params, uid_manager)
          
          

        model = getModule(model_type,model_params) 
             
    
        
        print ("Extracting")

        push2gpu_flag = True




        rangevar = ['train']#,'val','test']
        for i in range(3):
            for batch_idx,batch in enumerate(dataloaders[rangevar[i]]):
               print(batch_idx)   # 
               if batch_idx >=0:                
                      if debugPrint : print ("Pre train iters allocated and cached ", batch_idx,torch.cuda.memory_allocated()/1e+9,torch.cuda.memory_cached()/1e+9)
                      
                      
                      feats,frame_Names,batchsize= extract_iters(batch_idx,batch,model,model_type,push2gpu_flag)
                      

                      for frame_num in range(batchsize):
                          frame_name = frame_Names[frame_num]
       ##                   print(frame_name)
                          if not os.path.exists(frame_name[:-22]+'places_feats'):
                              os.mkdir(frame_name[:-22]+'places_feats')
                          frame_name = frame_name.replace('frames','places_feats')
                          
                          torch.save(feats[frame_num].detach().cpu(),frame_name.replace('.jpg','')+'.pt')

                   
                   


#<-------- each epoch ends here   

        print ("Extraction done")


  
  
  
# <-- param set ends here """ 

    
    #  save average performance all parameter combinations in one file  
#    dm.save_as_mat(['hyperparams','average'],log_avg_dict)      
    return trial_predictions
