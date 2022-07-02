
import torch
import torch.nn as nn
import datamanager_v7 as dm

from datetime import datetime
import os

from torch_utils import *
from log_utils import *
from train_eval_functions import evaluate
from loss_modules import RMSELoss
from VisionKinematicsDataset import *
from KinematicsDataset import *
import runner_v8 as runner

import numpy as np
from itertools import product
from collections import defaultdict
from collections import OrderedDict
from math import sqrt 

timestamp= datetime.now().strftime("%m_%d_%H_%M")


    

if __name__ == "__main__":

    dara_dir = "D:/Vision_Replication/data/vision_pretraining"

    config_file_name = "ConfigFiles/v6_cf_unstruct_scene-test.csv"

    expt_path = "D:/ExhaustiveUnstructuredModeFreeTests/OF_wo-norm_ph30_02_05_20_26/checkpoints"
    model_path = expt_path + '/checkpoint_param_2_trial_2.pt'
    model_type = 'OpticalFlow4Prosthetics'

    results_path= expt_path

    dm.set_results_dir_manual(results_path)


    param_vals = { 'num_epochs': [1000], 'batch_size':[100], 'minibatch_GD': True, 'num_trials':[3], 'shuffle': True,
                  
                   'seq_length':[10],'target_len':[1], 'pred_horizon':[30], # THESE ARE FIXED IN THE DATAMANAGER. NO LIST.
                   
                   'num_layers':[2],'num_hid_units':(32), 'num_features':(57), 'num_vis_features':[32,64,128],'num_vis_layers':[2],'numscenesample':1,'num_classes': [8], 'weighted_classification': [True],
                   'num_fc_layers': [4],'fc_taper':[2], 
                   'pretrained_LSTM': False, 'backprop_kinematics': False, 'backprop_vision': False, 'zero_like_vision':False,

                   'window_step':[1],'sub_sample_factor':[1], 'frame_subsample_factor':[1],

                   'noise_std':[0.02],'clip':[-1],'dropout':[0.0], 'pretrainC2A_w_Dropout':True, 'regularization':[0],
                   'learning_rate': [1e-5], 'custom_lr_decay_rate':[0.1], 'custom_lr_decay_step':[90],
                   'use_scheduler': [True], 'plateau_schedular_patience':5, 'decay_factor': 0.1,'EarlyStoppingPatience':30, 'Early_Stopping_delta':0,'valid_patience':1,
                   'num_workers':(12),'pin_memory':False,'non_blocking':True, 'device':(2)}

    param_vals=OrderedDict(param_vals)
     

    print(param_vals.keys())

    model_type='OpticalFlow4Prosthetics'##'SeqToTwo'## #['Conv','Vision','Scene','Optical','Spatiotemporal']
    use_features = True#False #True
    runner.SetTrainingOpts(sanityflag=False,use_single_batch=False,overlap_datatransfers=param_vals['non_blocking'],model_name=model_type,use_feats=use_features,indie_checkpoints=True)#,clipGradients,indie_checkpoints,vision_substrings,numscenesamples=paramvals['numscenesample'])
    scheduler_type = 'ReduceOnPlateau'    
    

    train_dict, test_dict, val_dict , uid_manager = dm.get_and_reformat_all_data(data_dir, config_file_name,param_vals)


    print ('test shape',test_dict['y'].shape)



    model=getModule(model_type,param_vals) 
    criterion = torch.nn.MSELoss(reduction='mean')#RMSELoss()

    dataloaders= getDataloader(train_dict,test_dict,val_dict,model_type,model_params, uid_manager)


    log_predictions=getLogger() 

    thisTrial_model_dict = torch.load(model_path)
    model.load_state_dict(thisTrial_model_dict['model_state_dict'])
       #optimiser.load_state_dict(bestModel_dict['optimiser_state_dict'])

    yTests,yPreds=evaluate(model,model_type,dataloaders['test'])

    this_trial_rmse=criterion(yPreds.view(yTest.shape), yTests).item()

    print ("RMSE",this_trial_rmse)
       
    log_predictions['y_preds'].append(yPreds.cpu().detach().numpy());
    log_predictions['y_tests'].append(yTests.cpu().detach().numpy())
    log_predictions['trial_info'].append(param_vals)
    log_predictions['test_RMSE'].append(this_trial_rmse)
       #log_predictions['Epoch vs Loss'].append(hist)
         
    dm.save_as_mat([model_type + '/preds','predictions'],log_predictions) 
    #dm.save_as_mat(['hyperparams','predictions_1testtrial'],log_predictions) 
    
