

import sys
sys.path.append("../utilsX") 

import device_utils

cuda_device_idx=0
device_utils.setCudaDevice(cuda_device_idx)

import torch
import torch.nn as nn
import datamanager_v7 as dm
import matplotlib.pyplot as plt
import runner_v8 as runner
import scipy.io
from datetime import datetime
import os
from collections import defaultdict

import device_utils

if __name__ == "__main__":

### Sequence length list doesn't work due to data formatting procedure
    param_vals = { 'num_epochs': [2], 'batch_size':[5], 'minibatch_GD': True, 'num_trials':[1], 'shuffle': True,
                   'seq_length':[10],'target_len':[1], 'pred_horizon':[3],
                   'num_layers':[1],'num_hid_units':(32), 'num_features':(57), 'num_vis_features':[512],'num_vis_layers':[2],'numscenesample':1,'num_classes': [8], 'weighted_classification': [True],
                   'num_fc_layers': [4],'fc_taper':[2],
                   'window_step':[1],'sub_sample_factor':[1], 'frame_subsample_factor':[1],'kernel_size':(7),
                   'noise_std':[0.02],'clip':[-1],'dropout':[0.0], 'pretrainC2A_w_Dropout':True, 'regularization':[0.001],
                   'learning_rate': [1e-1], 'custom_lr_decay_rate':[0.1], 'custom_lr_decay_step':[90],
                   'use_scheduler': [True], 'plateau_schedular_patience':5, 'decay_factor': 0.1,'EarlyStoppingPatience':20, 'Early_Stopping_delta':0,'valid_patience':1,
                   'num_workers':(1),'pin_memory':False,'non_blocking':True, 'device':(0), 'pretrained_LSTM': False, 'backprop_kinematics': True, 'backprop_vision': False, 'zero_like_vision':False}
    
#    data_dir = os.path.dirname(__file__) + '/data/vision_kinematics/'
#    config_file_name = "ConfigFiles/v6_cf_flat_wstop(Ankle).csv"
    
    model_type='OpticalFlow4Prosthetics'##'SeqToTwo'## #['Conv','Vision','Scene','Optical','Spatiotemporal']
    runner.SetTrainingOpts(sanityflag=True,use_single_batch=False,overlap_datatransfers=param_vals['non_blocking'],model_name=model_type)#,param_vals['backprop_kinematics'],clipGradients,indie_checkpoints,vision_substrings,numscenesamples=paramvals['numscenesample'])
    scheduler_type = 'ReduceOnPlateau'
    folder_annotation='OF_Pretraining_sanity'#'ModeFree_Unstructured_sanity'#'ModeFree_Test'


    data_dir = "D:/Vision_Replication/data/vision_pretraining/" #'/home/spc/pytorch_challenge-master/data/vision_kinematics/Unstructured_data/Unstructured_Data/' #"/home/spc/Mega-Tron/SeqtoSeq_Kinematics/"#"D:/Vision_Replication/data/vision_pretraining/" 
    config_file_name ="ConfigFiles/v6_cf_unstruct_scene-test.csv" # "v6_cf_flat(Ankle).csv"# "ConfigFiles/v6_cf_unstruct_pretraining.csv"
##    config_file_name = "ConfigFiles/v6_cf_unstruct(Ankle)-test.csv"
    

    ### ---< double check everything above > -----
    device_utils.setCudaDevice(param_vals['device'])

    timestamp= datetime.now().strftime("%m_%d_%H_%M")
    train_dict, test_dict, val_dict , uid_manager = dm.get_and_reformat_all_data(data_dir, config_file_name,param_vals)

    #assert
    assert(param_vals['numscenesample']<=min(param_vals['seq_length'])) 
    
    if model_type == 'Pretraining_Conv2Act' and param_vals['pretrainC2A_w_Dropout']== False:
        assert(len(param_vals['dropout']) == 1), "REPEATING SAME HYPERPARAMS Due to NO DROPOUT"

    if param_vals['use_scheduler']== False:
        assert(param_vals['valid_patience'] == 1), "VALIDATE PER EPOCH TO USE REDUCEONPLATEAU LR_SCHEDULER"        
    
    num_features_data=train_dict['x'].shape[2]
    assert(param_vals['num_features'] == num_features_data),"num features mismatch in data"
    
    print ("Sample factors", dm.get_field_from_config('sub_sample_factor'),dm.get_field_from_config('frame_subsample_factor'))
 
    dm.set_global_flags(model_type,timestamp,scheduler_type)
    
    #auto needs model and timestamp to be set prior
    dm.set_results_dir_auto(folder_annotation)
    dm.save_settings(param_vals)   
    
#    dm.save_tensors_as_mat(['inputs','trainMat'],train_dict)  
    dm.save_tensors_as_mat(['inputs','testMat'],test_dict)   
    
    
    trial_predictions = runner.train(train_dict,test_dict,val_dict,model_type,param_vals,uid_manager)
