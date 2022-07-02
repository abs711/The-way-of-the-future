

import sys
sys.path.append("D:/Vision_Replication/DL_Code/utilsX") #"/home/spc/pytorch_challenge-master/DL_Code/utilsX"

import device_utils

cuda_device_idx=0
device_utils.setCudaDevice(cuda_device_idx)

import torch
import torch.nn as nn
import datamanager_v7 as dm
import matplotlib.pyplot as plt
import extraction_runner
import runner_v8 as runner
import scipy.io
from datetime import datetime
import os
from collections import defaultdict

import device_utils

if __name__ == "__main__":

### Sequence length list doesn't work due to data formatting procedure
    param_vals = { 'batch_size':[90], 'shuffle': False,
                   'seq_length':[1],'target_len':[1], 'pred_horizon':[0],
                   'num_vis_features':[3],'numscenesample':1,
                   'window_step':[1],'sub_sample_factor':[1], 'frame_subsample_factor':[1],'kernel_size':(7),
                   'num_workers':(3),'pin_memory':False,'non_blocking':True, 'device':(0)}#8
    
    
    model_type='PlacesScene_ftex'##'SeqToTwo'## #['Conv','Vision','Scene','Optical','Spatiotemporal']

    data_dir = "D:/Vision_Replication/data/vision_pretraining/" #'/home/spc/pytorch_challenge-master/data/vision_kinematics/Unstructured_data/Unstructured_Data/' #"/home/spc/Mega-Tron/SeqtoSeq_Kinematics/"#"D:/Vision_Replication/data/vision_pretraining/" 
    config_file_name ="ConfigFiles/v6_cf_unstruct_scene-test.csv" # "v6_cf_flat(Ankle).csv"# "ConfigFiles/v6_cf_unstruct_pretraining.csv"
##    config_file_name = "ConfigFiles/v6_cf_unstruct(Ankle)-test.csv"

    runner.SetTrainingOpts(sanityflag=False,use_single_batch=False,overlap_datatransfers=param_vals['non_blocking'],model_name=model_type)#,clipGradients,indie_checkpoints,vision_substrings,numscenesamples=paramvals['numscenesample'])

    train_dict, test_dict, val_dict , uid_manager = dm.get_and_reformat_all_data(data_dir, config_file_name,param_vals)
    
##    dm.root_dir = data_dir
##    dm.root_datadir=data_dir + 'processed/'
##
##    config_df = dm.load_data_config(config_file_name)

##    
##    train_dir = config_df['train_datadir'].dropna().tolist())
##    test_dir = config_df['test_datadir'].dropna().tolist())
##    val_dir = config_df['val_datadir'].dropna().tolist())
##
##    for sub_dirs in train_dir :
    trial_predictions = extraction_runner.extract(train_dict,test_dict,val_dict,model_type,param_vals,uid_manager)
    
