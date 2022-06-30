

import sys
sys.path.append("../utilsX")

import datamanager_v7 as dm
import os


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
    


    data_dir = "./" 
    config_file_name ="ConfigFiles/v6_cf_unstruct_scene-test.csv" 
    


    train_dict, test_dict, val_dict , uid_manager = dm.get_and_reformat_all_data(data_dir, config_file_name,param_vals)

    print(train_dict.keys())
    
