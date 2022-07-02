# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:36:42 2020

@author: vijet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tcn_mod import TCN 
import Lstm_Modules
import  SeqToSeqModels
from ARFlow_transforms import sep_transforms
#from AttnSeqToSeqModels import *
from DA_NARX import *
from KinematicsDataset import *
from VisionKinematicsDataset import *
import datamanager_v7 as dm 
import convLSTM_modules
import VisionReplication_modules
import Vision4Prosthetics_modules 
import device_utils
import os
from torch.autograd import Variable
from LstmUserStat_Modules import *
from torch.optim.lr_scheduler import *
import runner_v8 as runner
from easydict import EasyDict

global cuda_device
USE_CUDA = torch.cuda.is_available()
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     
cuda_device = device_utils.getCudaDevice()
#os.environ['CUDA_VISIBLE_DEVICES'] = device_utils.getCudaDeviceidx()
print ("device for torch models and data",cuda_device)




def saveTorchModel(model, model_type, param_cnt, trial_num):
     
    print ("Model type", model_type )
 
    path=dm.get_model_results_dir() 
    dm.make_folder_in_results(path)
    path = path+ 'param_' + str(param_cnt) + '_trial_' + str(trial_num) + '.pt'
    print ("path for mdoel",path)
        
    if model_type =='SeqToOne' or model_type =='SeqToTwo' or model_type=='HistorySeqToTwo' :
        model_opt=model.optimizer;
        model_scheduler=model.scheduler;
        
        torch.save({
              'model_state_dict': model.state_dict(),
              'model_optimiser_state_dict': model_opt.state_dict(),
              'model_scheduler_state_dict': model_scheduler.state_dict()
              },path)
 
         
    elif model_type=='SeqToSeq_Ankle':
        enc_opt=model.encoder_optimizer;
        dec_opt=model.decoder_optimizer;
        torch.save({
              'model_state_dict': model.state_dict(),
              'encoder_optimiser_state_dict': enc_opt.state_dict(),
              'decoder_optimizer_state_dict': dec_opt.state_dict()
              },path)
    elif model_type=='Pretraining_Conv2Act':
        model_opt=model.optimizer;
        model_scheduler=model.scheduler;
        
        torch.save({
              'model_state_dict': model.state_dict(),
              'model_optimiser_state_dict': model_opt.state_dict(),
              'model_scheduler_state_dict': model_scheduler.state_dict()
              },path)
    
def add_gaussian(ins, is_training, mean, stddev):
    if is_training:
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        return ins + noise
    return ins


def getModule(model_type,model_params,dataloaders=None,override_opt=False,custom_optimizer=None,override_scheduler=False,custom_scheduler=None):
    
    
    
    if model_type =='SeqToOne' :
        print ("Model type", model_type )
        clipGrads= False
        model = Lstm_Modules.Lstm_SeqtoOne_Module(model_params)
        optimiser = torch.optim.Adam(model.parameters(),  lr=model_params['learning_rate'])
        model.setOptimizer(optimiser)
        
    elif model_type =='HistorySeqToOne' :
        print ("Model type", model_type )
        clipGrads= False
        model = Lstm_Modules.LstmHistory_SeqtoOne_Module(model_params)
        optimiser = torch.optim.Adam(model.parameters(),  lr=model_params['learning_rate'])
        model.setOptimizer(optimiser)    
 
    elif model_type =='SeqToTwo':
        print ("Model type",model_type )
        clipGrads= False
        model = Lstm_Modules.Lstm_SeqtoTwo_Module(model_params) 
        if override_opt == False:
            optimizer=torch.optim.Adam(model.parameters(),  lr=model_params['learning_rate'], weight_decay=model_params['regularization'])
        else:
            optimizer = custom_optimizer
        if override_scheduler == False:
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=model_params['plateau_schedular_patience'], factor=model_params['decay_factor'])
        else:
            scheduler = custom_scheduler
        model.setOptimizer(optimizer,scheduler)  
    elif model_type =='HistorySeqToTwo' :
        print ("Model type", model_type )
        clipGrads= False
        model = Lstm_Modules.LstmHistory_SeqtoTwo_Module(model_params)
        optimiser = torch.optim.Adam(model.parameters(),  lr=model_params['learning_rate'])
        model.setOptimizer(optimiser)
         
    elif model_type=='HistorySeqToSeq_Ankle':
        print ("SeqToSeq_Ankle")
        clipGrads= False
        encoder=SeqToSeqModels.Encoder_Ankle(model_params)
        encoder_opt= torch.optim.Adam(encoder.parameters(),  lr=model_params['learning_rate'])
        
        decoder=SeqToSeqModels.Decoder_Ankle(model_params)
        decoder_opt= torch.optim.Adam(decoder.parameters(),  lr=model_params['learning_rate'])
        
        model=SeqToSeqModels.EncoderDecoder_Ankle(encoder,decoder,model_params)
        model.setOptimizer(encoder_opt,decoder_opt)
 
    elif model_type=='AttnSeqToSeq_Ankle':
        
        print ("Attn_Ankle")
        clipGrads= True
        attn_model = 'dot'
        #attn_model = 'general'
        #attn_model = 'concat'
        encoder=Encoder_Ankle(model_params)
        encoder_opt= torch.optim.Adam(encoder.parameters(),  lr=model_params['learning_rate'])
        
        decoder=LuongAttnDecoder_Ankle(attn_model,model_params)
        decoder_opt= torch.optim.Adam(decoder.parameters(),  lr=model_params['learning_rate'])
        
        model=AttnEncoderDecoder_Ankle(encoder,decoder,model_params)
        model.setOptimizer(encoder_opt,decoder_opt)
    
            
    elif model_type =='TCN_Ankle':
        channel_sizes = [model_params['num_hid_units']] *model_params['tcn_levels']
        #print ("channels",channel_sizes)
        kernel_size =model_params['kernel_size'] 
        dropout = model_params['dropout']
        out_dim=1; input_channels=model_params['num_features']
        model = TCN(input_channels, out_dim, channel_sizes, kernel_size=kernel_size, dropout=dropout) 
         
        adam_optimizer=torch.optim.Adam(model.parameters(),  lr=model_params['learning_rate'])
        
        model.setOptimizer(adam_optimizer)
        
        
    elif model_type=='ConvLSTM_Ankle':
         print ("ConvLSTM_Ankle")
         path = os.path.dirname(__file__) + '\models\model_conv.pt' 
         print ("torch util path",path)
         model=convLSTM_modules.ConvLstm_Ankle(model_params,path)
         adam_optimizer=torch.optim.Adam(model.parameters(),  lr=model_params['learning_rate'])
        
         model.setOptimizer(adam_optimizer)
         
    elif model_type=='NARX_Ankle' or model_type=='NoHistoryNARX_Ankle':
        
        print ("NARX",model_type)
        encoder = Encoder_Ankle(model_params) 
        
        if model_type=='NoHistoryNARX_Ankle':
            print ("no history decoder set")
            decoder = NoHistory_Decoder_Ankle(model_params)

        else:    
            decoder = Decoder_Ankle(model_params)
        
        encoder_optimizer = optim.Adam(params=[p for p in encoder.parameters() if p.requires_grad], lr=model_params['learning_rate'])
        decoder_optimizer = optim.Adam(params=[p for p in decoder.parameters() if p.requires_grad], lr=model_params['learning_rate'])
        model = NARX_EncoderDecoder_Ankle(encoder, decoder, model_params)
        model.setOptimizer(encoder_optimizer, decoder_optimizer)
        
        test_num_samples=len(dataloaders['test'].dataset)
        val_num_samples=len(dataloaders['val'].dataset)
        
        model.test_attn_saver.set_size(test_num_samples)
        model.val_attn_saver.set_size(val_num_samples)
        
    elif model_type=='UserStatSeqToOne':
        print ("Model type", model_type )
        clipGrads= False
        model = LstmUserStat_SeqtoOne_Module(model_params)
        optimiser = torch.optim.Adam(model.parameters(),  lr=model_params['learning_rate'])
        model.setOptimizer(optimiser)
         
    elif model_type == 'Pretraining_Conv2Act' :
        print ("Model type", model_type )
        clipGrads= False              
##        path = os.path.dirname(__file__) + '\models\model_conv.pt' 
##        print ("torch util path",path)
        model=VisionReplication_modules.PretrainConv2Action(model_params)
        if override_opt == False:
            optimizer=torch.optim.Adam(model.parameters(),  lr=model_params['learning_rate'], weight_decay=model_params['regularization'])
        else:
            optimizer = custom_optimizer
        if override_scheduler == False:
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=model_params['plateau_schedular_patience'], factor=model_params['decay_factor'])
        else:
            scheduler = custom_scheduler
        model.setOptimizer(optimizer,scheduler)
        
    elif model_type == 'Scene4Prosthetics' :
        print ("Model type", model_type )
        clipGrads= False              
##        path = os.path.dirname(__file__) + '\models\model_conv.pt' 
##        print ("torch util path",path)
        model=Vision4Prosthetics_modules.Scene4Prosthetics(model_params)
        if override_opt == False:
            optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),  lr=model_params['learning_rate'], weight_decay=model_params['regularization'])
        else:
            optimizer = custom_optimizer
        if override_scheduler == False:
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=model_params['plateau_schedular_patience'], factor=model_params['decay_factor'])
        else:
            scheduler = custom_scheduler
        model.setOptimizer(optimizer,scheduler)

    elif model_type == 'OpticalFlow4Prosthetics' :
        print ("Model type", model_type )
        clipGrads= False              
##        path = os.path.dirname(__file__) + '\models\model_conv.pt' 
##        print ("torch util path",path)
        model=Vision4Prosthetics_modules.OpticalFlow4Prosthetics(model_params)
        if override_opt == False:
            optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),  lr=model_params['learning_rate'], weight_decay=model_params['regularization'])
        else:
            optimizer = custom_optimizer
        if override_scheduler == False:
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=model_params['plateau_schedular_patience'], factor=model_params['decay_factor'])
        else:
            scheduler = custom_scheduler
        model.setOptimizer(optimizer,scheduler)                    

    elif model_type == 'OpticalFlow_ftex' :
        print ("Model type", model_type )
        clipGrads= False              
##        path = os.path.dirname(__file__) + '\models\model_conv.pt' 
##        print ("torch util path",path)
        model=Vision4Prosthetics_modules.OpticalFlow_ftex(model_params)

    elif model_type == 'PlacesScene_ftex' :
        print ("Model type", model_type )
        clipGrads= False              
##        path = os.path.dirname(__file__) + '\models\model_conv.pt' 
##        print ("torch util path",path)
        model=Vision4Prosthetics_modules.Places_ftex(model_params)
        
    if USE_CUDA: model=model.to(device=cuda_device) 
 
    return  model

##def initialize_scheduler(scheduler_type,optimizer,model_params):
##    if scheduler_type == 
##    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=model_params['plateau_schedular_patience'], factor=model_params['decay_factor'])


def getImageTransform(model_type):


      #substring_list = ['Conv','Vision','Scene','Optical','Spatiotemporal']
    
    if 'OpticalFlow' in model_type: 
        cfg = {'model': {'upsample': False,'n_frames': 2,'reduce_dense': True},
                   'pretrained_model': 'checkpoints/Sintel/pwclite_ar.tar',
                   'test_shape': [384, 640],}        
        cfg = EasyDict(cfg)    
        data_transforms = {
        'train':transforms.Compose([
                sep_transforms.Zoom(*cfg.test_shape),
                sep_transforms.ArrayToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ]),
        'val':transforms.Compose([
                sep_transforms.Zoom(*cfg.test_shape),
                sep_transforms.ArrayToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ]),
        'test':transforms.Compose([
                sep_transforms.Zoom(*cfg.test_shape),
                sep_transforms.ArrayToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ]),
        }
    else:   
        data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),

            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
           'test': transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        }
    
    
    return data_transforms    
    

def getDataloader(train_dict,test_dict,val_dict,model_type,model_params, uid_manager):
    
    
      sanity_check,substring_list,num_scenesamples,_,_ = runner.GetTrainingOpts()
    
      #substring_list = ['Conv','Vision','Scene','Optical','Spatiotemporal']
    
      if any(substring in model_type for substring in substring_list):

          print('Loading Vision-Kinematics Dataset') 
          composed_transforms=getImageTransform(model_type)
             
          training_dataset=VisionKinematicsDataset(train_dict, uid_manager,transforms=composed_transforms['train'])
          test_dataset=VisionKinematicsDataset(test_dict, uid_manager,transforms=composed_transforms['test'])
          val_dataset=VisionKinematicsDataset(val_dict, uid_manager,transforms=composed_transforms['val'])
             
          training_loader=torch.utils.data.DataLoader(training_dataset,batch_size=model_params['batch_size'], num_workers=model_params['num_workers'], shuffle=model_params['shuffle'], pin_memory=model_params['pin_memory'])
          if sanity_check==False:
                  test_loader=torch.utils.data.DataLoader(test_dataset,model_params['batch_size'],num_workers=2, shuffle=False)#2
                  val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=model_params['batch_size'],num_workers=6, shuffle=False)#6
          else:
                  test_loader = None
                  val_loader = None         

          
          dataloaders={'train': training_loader,
                  'val': val_loader,
                  'test':test_loader                  }
      
      else :
         print('Loading Kinematics Dataset') 
         training_dataset=KinematicsDataset(train_dict)
         test_dataset=KinematicsDataset(test_dict)
         val_dataset=KinematicsDataset(val_dict)
         
         training_loader=torch.utils.data.DataLoader(training_dataset,batch_size=model_params['batch_size'],num_workers=model_params['num_workers'],  shuffle=True, pin_memory=model_params['pin_memory'])
         if sanity_check==False:
                 test_loader=torch.utils.data.DataLoader(test_dataset,model_params['batch_size'],num_workers=model_params['num_workers'], shuffle=False)
                 val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=model_params['batch_size'],num_workers=model_params['num_workers'], shuffle=False)
         else:
                 test_loader = None
                 val_loader = None
      
         dataloaders={'train': training_loader,
                  'val': val_loader,
                  'test':test_loader                   
                }
      
        
      return dataloaders

def save_attention(model,model_type,phase,cnt, iter_trial)  :
    
    if model_type =='NARX_Ankle' or model_type=='NoHistoryNARX_Ankle':
        if phase == 'val':
            val_dict=model.val_attn_saver.get_recorded_dict()
            dm.save_as_mat(['outputs','val_attn_data_trial'+str(iter_trial)],val_dict)
            
        elif  phase == 'test':
            test_dict=model.test_attn_saver.get_recorded_dict()
            dm.save_as_mat(['outputs','test_attn_data_trial'+str(iter_trial)],test_dict)   
     
    else : 
        print ("Error: This model type doesnt't support attentions")
        
        
    return    
         
def save_vision_preds(model,model_type,phase,cnt, iter_trial)  :
    
    _,substring_list,_,_,_ = runner.GetTrainingOpts()
    
      #substring_list = ['Conv','Vision','Scene','Optical','Spatiotemporal']
    
    if any(substring in model_type for substring in substring_list):
        if  phase == 'test':
            test_dict=model.saver.get_recorded_dict()
            dm.save_as_mat(['outputs','test_conv_data_trial'+str(iter_trial)],test_dict)   
     
    else : 
        print ("Error: This model type doesnt't support vision saves")
        
        
    return           
    
