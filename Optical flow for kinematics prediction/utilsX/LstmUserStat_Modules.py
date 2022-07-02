#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:03:06 2019

@author: raiv
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import device_utils
device = device_utils.getCudaDevice()
import torch_utils as t_utils #import add_gaussian
USE_CUDA = torch.cuda.is_available()

class LstmUserStat_SeqtoOne_Module(nn.Module):

    def __init__(self, model_params):
#         __init__(self, num_features, hidden_dim, batch_size, output_dim=1,
#                    num_layers=2,seq_len=1):
        super(LstmUserStat_SeqtoOne_Module, self).__init__()
        self.num_features =model_params['num_features']
        self.hidden_dim = model_params['num_hid_units']
        self.batch_size = model_params['batch_size']
        self.num_layers = model_params['num_layers']
        self.seq_len= model_params['seq_length']
        self.target_len= model_params['target_len']
        self.num_user_stats=model_params['num_stats']
        self.output_dim = 1 # for Sequence to One
        self.train_noise_std=model_params['noise_std']
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.num_features, self.hidden_dim, self.num_layers)
       
          # Define Fully Connected Layer for UserStats
        self.user_net = nn.Linear(self.num_user_stats,self.hidden_dim) # Same output dimension as LSTM
       
        # Define Fully Connected Layer to Merge LSTM and UserStats FC

        self.merge_net = nn.Linear(2 * self.hidden_dim,self. output_dim) # 2 * self.hidden_dim is the same as lstm.hidden_dim + fc.hidden_dim



    def init_hidden(self):
        # This is what we'll initialise our hidden state a
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input_seq,input_user_stats):

        
        input_seq=input_seq.permute(1,0,2) # seq length first, batch, input_features
        #print ("input shape",input.shape)
        lstm_out, self.hidden = self.lstm(input_seq)
        
        user_net_out= self.user_net(input_user_stats)

        lstm_user_cat=torch.cat((lstm_out[-1],user_net_out),1) 

        final_out= self.merge_net(lstm_user_cat)
        # print ("lstm out shape",lstm_out.shape ) [seq_len,batch_size,hid_dim]

        
        # print ("lstm o/p last timestep", lstm_out[-1].shape) [batch_size,hid_dim]
#        y_pred = self.linear(lstm_out[-1])
        
        #print ('y_pred.shape', y_pred.view(-1).shape) #
        return  final_out.unsqueeze(1).view(-1,1,self.output_dim)
    
    def setOptimizer(self,optimizer):
        self.adamOpt= optimizer
        
    def zeroGrad(self):
        self.adamOpt.zero_grad()
        
    def stepOptimizer(self):
         self.adamOpt.step()
         
    def set_phase(self,phase):
        if phase =='val' or phase =='test':
            self.eval()  
        elif phase =='train':
            self.train()
    
    def execute_model(self,batch,phase):
        if phase =='val' or phase =='test':
           self.set_phase(phase)
           return self.evaluate_model(batch)
            
        elif phase =='train':
           self.set_phase(phase)
           return self.train_model(batch)
             
             
             
    def train_model(self,batch):
        
        
        batchX,batchUserStat,batchYSeq=batch
        # first seq len -1 indices are history
                       
        #last target_len indices are actual target
        batchY=batchYSeq[:,-self.target_len:,:]

        if USE_CUDA:
          batchX=batchX.to(device)
          batchY=batchY.to(device)
          batchUserStat=batchUserStat.to(device) 
            
        batchX=t_utils.add_gaussian(batchX,is_training=True,mean=0,stddev=self.train_noise_std)
         
        batchY_pred= self.forward(batchX,batchUserStat)  
        
        return batchY,batchY_pred            
                       
    def evaluate_model(self,batch)  :
        
        with torch.no_grad():
            batchX,batchUserStat,batchYSeq=batch
            batchY=batchYSeq[:,-self.target_len:,:]

            if USE_CUDA:
                              
              batchX=batchX.to(device)
              batchY=batchY.to(device)
              batchUserStat=batchUserStat.to(device) 

            
            batchY_preds=self.forward(batchX,batchUserStat)
            return batchY,batchY_preds
