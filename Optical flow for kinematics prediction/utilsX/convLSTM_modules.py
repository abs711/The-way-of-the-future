# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 22:58:15 2020

@author: vijet
"""

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from Lstm_Modules import LstmHistory_SeqtoOne_Module
debugPrint = 1;
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
import device_utils
device = device_utils.getCudaDevice()

def decimate(tensor_input)  :
        idx = torch.arange(0, tensor_input.shape[0], step=5).to(device) 
        tensor_decimated=torch.index_select(tensor_input, 0, idx)
#        print ("input decimated",tensor_input.shape,tensor_decimated.shape)
        return tensor_decimated
    
class conv_saver():
     
    def __init__(self):
        self.batch_size=0
        self.y_tests=[]
#        self.x_frames=[]
        self.y_preds=[]
        self.vision_preds=[]
        
    def set_size(self,size) :
        self.batch_size=size
 
        
    def record_data (self,x_frames,y_tests,y_preds,vision_preds):
        
        
#        self.x_frames.append(decimate(x_frames).detach().cpu());
        self.y_tests.append(decimate(y_tests).detach().cpu());
        self.y_preds.append(decimate(y_preds).detach().cpu());
        self.vision_preds.append(decimate(vision_preds).detach().cpu())
        
    def get_recorded_dict(self):
     
        
        def cat_em_up(values_list):
            if len(values_list) > 0 :
                big_cat=torch.cat(values_list)   
            else :
                print ("no cat in the bag")
                big_cat=torch.zeros([1])
                
            return big_cat.detach().cpu().numpy()    
        
            
#        tic=time.clock() 
        data={
#              'x_frames':cat_em_up(self.x_frames),
              'y_tests': cat_em_up(self.y_tests),
               'y_preds': cat_em_up(self.y_preds),
              'vision_preds': cat_em_up(self.vision_preds)
               }
     
        return data
    
    
class ConvLstm_Ankle(nn.Module):

    def __init__(self, model_params,path):
        
        super(ConvLstm_Ankle, self).__init__()
        self.num_vision_classes=2

        
        self.num_features =model_params['num_features']
        self.hidden_dim = model_params['num_hid_units']
        self.batch_size = model_params['batch_size']
        self.num_layers = model_params['num_layers']
        self.seq_len= model_params['seq_length']
        self.target_len= model_params['target_len']
        self.out_dim=1
        self.vision_feature_extracter= self.get_pretrained_vision_model(path)
        
        self.LSTM_vision_net=nn.LSTM(self.num_vision_classes, self.hidden_dim, self.num_layers)
        self.LSTM_kinematics_net= nn.LSTM(self.num_features+1, self.hidden_dim, self.num_layers) # +1 for y history 

        self.final_fc=nn.Linear(self.hidden_dim*2,self.out_dim)
        
        self.saver=conv_saver()
    def get_pretrained_vision_model(self,path):
        
         print ("conv path",path)
          
         model = models.resnet18(pretrained=True)
         num_ftrs = model.fc.in_features
         model.fc = nn.Linear(num_ftrs, self.num_vision_classes)
         
         optimiser = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    
         model_dict=  torch.load(path)
         model.load_state_dict( model_dict['model_state_dict'] )
    
         optimiser=model_dict['optimiser_state_dict']   
         
         return model
     
    def forward (self,input_seq,y_history,batchY,batchUserStat,batchFrames,phase)  :
        
        # cat history and input
         last_y_history=y_history[:,self.seq_len-2,:].unsqueeze(2)
         y_history=torch.cat((y_history,last_y_history),dim=1)
       
         input_seq=torch.cat((input_seq,y_history),dim=2)
         input_seq=input_seq.permute(1,0,2) # seq length first, batch, input_features
        
        # get vision outs
         vision_outs=[];preds_seq=[];
         self.vision_feature_extracter.eval()
         with torch.no_grad():
             for frame_idx in range(batchFrames.shape[1]):
                 outs=self.vision_feature_extracter(batchFrames[:,frame_idx])
                 probs = F.softmax(outs, dim=1)
                 _, preds = torch.max(outs, 1)
                 #print("probs, preds",probs.data,preds.data)
                 vision_outs.append(probs)
                 preds_seq.append(preds)
                 
         vision_outs=torch.stack(vision_outs,dim=1)
         preds_seq=torch.stack(preds_seq,dim=1)
         
         vision_lstm_outs,vision_hidden_last  =self.LSTM_vision_net(vision_outs.permute(1,0,2))
         kine_lstm_outs,kine_hidden_last=self.LSTM_kinematics_net(input_seq) 
         
         vision_kine_cat= torch.cat((vision_hidden_last[0],kine_hidden_last[0]),dim=2).permute(1,0,2)
         
#         if debugPrint : print ("forward allocated and cached ", torch.cuda.memory_allocated()/1e+9,torch.cuda.memory_cached()/1e+9)
         final_out=self.final_fc(vision_kine_cat)
         
         with torch.no_grad():
           if phase =='test':
                #print ("recording test attentions")
                
                
                self.saver.record_data(batchFrames,batchY,final_out,vision_outs)
                
         return  final_out.view(-1,self.out_dim)
     
    def setOptimizer(self,optimizer):
        self.adamOpt= optimizer
        
    def zeroGrad(self):
        self.adamOpt.zero_grad()
        
    def stepOptimizer(self):
         self.adamOpt.step()
    
    def set_phase(self,phase):
        if phase =='val' or phase =='test':
            self.eval()  
        elif phase=='train':
            self.train()   
        
