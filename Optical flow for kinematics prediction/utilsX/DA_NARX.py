

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:03:06 2019

@author: raiv
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
debugPrint=0;

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NARX_saver():
     
    def __init__(self):
        self.batch_size=0
    #    self.val_size=val_size
        
        self.features_attn_wts_collect=[];
        self.input_encoded_collect=[];
        
        self.x=[];
        self.y_preds=[];
        self.y_tests=[];
        self.dec_hidden_collect=[];
        self.contexts_collect=[]
        self.temporal_attn_wts_collect=[];
        self.x_weighted_collect=[]
    def set_size(self,size) :
        self.batch_size=size
        
    def record_data (self,y_tests, y_preds,features_attn_wts, input_encoded,x_weighted, dec_hidden,contexts_all,temporal_attn_wts):
        
        
        self.y_tests.append(y_tests);
        self.y_preds.append(y_preds);
        
        self.features_attn_wts_collect.append(features_attn_wts);
        self.input_encoded_collect.append(input_encoded);
        self.x_weighted_collect.append(x_weighted)
        self.dec_hidden_collect.append(dec_hidden);
        self.contexts_collect.append(contexts_all)
        self.temporal_attn_wts_collect.append(temporal_attn_wts);
        
    def get_recorded_dict(self):
        def stack_em_up(value_list):
            return torch.stack(value_list).detach().cpu().numpy()
        def cat_em_up(values_list):
            if len(values_list) > 0 :
                big_cat=torch.cat(values_list)   
            else :
                print ("no cat in the bag")
                big_cat=torch.zeros([1])
                
            return big_cat.detach().cpu().numpy()    
        
            
#        tic=time.clock() 
        data={
                'x':cat_em_up(self.x),
              'y_tests': cat_em_up(self.y_tests),
               'y_preds': cat_em_up(self.y_preds),
               'feature_attn_wts':cat_em_up(self.features_attn_wts_collect),
               'input_encoded':cat_em_up(self.input_encoded_collect),
               'x_weighted':cat_em_up(self.x_weighted_collect),
               'dec_hidden':cat_em_up(self.dec_hidden_collect),
               'contexts':cat_em_up(self.contexts_collect),
               'temporal_attn_wts':cat_em_up(self.temporal_attn_wts_collect)               
               }
#        toc=time.clock()
        
#        print ("saved sizes (time)\n",toc-tic)
#        for key,val in data.items():               
#                print ("key",key +"\tshape",val.shape)
#            
            
        return data
        
class NARX_EncoderDecoder_Ankle(nn.Module):
    
    def __init__(self, encoder, decoder,model_params):
        super(NARX_EncoderDecoder_Ankle, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        self.batch_size = model_params['batch_size']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        
        self.test_attn_saver=NARX_saver()
        self.val_attn_saver=NARX_saver()

        
    def forward(self, input_seqX,y_history,y_targets, phase):
       
        features_attn_wts, input_encoded,x_weighted = self.encoder(input_seqX)
        y_preds, dec_hidden,contexts_all,temporal_attn_wts = self.decoder(input_encoded, y_history)
    
        with torch.no_grad():
            if phase=='val': 
                #print ("recording val attentions")
                self.val_attn_saver.record_data(y_targets,y_preds,features_attn_wts, input_encoded,x_weighted, dec_hidden,contexts_all,temporal_attn_wts)
            elif phase =='test':
                self.test_attn_saver.x.append(input_seqX)
                #print ("recording test attentions")
                self.test_attn_saver.record_data(y_targets,y_preds,features_attn_wts, input_encoded, x_weighted, dec_hidden,contexts_all,temporal_attn_wts)

        
        return y_preds

    def setOptimizer(self,enc_opt, dec_opt):
        self.encoder_optimizer=enc_opt
        self.decoder_optimizer=dec_opt
        
    def zeroGrad(self):
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            
    def stepOptimizer(self):
             self.encoder_optimizer.step()
             self.decoder_optimizer.step()  
    def set_phase(self,phase):
        if phase =='val' or phase =='test':
            self.encoder.eval()  
            self.decoder.eval()
             
class Encoder_Ankle(nn.Module):

    def __init__(self, model_params):
#         __init__(self, num_features, hidden_dim, batch_size, output_dim=1,
#                    num_layers=2,seq_len=1):
        super(Encoder_Ankle, self).__init__()
        self.num_features =model_params['num_features']
        self.hidden_dim = model_params['num_hid_units']
        self.batch_size = model_params['batch_size']
        self.num_layers = model_params['num_layers']
        self.seq_len= model_params['seq_length']-1
        # Define the LSTM layer
        enc_concat_net=nn.Linear((2*self.hidden_dim+self.seq_len),self.seq_len)
        non_linear=nn.Tanh()
        enc_shaper_net=nn.Linear(self.seq_len,1)

        self.enc_attn_net=nn.Sequential(enc_concat_net,non_linear,enc_shaper_net)
        self.lstm_enc = nn.LSTM(self.num_features, self.hidden_dim, self.num_layers)



    def init_hidden(self,input_batch):
        return (torch.zeros(self.num_layers, input_batch.shape[0], self.hidden_dim).to(device),
                torch.zeros(self.num_layers, input_batch.shape[0], self.hidden_dim).to(device))

    def forward(self, input_batch):
        if debugPrint: print ("encoder input", input_batch.shape)

        
        enc_hidden_all=torch.zeros([input_batch.shape[0],self.seq_len,self.hidden_dim]).to(device)
        attn_features_all=torch.zeros([input_batch.shape[0],self.seq_len,self.num_features]).to(device)
        x_weighted=torch.zeros([input_batch.shape[0],self.seq_len,self.num_features]).to(device)

        enc_hidden_t,enc_cell_t=self.init_hidden(input_batch) # num_layer, batch, hidden dim 


        for t in range(self.seq_len):
    
    
    
            x_aug=torch.cat((enc_hidden_t.repeat(self.num_features,1,1).permute(1,0,2),
                         enc_cell_t.repeat(self.num_features,1,1).permute(1,0,2),
                         input_batch.permute(0,2,1)),dim=2) # batch_size,num_feat,seq_len
    
            if debugPrint: print ("x aug shape",x_aug.shape)
            alingment_features_t=self.enc_attn_net(x_aug)
            feature_attn_weights=F.softmax(alingment_features_t.view(-1,self.num_features),dim=1) # softmax along features dim ->
            x_weighted_t=torch.mul(feature_attn_weights,input_batch[:,t,:]).unsqueeze(0) # shape it 1,batch,features
            
            _,lstm_state_t=self.lstm_enc(x_weighted_t,(enc_hidden_t,enc_cell_t)) # state.shape= (num_layers * num_directions, batch, hidden_size)
            (enc_hidden_t,enc_cell_t)=lstm_state_t
            
            enc_hidden_all[:,t,:]=enc_hidden_t  
            attn_features_all[:,t,:]=feature_attn_weights
            x_weighted[:,t,:]=x_weighted_t
        
        return  attn_features_all, enc_hidden_all,x_weighted


class Decoder_Ankle(nn.Module):
    def __init__(self, model_params):
        super(Decoder_Ankle, self).__init__()
        self.num_features =model_params['num_features']
        self.hidden_dim = model_params['num_hid_units']
        self.batch_size = model_params['batch_size']
        self.num_layers = model_params['num_layers']
        self.seq_len = model_params['seq_length']-1
        self.out_dim=1 #Ankle 
        dec_concat_net=nn.Linear((3*self.hidden_dim),self.hidden_dim)
        non_linear=nn.Tanh()
        dec_shaper_net=nn.Linear(self.hidden_dim,1)

        self.dec_attn_net=nn.Sequential(dec_concat_net,non_linear,dec_shaper_net)

        self.dec_feeder_net=nn.Linear(self.hidden_dim+1,self.out_dim)

        self.dec_final_fc=nn.Sequential(nn.Linear(2*self.hidden_dim,self.hidden_dim),nn.Linear(self.hidden_dim,self.out_dim))
        self.lstm_dec = nn.LSTM(self.out_dim, self.hidden_dim, self.num_layers)
    
    


    def init_hidden(self,input_batch):
        return (torch.zeros(self.num_layers, input_batch.shape[0], self.hidden_dim).to(device),
                torch.zeros(self.num_layers, input_batch.shape[0], self.hidden_dim).to(device))


    def forward(self, enc_hidden_all,y_history):
        

        dec_hidden_all=torch.zeros(enc_hidden_all.shape[0],self.seq_len,self.hidden_dim)    
        context_all=  torch.zeros(enc_hidden_all.shape[0],self.seq_len,self.hidden_dim)
        
        
        
        dec_hidden_t,dec_cell_t=self.init_hidden(enc_hidden_all) # num_layer, batch, hidden dim 
#        y_history=torch.cat((torch.zeros(self.batch_size,1,1).cuda(),y_history),dim=1)
        
        temporal_attn_weights_all=torch.zeros(enc_hidden_all.shape[0],self.seq_len,self.seq_len)
        
        for t in range(self.seq_len):
    
            decoder_state_aug=torch.cat((dec_hidden_t.repeat(self.seq_len,1,1).permute(1,0,2),
                                        dec_cell_t.repeat(self.seq_len,1,1).permute(1,0,2),
                                        enc_hidden_all),dim=2)
            
            alignment_temporal_t=self.dec_attn_net(decoder_state_aug)
            
            temporal_attn_weights=F.softmax(alignment_temporal_t.view(-1,self.seq_len),dim=1)
            if debugPrint: ("tempora weights shape",temporal_attn_weights.shape)
            
            context_t=torch.bmm(temporal_attn_weights.unsqueeze(1),enc_hidden_all).squeeze(1) # outs [b,1,m] ->squeezed to [b,m]
           
            context_all[:,t,:]=context_t
            
 
            y_tilde=self.dec_feeder_net(torch.cat((y_history[:,t],context_t),dim=1)).unsqueeze(0)
            
            _,dec_states_t=self.lstm_dec(y_tilde,(dec_hidden_t,dec_cell_t))
            (dec_hidden_t,dec_cell_t)=dec_states_t
    
            dec_hidden_all[:,t,:]=dec_states_t[0]
            temporal_attn_weights_all[:,t,:]=temporal_attn_weights
            
        y_hats=self.dec_final_fc(torch.cat((dec_hidden_t[0],context_t),dim=1))
    
        
        
        return y_hats, dec_hidden_all,context_all,temporal_attn_weights_all
    
         

class NoHistory_Decoder_Ankle(nn.Module):
    def __init__(self, model_params):
        super(NoHistory_Decoder_Ankle, self).__init__()
        self.num_features =model_params['num_features']
        self.hidden_dim = model_params['num_hid_units']
        self.batch_size = model_params['batch_size']
        self.num_layers = model_params['num_layers']
        self.seq_len = model_params['seq_length']-1
        self.out_dim=1 #Ankle 
        dec_concat_net=nn.Linear((3*self.hidden_dim),self.hidden_dim)
        non_linear=nn.Tanh()
        dec_shaper_net=nn.Linear(self.hidden_dim,1)

        self.dec_attn_net=nn.Sequential(dec_concat_net,non_linear,dec_shaper_net)

        self.dec_feeder_net=nn.Linear(self.hidden_dim,self.out_dim)

        self.dec_final_fc=nn.Sequential(nn.Linear(2*self.hidden_dim,self.hidden_dim),nn.Linear(self.hidden_dim,self.out_dim))
        self.lstm_dec = nn.LSTM(self.out_dim, self.hidden_dim, self.num_layers)
    
    


    def init_hidden(self,input_batch):
        return (torch.zeros(self.num_layers, input_batch.shape[0], self.hidden_dim).to(device),
                torch.zeros(self.num_layers, input_batch.shape[0], self.hidden_dim).to(device))


    def forward(self, enc_hidden_all,y_history):
        

        dec_hidden_all=torch.zeros(enc_hidden_all.shape[0],self.seq_len,self.hidden_dim)    
        context_all=  torch.zeros(enc_hidden_all.shape[0],self.seq_len,self.hidden_dim)
        
        
        
        dec_hidden_t,dec_cell_t=self.init_hidden(enc_hidden_all) # num_layer, batch, hidden dim 
#        y_history=torch.cat((torch.zeros(self.batch_size,1,1).cuda(),y_history),dim=1)
        
        temporal_attn_weights_all=torch.zeros(enc_hidden_all.shape[0],self.seq_len,self.seq_len)
        
        for t in range(self.seq_len):
    
            decoder_state_aug=torch.cat((dec_hidden_t.repeat(self.seq_len,1,1).permute(1,0,2),
                                        dec_cell_t.repeat(self.seq_len,1,1).permute(1,0,2),
                                        enc_hidden_all),dim=2)
            
            alignment_temporal_t=self.dec_attn_net(decoder_state_aug)
            
            temporal_attn_weights=F.softmax(alignment_temporal_t.view(-1,self.seq_len),dim=1)
            if debugPrint: ("tempora weights shape",temporal_attn_weights.shape)
            
            context_t=torch.bmm(temporal_attn_weights.unsqueeze(1),enc_hidden_all).squeeze(1) # outs [b,1,m] ->squeezed to [b,m]
           
            context_all[:,t,:]=context_t
            
 
            y_tilde=self.dec_feeder_net(context_t).unsqueeze(0)
            
            _,dec_states_t=self.lstm_dec(y_tilde,(dec_hidden_t,dec_cell_t))
            (dec_hidden_t,dec_cell_t)=dec_states_t
    
            dec_hidden_all[:,t,:]=dec_states_t[0]
            temporal_attn_weights_all[:,t,:]=temporal_attn_weights
            
        y_hats=self.dec_final_fc(torch.cat((dec_hidden_t[0],context_t),dim=1))
    
        
        
        return y_hats, dec_hidden_all,context_all,temporal_attn_weights_all        

  