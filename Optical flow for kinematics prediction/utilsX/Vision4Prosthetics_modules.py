import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from Lstm_Modules import LstmHistory_SeqtoOne_Module
from random import randrange
from easydict import EasyDict
##from torchvision import transforms
from ARFlow_transforms import sep_transforms
from ARFlow_utils.flow_utils import flow_to_image, resize_flow
from ARFlow_utils.torch_utils import restore_model
from models.pwclite import PWCLite
import PIL

debugPrint = 1;
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
import device_utils
device = device_utils.getCudaDevice()

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = device_utils.getCudaDeviceidx()

def decimate(tensor_input)  :
        idx = torch.arange(0, tensor_input.shape[0], step=5).to(device) 
        tensor_decimated=torch.index_select(tensor_input, 0, idx)
#        print ("input decimated",tensor_input.shape,tensor_decimated.shape)
        return tensor_decimated
    
class conv_saver():
     
    def __init__(self):
        self.batch_size=0
        self.y_hists=[]
#        self.x_frames=[]
        self.y_preds=[]
        self.y_future=[]
        
    def set_size(self,size) :
        self.batch_size=size
 
        
    def record_data (self,x_frames,y_hists,y_future,y_preds):
        
        
#        self.x_frames.append(decimate(x_frames).detach().cpu());
        if y_hists != None:self.y_hists.append(decimate(y_hists).detach().cpu())
        self.y_preds.append(decimate(y_preds).detach().cpu());
        if y_future != None:self.y_future.append(decimate(y_future).detach().cpu())
        
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
              'y_hists': cat_em_up(self.y_hists) if self.y_hists != None else 'None',
               'y_preds': cat_em_up(self.y_preds),
              'y_future': cat_em_up(self.y_future)
               }
     
        return data    
    
class Scene4Prosthetics(nn.Module):

    def __init__(self, model_params):
        
        super(Scene4Prosthetics, self).__init__()

        self.zero_like_vision = model_params['zero_like_vision']
        self.model_params = model_params
        self.num_vis_features = model_params['num_vis_features']
        self.num_features = model_params['num_features']
        self.hidden_dim = model_params['num_hid_units']
        self.batch_size = model_params['batch_size']
        self.num_layers = model_params['num_layers']
        self.seq_len= model_params['seq_length']
        self.num_frames =  self.seq_len
        self.target_len= model_params['target_len']
        self.train_noise_std=model_params['noise_std']
        self.lstm = self.get_kinematics_model(model_params)
        self.vision_features = self.get_vision_model(model_params)
        self.num_fc_layers = model_params['num_fc_layers']
        self.fc_layers =list(range(self.num_fc_layers))
        print(self.fc_layers)
        self.num_fc_features =self.hidden_dim+self.num_vis_features #model_params['num_features']                
##        self.fusionlayer1 = nn.Linear(,64)
##        self.fusionlayer2 = nn.Linear(64,16)

        taper = model_params['fc_taper']
        earlyexit_flag = False

        for i in self.fc_layers:
                if int(self.num_fc_features/(taper)**(i))<=2:
                        #assert(self.num_fc_features/(taper)**i>1),"Too many layers/ Too high Taper. Only 1 hidden unit in some layers"
                        earlyexit_flag = True
                        break
##                        setattr(self, 'fc{}'.format(i), nn.Linear(1, 1)
##                                print('Too many layers/ Too high Taper. Only 1 hidden unit in some layers')        
                else:
                                setattr(self, 'fc{}'.format(i), nn.Linear(int(self.num_fc_features/(taper)**i), int(self.num_fc_features/(taper)**(i+1))))
        final_fc_dim = int(self.num_fc_features/(taper)**(i+1)) if earlyexit_flag ==False else int(self.num_fc_features/(taper)**(i))

        self.fc_layers = self.fc_layers[0:i+1] if earlyexit_flag ==False else self.fc_layers[0:i]
        
        print(self.fc_layers)
        self.output_dim = 2 # for Sequence to One
        self.output_layer = nn.Linear(final_fc_dim,self.output_dim)
        self.saver=conv_saver()

        
    def get_vision_model(self,model_params):
              
        arch = 'resnet18'

        # load the pre-trained weights
        model_file = '%s_places365.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)

        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,self.num_vis_features)
        if model_params['backprop_vision']==False:
            for param in model.parameters():
                param.requires_grad = False
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True

##        for name, param in model.named_parameters():
##            print('Vision',name, ':', param.requires_grad)    


        return model
    
    def get_kinematics_model(self,model_params):
        model = nn.LSTM(self.num_features, self.hidden_dim, self.num_layers)

        model_dict = model.state_dict()


        
        if model_params['pretrained_LSTM'] == True:
            pretrained_model = torch.load(checkpoint_path+'bestmodefree.pt')
            pretrained_dict = pretrained_model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            model.load_state_dict(pretrained_dict)
            model.load_state_dict()
            
        if model_params['backprop_kinematics']==False:
            for param in model.parameters():
                param.requires_grad = False
                
##        for name, param in model.named_parameters():
##            print('Kinematics', name, ':', param.requires_grad)            

        return model

    def forward (self,phase,input_seq,batchFrames,y_history=None,batchY=None,batchUserStat=None)  :
         
##         print('History Size:', y_history.size())
         print('Frames Size:', batchFrames.size()) 
##         print('Target',batchY)
         input_seq=input_seq.permute(1,0,2) # seq length first, batch, input_features   
         kine_feats, self.hidden = self.lstm(input_seq)              
         #del input_seq
         #frame_id = randrange(self.seq_len)
##       print('Frame selected: ', frame_id)
         if self.zero_like_vision==True:
                 frame_feats = torch.zeros_like(self.vision_features(batchFrames))#[:,frame_id,:,:,:]))
         else:
                 frame_feats = self.vision_features(batchFrames)#[:,frame_id,:,:,:])
         #del batchFrames
        
##         print('Frame Feats Size',frame_feats.size())
##         print('Kine Feats Size',kine_feats.view(kine_feats.size(0),-1).size())
##         print('Kine Reshaped', kine_feats[-1].view(kine_feats[-1].size(0),-1).size())
         feats_cat = torch.cat((frame_feats,kine_feats[-1].view(kine_feats[-1].size(0),-1)),1)   

##         print('Cat Feat size', feats_cat.size())

         for i in self.fc_layers:
                                fc_i = getattr(self, 'fc{}'.format(i))
                                feats_cat = F.leaky_relu(fc_i(feats_cat))

##         final_feats = F.leaky_relu(self.fusionlayer2(F.leaky_relu(self.fusionlayer1(feats_cat))))
         output = self.output_layer(feats_cat)                      

##         print('Frame1 Size:', frame1_feats.size())
##         print('Frame2 Size:', frame2_feats.size())
         
##         print('feats size:', feats_cat.size())

                
         
         
         with torch.no_grad():
           if phase =='test':
                #print ("recording test attentions")
                
                
                self.saver.record_data(batchFrames,y_history,batchY,output)
                
         return  output#, torch.LongTensor(output_indices)
     
    def setOptimizer(self,optimizer,scheduler):
        self.optimizer= optimizer
        self.scheduler= scheduler
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def stepOptimizer(self,valid_loss=None):
         self.optimizer.step()

    def stepScheduler(self,valid_loss=None):
         self.scheduler.step(valid_loss)
                 
    def getOptimizer(self):
         return self.optimizer    


    def getScheduler(self):
         return self.scheduler   

                 
    
    def set_phase(self,phase):
        if phase =='val' or phase =='test':
            self.eval()  
        elif phase=='train':
            self.train()


class OpticalFlow4Prosthetics(nn.Module):

    def __init__(self, model_params):
        
        super(OpticalFlow4Prosthetics, self).__init__()

        cfg = {'model': {'upsample': False,'n_frames': 2,'reduce_dense': True},
                   'pretrained_model': 'checkpoints/Sintel/pwclite_ar.tar',
                   'test_shape': [384, 640],}        
        self.cfg = EasyDict(cfg)
        self.zero_like_vision = model_params['zero_like_vision']
        self.model_params = model_params
        self.num_vis_features = model_params['num_vis_features']
        self.num_vis_layers = model_params['num_vis_layers']
        self.num_features = model_params['num_features']
        self.hidden_dim = model_params['num_hid_units']
        self.batch_size = model_params['batch_size']
        self.num_layers = model_params['num_layers']
        self.seq_len= model_params['seq_length']
        self.num_frames =  self.seq_len
        self.target_len= model_params['target_len']
        self.train_noise_std=model_params['noise_std']
        self.vision_lstm = nn.LSTM(1920, self.num_vis_features, self.num_vis_layers)        
        self.lstm = self.get_kinematics_model(model_params)
##        self.vision_features = self.get_vision_model(model_params)
        self.num_fc_layers = model_params['num_fc_layers']
        self.fc_layers =list(range(self.num_fc_layers))
        print(self.fc_layers)
        self.num_fc_features =self.hidden_dim+self.num_vis_features #model_params['num_features']                
##        self.fusionlayer1 = nn.Linear(,64)
##        self.fusionlayer2 = nn.Linear(64,16)

        taper = model_params['fc_taper']
        earlyexit_flag = False

        for i in self.fc_layers:
                if int(self.num_fc_features/(taper)**(i))<=2:
                        #assert(self.num_fc_features/(taper)**i>1),"Too many layers/ Too high Taper. Only 1 hidden unit in some layers"
                        earlyexit_flag = True
                        break
##                        setattr(self, 'fc{}'.format(i), nn.Linear(1, 1)
##                                print('Too many layers/ Too high Taper. Only 1 hidden unit in some layers')        
                else:
                                setattr(self, 'fc{}'.format(i), nn.Linear(int(self.num_fc_features/(taper)**i), int(self.num_fc_features/(taper)**(i+1))))
        final_fc_dim = int(self.num_fc_features/(taper)**(i+1)) if earlyexit_flag ==False else int(self.num_fc_features/(taper)**(i))

        self.fc_layers = self.fc_layers[0:i+1] if earlyexit_flag ==False else self.fc_layers[0:i]
        
        print(self.fc_layers)
        self.output_dim = 2 # for Sequence to One
        self.output_layer = nn.Linear(final_fc_dim,self.output_dim)
        self.saver=conv_saver()

        
##    def get_vision_model(self,model_params):
##              
##        arch = 'PCW_lite'
##
##        # load the pre-trained weights
##
##        model = PWCLite(self.cfg.model)
##        model = restore_model(model, self.cfg.pretrained_model)
##        
##
##        if model_params['backprop_vision']==False:
##                for param in model.parameters():
##                        param.requires_grad = False
##           
####        for name, param in model.named_parameters():
####            print('Vision',name, ':', param.requires_grad)    
##
##
##        return model
    
    def get_kinematics_model(self,model_params):
        model = nn.LSTM(self.num_features, self.hidden_dim, self.num_layers)

        model_dict = model.state_dict()


        
        if model_params['pretrained_LSTM'] == True:
            pretrained_model = torch.load(checkpoint_path+'bestmodefree.pt')
            pretrained_dict = pretrained_model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            model.load_state_dict(pretrained_dict)
            model.load_state_dict()
            
        if model_params['backprop_kinematics']==False:
            for param in model.parameters():
                param.requires_grad = False
                
##        for name, param in model.named_parameters():
##            print('Kinematics', name, ':', param.requires_grad)            

        return model

    def forward (self,phase,input_seq,batchFrames,y_history=None,batchY=None,batchUserStat=None)  :
##         print('Kine IP Size:', input_seq.size()) 
##         print('History Size:', y_history.size())
##         print('Frames Size:', batchFrames.size()) 
##         print('Target',batchY)
         input_seq=input_seq.permute(1,0,2) # seq length first, batch, input_features   
         kine_feats, self.hidden = self.lstm(input_seq)              
         #del input_seq
         #frame_id = randrange(self.seq_len)
##       print('Frame selected: ', frame_id)
         if self.zero_like_vision==True:
                 frame_feats = torch.zeros_like(self.vision_features(batchFrames))#[:,frame_id,:,:,:]))
         elif self.model_params['backprop_vision']==False:        
                 batchFrames = torch.squeeze(batchFrames,dim=2)
##                 print(batchFrames.size())
                 batchFrames = batchFrames.permute(1,0,2)
                 frame_feats, self.hidden2 = self.vision_lstm(batchFrames)
         else:
                 frame_feats = self.vision_features(batchFrames)#[:,frame_id,:,:,:])
         #del batchFrames
        
##         print('Frame Feats Size',frame_feats.size())
##         print('Kine Feats Size',kine_feats.size())

         feats_cat = torch.cat((frame_feats[-1].view(kine_feats[-1].size(0),-1),kine_feats[-1].view(kine_feats[-1].size(0),-1)),1)   

##         print('Cat Feat size', feats_cat.size())

         for i in self.fc_layers:
                                fc_i = getattr(self, 'fc{}'.format(i))
                                feats_cat = F.leaky_relu(fc_i(feats_cat))

##         final_feats = F.leaky_relu(self.fusionlayer2(F.leaky_relu(self.fusionlayer1(feats_cat))))
         output = self.output_layer(feats_cat)                      

##         print('Frame1 Size:', frame1_feats.size())
##         print('Frame2 Size:', frame2_feats.size())
         
##         print('feats size:', feats_cat.size())

                
         
         
         with torch.no_grad():
           if phase =='test':
                #print ("recording test attentions")
                
                
                self.saver.record_data(batchFrames,y_history,batchY,output)
                
         return  output#, torch.LongTensor(output_indices)
     
    def setOptimizer(self,optimizer,scheduler):
        self.optimizer= optimizer
        self.scheduler= scheduler
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def stepOptimizer(self,valid_loss=None):
         self.optimizer.step()

    def stepScheduler(self,valid_loss=None):
         self.scheduler.step(valid_loss)
                 
    def getOptimizer(self):
         return self.optimizer    


    def getScheduler(self):
         return self.scheduler   

                 
    
    def set_phase(self,phase):
        if phase =='val' or phase =='test':
            self.eval()  
        elif phase=='train':
            self.train()               
        

#####################################################################
            
class OpticalFlow_ftex(nn.Module):

    def __init__(self, model_params):
        
        super(OpticalFlow_ftex, self).__init__()

        cfg = {'model': {'upsample': False,'n_frames': 2,'reduce_dense': True},
                   'pretrained_model': 'checkpoints/Sintel/pwclite_ar.tar',
                   'test_shape': [384, 640],}        
        self.cfg = EasyDict(cfg)
        self.model_params = model_params
        self.num_vis_features = model_params['num_vis_features']
        self.batch_size = model_params['batch_size']
        self.seq_len= model_params['seq_length']
        self.num_frames =  self.seq_len
        self.target_len= model_params['target_len']
        self.vision_features = self.get_vision_model(model_params)

        
    def get_vision_model(self,model_params):
              
        arch = 'PCW_lite'

        # load the pre-trained weights

        model = PWCLite(self.cfg.model)
        model = restore_model(model, self.cfg.pretrained_model)
        
        for param in model.parameters():
                param.requires_grad = False
           
##        for name, param in model.named_parameters():
##            print('Vision',name, ':', param.requires_grad)    


        return model
    


    def forward (self,phase,input_seq,batchFrames,y_history=None,batchY=None,batchUserStat=None)  :
         
##         print('History Size:', y_history.size())
         #print('Frames Size:', batchFrames.size())
                 
         
         frame_feats = self.vision_features(batchFrames)['flows_fw'][0]#[:,frame_id,:,:,:])
         #del batchFrames
        
         #print('Frame Feats Size',frame_feats.size())

                
         

                
         return  frame_feats#, torch.LongTensor(output_indices)
     

                 
    
    def set_phase(self,phase):
        if phase =='val' or phase =='test':
            self.eval()  
        elif phase=='train':
            self.train()        




class Places_ftex(nn.Module):

    def __init__(self, model_params):
        
        super(Places_ftex, self).__init__()

        self.model_params = model_params
        self.num_vis_features = model_params['num_vis_features']
        self.batch_size = model_params['batch_size']
        self.seq_len= model_params['seq_length']
        self.num_frames =  self.seq_len
        self.target_len= model_params['target_len']
        self.vision_features = self.get_vision_model(model_params)
        

        
    def get_vision_model(self,model_params):
              
        arch = 'resnet18'

        # load the pre-trained weights
        model_file = '%s_places365.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)

        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        num_ftrs = model.fc.in_features
        model.fc = nn.Identity(num_ftrs)#nn.Linear(num_ftrs,self.num_vis_features)

##        for name, param in model.named_parameters():
##            print('Vision',name, ':', param.requires_grad)    
        return model


    def forward (self,phase,input_seq,batchFrames,y_history=None,batchY=None,batchUserStat=None)  :
         
##         print('History Size:', y_history.size())
         #print('Frames Size:', batchFrames.size())
                 
         
         frame_feats = self.vision_features(batchFrames)#[:,frame_id,:,:,:])
         #del batchFrames
        
         #print('Frame Feats Size',frame_feats.size())
     
         return  frame_feats#, torch.LongTensor(output_indices)
              
    
    def set_phase(self,phase):
        if phase =='val' or phase =='test':
            self.eval()  
        elif phase=='train':
            self.train()             


