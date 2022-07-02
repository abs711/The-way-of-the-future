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
        self.y_hists=[]
#        self.x_frames=[]
        self.y_preds=[]
        self.y_future=[]
        
    def set_size(self,size) :
        self.batch_size=size
 
        
    def record_data (self,x_frames,y_hists,y_future,y_preds):
        
        
#        self.x_frames.append(decimate(x_frames).detach().cpu());
        self.y_hists.append(decimate(y_hists).detach().cpu());
        self.y_preds.append(decimate(y_preds).detach().cpu());
        self.y_future.append(decimate(y_future).detach().cpu())
        
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
              'y_hists': cat_em_up(self.y_hists),
               'y_preds': cat_em_up(self.y_preds),
              'y_future': cat_em_up(self.y_future)
               }
     
        return data    
    
class PretrainConv2Action(nn.Module):

    def __init__(self, model_params):
        
        super(PretrainConv2Action, self).__init__()

        assert model_params['seq_length'] == 2, "PretrainConv2Action supports seq-len=2."
        assert model_params['target_len'] == 1, "target length not correct; set it to 1."
        self.model_params = model_params
        self.num_classes = 8
        self.imus = list(range(6))
        self.num_features =512#model_params['num_features']
        self.batch_size = model_params['batch_size']
        self.num_fc_layers = model_params['num_fc_layers']
        self.fc_layers =list(range(self.num_fc_layers))
        self.seq_len= model_params['seq_length']
        self.num_frames =  self.seq_len
        self.target_len= model_params['target_len']
        taper = model_params['fc_taper']
        
        for i in self.fc_layers:
                if int(self.num_features/(taper)**i)<=1:
                        assert(self.num_features/(taper)**i>1),"Too many layers/ Too high Taper. Only 1 hidden unit in some layers"

##                        setattr(self, 'fc{}'.format(i), nn.Linear(1, 1)
##                                print('Too many layers/ Too high Taper. Only 1 hidden unit in some layers')        
                else:
                                setattr(self, 'fc{}'.format(i), nn.Linear(int(self.num_features/(taper)**i), int(self.num_features/(taper)**(i+1))))
        
        for i in self.imus:
                                setattr(self, 'imu{}'.format(i), nn.Linear(self.num_frames * int(self.num_features/(taper)**self.num_fc_layers), self.num_classes))
                
        self.dropout = nn.Dropout(p=model_params['dropout'])
        self.resnet_features = self.get_vision_model(model_params)                
        self.saver=conv_saver()
    def get_vision_model(self,model_params):
        
             
         model = models.resnet18(pretrained=True)
         if model_params['pretrainC2A_w_Dropout'] == False:
                 num_ftrs = model.fc.in_features        
                 model.fc = nn.Linear(num_ftrs, num_ftrs) #self.num_vision_classes)
         elif model_params['pretrainC2A_w_Dropout'] == True:
                 num_ftrs = model.fc.in_features        
                 model.fc = nn.Identity(num_ftrs, num_ftrs) #self.num_vision_classes)
                  
         return model
     
    def forward (self,input_seq,y_history,batchY,batchUserStat,batchFrames,phase)  :
         
         assert(self.seq_len == 2),"Siamese: 2 frames only"
##         print('History Size:', y_history.size())
##         print('Frames Size:', batchFrames.size()) 
##         print('Target',batchY)         
         frame1_feats = self.resnet_features(batchFrames[:,0,:,:,:])
         frame1_feats = frame1_feats.view(frame1_feats.size(0), -1)

         for i in self.fc_layers:
                                fc_i = getattr(self, 'fc{}'.format(i))
                                frame1_feats = F.relu(fc_i(frame1_feats))


         frame2_feats = self.resnet_features(batchFrames[:,1,:,:,:])
         frame2_feats = frame2_feats.view(frame2_feats.size(0), -1)

         for i in self.fc_layers:
                                fc_i = getattr(self, 'fc{}'.format(i))
                                frame2_feats = F.relu(fc_i(frame2_feats))

         if self.model_params['pretrainC2A_w_Dropout'] == True:
                 
                 frame1_feats = self.dropout(frame1_feats)
       
                 frame2_feats = self.dropout(frame2_feats)                
         
         feats_cat = torch.cat((frame1_feats,frame2_feats),1)
         output_indices = list(range(y_history.size(1) - self.target_len, y_history.size(1)))

##         print('Frame1 Size:', frame1_feats.size())
##         print('Frame2 Size:', frame2_feats.size())
         
##         print('feats size:', feats_cat.size())
         imu_out = []
         for i in self.imus:
                 imu_i = getattr(self, 'imu{}'.format(i))
                 imu_out.append(imu_i(feats_cat))
##         print(imu_out)   
                
         
         
         with torch.no_grad():
           if phase =='test':
                #print ("recording test attentions")
                
                
                self.saver.record_data(batchFrames,y_history,batchY,torch.stack(imu_out,dim=1).unsqueeze(1))
                
         return  torch.stack(imu_out,dim=1).unsqueeze(1)#, torch.LongTensor(output_indices)
     
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
        
