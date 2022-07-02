

from __future__ import print_function, division
import os
import torch
import pandas as pd
import time
import numpy as np
import datamanager_v7 as dm
import random
from torch.utils.data import Dataset, DataLoader
##from torchvision import transforms, utils
from PIL import Image
import imageio

import runner_v8 as runner

#class ToTensor(object):
#    """Convert ndarrays in sample to Tensors."""
#
#    def __call__(self, image):
##        image, landmarks = sample['image'], sample['landmarks']
#
#        # swap color axis because
#        # numpy image: H x W x C
#        # torch image: C X H X W
#        print ("transpose imag")
#        image = image.transpose((2, 0, 1))
#        return {'image': torch.from_numpy(image)}

class VisionKinematicsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_dict, uid_manager,transforms=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.sanity_check,self.substring_list,self.num_scenesamples,self.model_type,self.backprop_vision = runner.GetTrainingOpts()
        
        self.inputX_tensor=data_dict['x']
        self.userStats_tensor=data_dict['userStats']
        self.targetY_tensor=data_dict['y']
        self.frameNums_tensor=data_dict['frameNums']
        self.IDs=data_dict['ids']
        
        # to do - extract from config perhaps and not hardcoded to axis dim
        self.seq_length=self.inputX_tensor.shape[1]
        
        self.frames_stepsize=int(dm.get_field_from_config('frame_subsample_factor'))
##        self.range_images=[]
##        for bins in [range(i, i+self.frames_stepsize) for i in range(0, self.seq_length, self.frames_stepsize)]:
##            self.range_images.append(bins)
        
        #self.frame_sample_indices=torch.LongTensor(frame_sample_idx_list)
        self.num_chars_frameName=8;
        
        self.root_datadir = dm.get_data_dir()
        
        self.image_width=256 
        self.transforms = transforms
        self.uid_manager = uid_manager
    def __len__(self):
        return len(self.inputX_tensor)

    def __getitem__(self, idx):
        #tic_frame=time.clock()
        use_pretrained = not self.backprop_vision
        inputSeq=self.inputX_tensor[idx]
        userStat=self.userStats_tensor[idx]
##        print(self.IDs[idx])
        thisID= self.uid_manager.getUID(self.IDs[idx])#dm.index2UID(self.IDs[idx])
        this_frameNums=self.frameNums_tensor[idx]  # one sample sequence of all image numbers
        
        # convert string
        #op-00000001.jpg
        frames=[]
        

        shortlist = [len(this_frameNums)//2] if self.num_scenesamples==1 else random.sample(range(len(this_frameNums)), self.num_scenesamples)
        #print(shortlist)
##        for current_range in self.range_images:
        idx_img_list = shortlist if 'Scene' in self.model_type else range(len(this_frameNums))
##        print("idx_img_list: ",idx_img_list)
        for idx_img in idx_img_list:#current_range :
            #print('image_idx',idx_img)
            #tic_frame=time.clock()
            frameNum=this_frameNums[idx_img].item() # single image number 
##            print ("trying frame num",frameNum)
##            try: 
            frameName=thisID + '/frames/'+'op-'+str(frameNum).zfill(self.num_chars_frameName)+'.jpg'
            frameName=os.path.join(self.root_datadir,frameName)

            

            if '_ftex' in self.model_type:
                use_pretrained = False
            
            
            if 'OpticalFlow' in self.model_type and use_pretrained == False:
                frame = imageio.imread(frameName).astype(np.float32)
            elif 'OpticalFlow' in self.model_type and use_pretrained == True:
                frame = torch.load(frameName.replace('frames','motion_feats').replace('.jpg','')+'.pt')
##            elif 'Scene' in self.model_type and use_pretrained == True:
##                frame = torch.load(frameName.replace('frames','scene_feats').replace('.jpg','')+'.pt')
            else:
                frame = Image.open(frameName)
                
            
                
            if self.transforms and use_pretrained == False:
                trans_frame = self.transforms(frame)
                frames.append(trans_frame) #check order
            else:
                frame = frame.view(1,-1)
                frames.append(frame)
                #print (" found an image in this range",current_range, frameName)
##                break ;# this continues to look in nex range of images/ outer loop
##            except:
##                continue; # if image doesnt exits, continue inner loop 
            #toc_frame=time.clock()
            #print("Frame Time",toc_frame-tic_frame)
                
            
        frames_tensor=frames[0] if self.num_scenesamples==1 and 'Scene' in self.model_type else torch.stack(frames)
                           
        if self.model_type == 'OpticalFlow_ftex':
            frames_tensor = torch.cat(frames,0)
            
               
        target=self.targetY_tensor[idx]
        #toc_frame=time.clock()
        #print("Frame Time",toc_frame-tic_frame)
        if '_ftex' in self.model_type:
            return inputSeq,userStat,frames_tensor,target,frameName
        else:
            return inputSeq,userStat,frames_tensor,target
