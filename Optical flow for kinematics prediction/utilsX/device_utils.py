# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 22:46:21 2020

@author: vijet
"""

import os
import torch

cuda_device=None
USE_CUDA = torch.cuda.is_available()

def setCudaDevice(cuda_idx):
    global cuda_device 
    global cuda_device_str
    cuda_device_str='cuda:'+str(cuda_idx)
  
    cuda_device = torch.device(cuda_device_str if torch.cuda.is_available() else 'cpu')     

    print ("cuda_device set to",cuda_device)
    
def getCudaDevice():
    global cuda_device
    return cuda_device

def getCudaDeviceidx():
    global cuda_device
    return cuda_device_str.replace('cuda:','')

