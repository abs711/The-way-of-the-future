# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 21:53:19 2020

@author: vijet
"""

import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch Main file')
parser.add_argument('--device', type=str, default='cuda:0', metavar='N',
                    help='cuda device id default 0')
args = parser.parse_args()
if  torch.cuda.is_available():
    model_device = torch.device(args.device)
    print ("cuda device",model_device)
    
else:
    
    model_device = torch.device('cpu')
    print ("device",args.device )
