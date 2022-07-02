#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:41:22 2019

@author: raiv
"""



from __future__ import print_function, division
import os
import torch
import pandas as pd

import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



class KinematicsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_dict):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.inputX_tensor=data_dict['x']
        self.userStats_tensor=data_dict['userStats']
        self.targetY_tensor=data_dict['y']
        
        self.IDs=data_dict['ids']
    def __len__(self):
        return len(self.inputX_tensor)

    def __getitem__(self, idx):
        
        inputSeq=self.inputX_tensor[idx]
        userStat=self.userStats_tensor[idx]
        
        thisID=self.IDs[idx]
        
        target=self.targetY_tensor[idx]
        
        return inputSeq,userStat,target

