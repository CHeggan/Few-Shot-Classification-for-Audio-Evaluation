"""
Dataset which simply wraps around a fully loaded set of either samples or 
    features and labels. In practice it acts similarity to the basic Dataset 
    class from pyTorch directly but can be retrofitted more easily. In simple 
    applications, can be swapped with the basic torch version 
"""
################################################################################
# IMPORTS
################################################################################
import os

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

################################################################################
# PATH FINDER DATASET CLASS
################################################################################
class FullSetWrapper(Dataset):
    def __init__(self, 
        full_set,
        labels):

        self.full_set = full_set
        self.labels = labels

        # Get an array of the unique numeric labels included in this set
        self.unique_classes = torch.unique(self.labels).cpu().numpy()

    def __getitem__(self, item):
        return self.full_set[item], self.labels[item]

    def __len__(self):
        return self.full_set.shape[0]


