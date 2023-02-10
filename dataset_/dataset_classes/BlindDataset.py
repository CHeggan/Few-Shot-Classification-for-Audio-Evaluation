################################################################################
# IMPORTS
################################################################################
import os
import torch
import torchaudio
import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset
from dataset_.dataset_utils import per_sample_scale, nothing_func, given_stats_scale

################################################################################
# BLIND DATASET CLASS (DONT HAVE ANY ADDITIONAL INFORMATION OUTSIDE OF STRUCTURE)
################################################################################
class Blind_Dataset(Dataset):
    """Torch dataset object for wrapping around and iterating over few-shot datasets
        which are often nested class-wise.  This dataset aims to be the 'blind'
        version which makes little assumptions about the data outside of its
        nested structures. 

    Args:
        Dataset (class): Inherits the pytorch dataset object
    """
    def __init__(self, data_path, norm, ext, stats=None):
 
        self.ext = ext
        self.norm = norm
        self.data_path = data_path

        self.expected_files = self.find_file_paths(data_path, ext)

        # Set the normalisation function needed, default to not need stat path
        self.norm_func = self.set_norm_func(norm, stats=stats)


    def set_norm_func(self, norm, stats):
        """Grabs the relevant normalisation function to use when getting data

        Args:
            norm (str): The normalisation types to use
            stats_file (str): Path to the global norm stats of data being used

        Raises:
            ValueError: If normalisation type not recognised, inform user

        Returns:
            function: The normalisation function to call over incoming data
        """
        if norm == 'l2':
            norm_func = preprocessing.normalize

        elif norm == 'None':
            norm_func = nothing_func

        elif norm == 'sample':
            norm_func = per_sample_scale

        elif norm == 'global':
            self.mu = stats[0]
            self.sigma = stats[1]
            norm_func = given_stats_scale
        else:
            raise ValueError('Passed norm type unsupported')

        return norm_func

    def __getitem__(self, item):
        if self.ext == '.npy':
            sample = np.load(self.expected_files[item], allow_pickle=True)
            sample = torch.from_numpy(sample)
        elif self.ext == '.pt':
            sample, label = torch.load(self.expected_files[item])
        elif self.ext == '.wav':
            sample, org_sr = torchaudio.load(self.expected_files[item])
            sample = torchaudio.functional.resample(sample, orig_freq=org_sr, new_freq=16000)


        # Deals with normalisation of various types
        if self.norm in ['global']:
            sample = self.norm_func(sample, self.mu, self.sigma)
        else:
            sample = self.norm_func(sample)

        return sample, self.expected_files[item].split('.')[0] #(dont want to take extension with us)
    

    def __len__(self):
        return len(self.expected_files)

    
    def find_file_paths(self, data_path, ext):
        file_paths = []
        for root, folders, files in os.walk(data_path):
            for f in files:
                if f.endswith(ext):
                    path = os.path.join(root, f)
                    path = os.path.normpath(path)
                    file_paths.append(path)
        
        if len(file_paths) == 0:
            raise ValueError(f'No files found with extension {ext} in directory {data_path}')

        return file_paths











