"""
File contains a basic but modular feature loading class. Its most basic function
    is to iterate over the class folders found in the primary path and store
    both string based labels and sample paths
"""
################################################################################
# IMPORTS
################################################################################
import os

import torch
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
from torch.utils.data import Dataset

from dataset_.dataset_utils import per_sample_scale, nothing_func, given_stats_scale, enforce_length, split_tensor_and_enforce

################################################################################
# PATH FINDER DATASET CLASS
################################################################################
class PathFinderSet(Dataset):
    def __init__(self, 
        classes,
        class_root_dir,
        norm,
        stats,
        variable,
        sample_rep_length,
        ext='.pt'):

        self.ext = ext
        self.norm = norm
        self.classes = np.array(classes, dtype='U256')
        self.num_classes = len(self.classes)

        self.variable = variable 
        
        # Set the fixed representation length of 
        self.sample_rep_length = sample_rep_length

        self.class_paths = np.char.add(class_root_dir + '/', self.classes)

        # We provide class paths to samples and create a file path and class list
        self.file_paths, self.str_labels, self.int_labels = self.find_file_paths(self.class_paths, self.ext)

        # Get an array of the unique class labels included in this set
        #print(np.array(self.labels))
        self.str_unique_classes = np.unique(np.array(self.str_labels, dtype='U256'))

        self.labels = self.int_labels
        self.unique_classes = torch.unique(self.labels).cpu().numpy()

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
        sample_path = self.file_paths[item]
        label = self.int_labels[item]

        if self.ext == '.pt':
            sample = torch.load(sample_path, map_location=torch.device('cpu'))
        elif self.ext == '.npy':
            sample = np.load(sample_path, allow_pickle=True)
            sample = torch.from_numpy(sample)

        sample.requires_grad = False

        # If we give sample length == 0, we return the whole sample
        if self.sample_rep_length == 0:
            sample = torch.flatten(sample).squeeze()

        # Otherwise we sub select some section of it
        else:
            # Flattens sample so that it can be re-split into the representation length we want
            # To make this work, all datasets get passed back as if they were variable (this needs
            # to be accounted for in config for datasets)
            sample = torch.flatten(sample).squeeze()
            sample = split_tensor_and_enforce(sample, self.sample_rep_length)

            # Make this work for samples that are not of same length and from a diff dataset, i.e NSYNTH
            #   which is processed at 4s instead of the other at 5
            # sample = enforce_length(sample, 80000)

            # # Combines multi slices into 1. This isnt idea but for sake of uniformity,
            # #   simplicity and speed, it worth it for the small info tradeoff
            # if sample.ndim > 1:
            #     sample = sample.mean(dim=0)

        # # Deals with normalisation of various types
        # if self.norm in ['global']:
        #     sample = self.norm_func(sample, self.mu, self.sigma)

        # else:
        #     sample = self.norm_func(sample)


        # Some samples are coming in all 0s, when normalised they change to nans
        # We catch that here and convert nans to 0s
        if np.isnan(np.min(sample.numpy())):
            np.nan_to_num(sample, copy=False, nan=0)


        return sample, label


    def __len__(self):
        return len(self.file_paths)


    def find_file_paths(self, class_paths, ext):
        file_paths = []
        class_labels = []
        numeric_labels = []

        for idx, class_folder in enumerate(class_paths):
            all_files = os.listdir(class_folder)
            num_files = len(all_files)
            # Grabs name of class, we need this to generate labels
            class_name = class_folder.split(os.sep)[-1]
            # Generate a list of matching class names for our input samples
            # This is potentially helpful for few-shot experiments
            class_names = [class_name] * num_files

            all_files = np.array(all_files, dtype='U256')
            class_folder = class_folder + os.sep
            all_files = np.char.add(class_folder, all_files)

            # Create an integer set of labels
            num_labels = torch.full(size=(len(all_files),), fill_value=idx)

            file_paths.append( all_files )
            class_labels.append( class_names )
            numeric_labels.append(num_labels)


        file_paths = [item for sublist in file_paths for item in sublist]
        class_labels = [item for sublist in class_labels for item in sublist]

        file_paths = np.array(file_paths)
        class_labels = np.array(class_labels)
        numeric_labels = torch.concat(numeric_labels)

        return file_paths, class_labels, numeric_labels



