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
from torch.utils.data import Dataset

################################################################################
# PATH FINDER DATASET CLASS
################################################################################
class PathFinderSet(Dataset):
    def __init__(self, 
        classes,
        class_root_dir,
        ext='.pt'):

        self.ext = ext
        self.classes = np.array(classes, dtype='U256')
        self.num_classes = len(self.classes)

        self.class_paths = np.char.add(class_root_dir + '/', self.classes)

        # We provide class paths to samples and create a file path and class list
        self.file_paths, self.str_labels, self.int_labels = self.find_file_paths(self.class_paths, self.ext)

        # Get an array of the unique class labels included in this set
        #print(np.array(self.labels))
        self.str_unique_classes = np.unique(np.array(self.str_labels, dtype='U256'))


    def __getitem__(self, item):
        sample_path = self.file_paths[item]
        label = self.int_labels[item]

        if self.ext == '.pt':
            sample = torch.load(sample_path, map_location=torch.device('cpu'))
        elif self.ext == '.npy':
            sample = np.load(sample_path, allow_pickle=True)
            sample = torch.from_numpy(sample)

        sample.requires_grad = False

        # # Combines multi slices into 1. This isnt idea but for sake of uniformity,
        # #   simplicity and speed, it worth it for the small info tradeoff
        # if sample.ndim > 1:
        #     sample = sample.mean(dim=0)

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



