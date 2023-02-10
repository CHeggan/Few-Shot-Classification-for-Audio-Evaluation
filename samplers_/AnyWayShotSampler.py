"""
File contains a any-way any-shot task sampling class. This class was originally
    designed to work with class label dataframes but has now been modified to
    work with static tensors. The hope is that this version is faster in practice.
"""

###############################################################################
# IMPORTS
###############################################################################
import sys
import math
import time
import random

import torch
import numpy as np
from torch.utils.data import Sampler

###############################################################################
# ANY-WAY ANY-SHOT TASK SAMPLING CLASS
###############################################################################
class AnyWayShotSampler(Sampler):
    def __init__(self, 
                dataset,
                batch_size,
                n_way,
                k_shot,
                q_queries,
                num_tasks):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.q_queries = q_queries
        self.num_tasks = num_tasks

        # Collection of randomly selected ns and ks
        self.n_ways = n_way
        self.k_shots = k_shot


        # An idx array that we can use to track masked files/labels etc
        self.idx_array = np.arange(0, dataset.labels.shape[-1], 1)

        # Number of batches requires is the total tasks divided by batch size. For
        #   ease of sampling, we enforce exact divisibility
        if num_tasks % batch_size != 0:
            raise ValueError(f'Please make sure the total number of test tasks is \
                exactly divisible by the batch size. Right now the division returns \
                    {round(num_tasks/batch_size, 2)}')

        else:
            self.num_batches = math.ceil(num_tasks / batch_size)

        # Generate a list of tensors that correspond with unique classes
        # Generate a full dictionary (class_name, relevant indices) for the dataset
        all_num_samples = []
        self.label_dicts = {}
        for class_name in self.dataset.unique_classes:
            label_mask = np.isin(self.dataset.labels, class_name)
            samples = self.idx_array[label_mask]
            self.label_dicts[class_name] = samples

            all_num_samples.append(len(samples))


    def __len__(self):
        return self.num_batches


    def __iter__(self):
        # We iterate over the total number of batches first. This is because at
        #   the end of any given batch generation we yield it to the sampler. When
        #   total batches run out it will restart much like a dataloader setup
        for idy in range(self.num_tasks):
            batch = []

            n_way = self.n_ways[idy]
            k_shot = self.k_shots[idy]

            for task_idx in range(self.batch_size):

                task_classes = np.random.choice(self.dataset.unique_classes,
                    size=n_way,
                    replace=False)

                # Finds the relevant class samples
                label_mask = np.isin(self.dataset.labels, task_classes)

                labels = self.dataset.labels[label_mask]
                idx_array = self.idx_array[label_mask]

                all_supports = []
                all_queries = []

                for n in task_classes:
                    class_samples = idx_array[labels == n]

                    # By sampling k-shot and queries at same time, we can enforce no dupes
                    sup_quer = np.random.choice(class_samples,
                        size=k_shot + self.q_queries,
                        replace=False)

                    n_supports = sup_quer[:k_shot]

                    n_queries = sup_quer[k_shot:(k_shot + self.q_queries)]            

                    all_supports.append( n_supports )
                    all_queries.append( n_queries )

                all_supports = np.concatenate(all_supports)
                all_queries = np.concatenate(all_queries)

                all = np.concatenate((all_supports, all_queries))
                batch.append(all)

            # Yield pauses the function saving its states and later continues from there
            yield np.concatenate(batch)