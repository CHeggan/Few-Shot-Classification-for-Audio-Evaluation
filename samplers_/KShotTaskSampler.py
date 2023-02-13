"""
File contains a N-way K-shot task sampling class. This class was originally
    designed to work with class label dataframes but has now been modified to
    work with static tensors. The hope is that this version is faster in practice.

Original plan way to modify the class to do sampling directly out of the fully 
    loaded  feature tensor but this would require very significant changes and
    would lock the option of lazy loading for our dataset class (I think).
"""

###############################################################################
# IMPORTS
###############################################################################
from random import sample
import sys
import math
import time

import torch
import numpy as np
from torch.utils.data import Sampler

###############################################################################
# N-WAY K-SHOT TASK SAMPLING CLASS
###############################################################################
class KShotTaskSampler(Sampler):
    def __init__(self, 
                dataset,
                batch_size,
                n_way,
                k_shot,
                q_queries,
                num_tasks):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.num_tasks = num_tasks

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
        self.label_dicts = {}
        for class_name in self.dataset.unique_classes:

            label_mask = np.isin(self.dataset.labels, class_name)
            samples = self.idx_array[label_mask]

            if len(samples) < (self.k_shot + self.q_queries):
                print(class_name, len(samples), self.k_shot + self.q_queries)
                print(f'Issue with num samples in class: {class_name}. Removing from class selection')
                self.dataset.unique_classes = np.delete(self.dataset.unique_classes, np.where(self.dataset.unique_classes == class_name))

            self.label_dicts[class_name] = samples


    def __len__(self):
        return self.num_batches



    def __iter__1(self):
        # We iterate over the total number of batches first. This is because at
        #   the end of any given batch generation we yield it to the sampler. When
        #   total batches run out it will restart much like a dataloader setup
        for idy in range(self.num_batches):
            batch = []
            for task_idx in range(self.batch_size):

                task_classes = np.random.choice(self.dataset.unique_classes,
                    size=self.n_way,
                    replace=False)

                # Finds the relevant class samples
                label_mask = np.isin(self.dataset.labels, task_classes)

                labels = self.dataset.labels[label_mask]
                end_part = time.time()
                idx_array = self.idx_array[label_mask]

                all_supports = []
                all_queries = []


                class_start = time.time()
                for n in task_classes:
                    class_samples = idx_array[labels == n]

                    # By sampling k-shot and queries at same time, we can enforce no dupes
                    sup_quer = np.random.choice(class_samples,
                        size=self.k_shot + self.q_queries,
                        replace=False)

                    n_supports = sup_quer[:self.k_shot]
                    n_queries = sup_quer[self.k_shot:(self.k_shot + self.q_queries)]            

                    all_supports.append( n_supports )
                    all_queries.append( n_queries )

                all_supports = np.concatenate(all_supports)
                all_queries = np.concatenate(all_queries)

                all = np.concatenate((all_supports, all_queries))
                # for num in all:
                #     batch.append()
                batch.append(all)

            # Yield pauses the function saving its states and later continues from there
            yield np.concatenate(batch)

    def shuffle(self, arr):
        np.random.shuffle(arr)
        return arr

    def __iter__(self):
        # We iterate over the total number of batches first. This is because at
        #   the end of any given batch generation we yield it to the sampler. When
        #   total batches run out it will restart much like a dataloader setup
        for _ in range(self.num_batches):

            # Instead of iterating over actual tasks in a batch, we do batch tensors
            # Generate all batches of classes 
            class_selections = self.random_choice_noreplace2(
                self.dataset.unique_classes, self.n_way, self.batch_size)

            class_selections = np.reshape(class_selections, (self.batch_size*self.n_way, -1))
            class_selections = class_selections.squeeze()
            
            sample_arr = [self.shuffle(self.label_dicts[x])[:self.k_shot + self.q_queries] for x in class_selections]
            sample_arr = np.array(sample_arr)
            sample_arr = np.reshape(sample_arr, (self.batch_size, self.n_way, -1))

            support_arr = sample_arr[:, :, :self.k_shot]
            query_arr = sample_arr[:, :, self.k_shot:(self.k_shot) + self.q_queries]
            support_arr = np.reshape(support_arr, (self.batch_size, -1))
            query_arr = np.reshape(query_arr, (self.batch_size, -1))

            sample_arr = np.concatenate((support_arr, query_arr), axis=1)

            flat = sample_arr.flatten()

            yield flat


    def random_choice_noreplace2(self, l, n_sample, num_draw):
        '''
        l: 1-D array or list
        n_sample: sample size for each draw
        num_draw: number of draws

        Intuition: Randomly generate numbers, get the index of the smallest n_sample number for each row.
        '''
        l = np.array(l)
        return l[np.argpartition(np.random.rand(num_draw,len(l)), n_sample-1,axis=-1)[:,:n_sample]]