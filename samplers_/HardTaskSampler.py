"""
File contains a hard/easy task sampler. Algorithm for generating such tasks is 
    based off of this work https://arxiv.org/pdf/2110.13953.pdf
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
# HARD/EASY TASK SAMPLING CLASS
###############################################################################
class HardTaskSampler(Sampler):
    def __init__(self, 
                dataset,
                batch_size,
                n_way,
                k_shot,
                q_queries,
                classifier,
                diff, #easy/hard
                device,
                num_tasks):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.q_queries = q_queries
        self.num_tasks = num_tasks

        self.n_way = n_way
        self.k_shot = k_shot

        self.diff = diff
        self.device = device
        self.classifier = classifier

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

            if len(samples) < (self.k_shot + self.q_queries):
                print(f'Issue with num samples in class: {class_name}. Removing from class selection')
                self.dataset.unique_classes = np.delete(self.dataset.unique_classes, np.where(self.dataset.unique_classes == class_name))
                
            all_num_samples.append(len(samples))


        self.max_k_shot = min( min(all_num_samples) - self.q_queries, 15)
        self.max_n_way = min(len(self.dataset.unique_classes), 15)



    def __len__(self):
        return self.num_batches


    def __iter__(self):
        # We iterate over the total number of batches first. This is because at
        #   the end of any given batch generation we yield it to the sampler. When
        #   total batches run out it will restart much like a dataloader setup
        for idy in range(self.num_tasks):
            batch = []

            for task_idx in range(self.batch_size):

                task_classes = np.random.choice(self.dataset.unique_classes,
                    size=self.n_way,
                    replace=False)

                # Finds the relevant class samples
                label_mask = np.isin(self.dataset.labels, task_classes)

                labels = self.dataset.labels[label_mask]
                idx_array = self.idx_array[label_mask]

                all_supports = []
                all_queries = []

                remaining_ks = []

                # Generate initial support set and fixed query set
                for n in task_classes:
                    class_samples = idx_array[labels == n]

                    # By sampling k-shot and queries at same time, we can enforce no dupes
                    sup_quer = np.random.choice(class_samples,
                        size=self.k_shot + self.q_queries,
                        replace=False)

                    n_supports = sup_quer[:self.k_shot]

                    n_queries = sup_quer[self.k_shot:(self.k_shot + self.q_queries)]            

                    # Find out and store what samples are left
                    samples_left = [ele for ele in class_samples if ele not in sup_quer]
                    remaining_ks.append(samples_left)

                    all_supports.append( n_supports )
                    all_queries.append( n_queries )

                all_supports = np.concatenate(all_supports)
                all_queries = np.concatenate(all_queries)

                # Load in actual query set
                query_set_feats = torch.zeros(size=(len(all_queries), 64))
                query_set_labels = []
                for q_idx, actual_index in enumerate(all_queries):
                    feats, label = self.dataset.__getitem__(actual_index)
                    query_set_feats[q_idx] = feats
                    query_set_labels.append(label)


                x_quer, y_quer = self.batch_fn( (query_set_feats, query_set_labels), format='query')

                sup_set_feats = torch.zeros(size=(len(all_supports), 64))
                sup_set_labels = []
                for s_idx, actual_index in enumerate(all_supports):
                    feats, label = self.dataset.__getitem__(actual_index)
                    sup_set_feats[s_idx] = feats
                    sup_set_labels.append(label)

                x_sup, y_sup = self.batch_fn( (sup_set_feats, sup_set_labels), format='support')

                # Iterate over classes and supports within those classes
                for idn, n in enumerate(task_classes):
                    relevant_ks = all_supports[idn*self.k_shot:(idn+1)*self.k_shot]
                    for idk, k in enumerate(relevant_ks):
                        # What is the support set idx that we wish to replace
                        within_support_idx = np.where(k==all_supports)[0][0]

                        x_sup, ks_left_over, new_best_idx = self.search_over_k(replace_idx=within_support_idx, 
                            remaining_ks=remaining_ks[idn], 
                            og_k_sample=all_supports[within_support_idx],
                            s_feats=x_sup,
                            s_labels=y_sup,
                            q_feats=x_quer,
                            q_labels=y_quer)

                        remaining_ks[idn] = ks_left_over

                        all_supports[within_support_idx] = new_best_idx

                all = np.concatenate((all_supports, all_queries))
                batch.append(all)

            # Yield pauses the function saving its states and later continues from there
            yield np.concatenate(batch)


    def search_over_k(self, replace_idx, remaining_ks, og_k_sample, s_feats, s_labels, q_feats, q_labels):
        # Generate base accuracy

        full_task = torch.cat([s_feats, q_feats], dim=1)
        all_labels = torch.cat([s_labels, q_labels], dim=1)

        if self.classifier.name == 'linear':
            single_acc, task_support_loss = self.classifier.fixed_length(full_task, all_labels)
            single_acc = single_acc[0].item()

        elif self.classifier.name == 'NCC':
            single_acc = self.classifier.fixed_length(full_task, all_labels)[0].item()

        random.shuffle(remaining_ks)

        best_new_k_index = og_k_sample
        acc_test = single_acc
        for idz, new_k_index in enumerate(remaining_ks):
            new_sample, label = self.dataset.__getitem__(new_k_index)
            s_feats[:, replace_idx] = new_sample

            full_task = torch.cat([s_feats, q_feats], dim=1)

            if self.classifier.name == 'linear':
                single_acc, task_support_loss = self.classifier.fixed_length(full_task, all_labels)
                single_acc = single_acc[0].item()

            elif self.classifier.name == 'NCC':
                single_acc = self.classifier.fixed_length(full_task, all_labels)[0].item()

            if self.diff == 'easy':
                if np.greater(single_acc, acc_test):
                    acc_test = single_acc
                    best_new_k_index = new_k_index

            elif self.diff == 'hard':
                if np.less(single_acc, acc_test):
                    acc_test = single_acc
                    best_new_k_index = new_k_index

        new_sample, label = self.dataset.__getitem__(best_new_k_index)
        s_feats[:, replace_idx] = new_sample

        # Add back in the og sample index
        remaining_ks.append(og_k_sample)
        # Remove the best selected sample for next iteration 
        remaining_ks.remove(best_new_k_index)
        # This setup naturally deals with the case where the best k index is the original

        return s_feats, remaining_ks, best_new_k_index


    def batch_fn(self, batch, format):
        x, y = batch

        x = x.squeeze()

        if format == 'query':
            middle_term = self.n_way*self.q_queries
        elif format == 'support':
            middle_term = self.n_way*self.k_shot

        if x.ndim > 2:
            x = x.reshape(self.batch_size, (middle_term),
                                x.shape[-2], x.shape[-1])
        elif x.ndim == 2:
            x = x.reshape(self.batch_size, (middle_term),
                     x.shape[-1])

        x = x.float().to(self.device)

        if format == 'query':
            y = torch.floor( torch.arange(0, self.n_way, 1/self.q_queries) )
        elif format == 'support':
            y = torch.floor( torch.arange(0, self.n_way, 1/self.k_shot) )

        # Creates a batch dimension and then repeats across it
        y = y.unsqueeze(0).repeat(self.batch_size, 1)

        y = y.long().to(self.device)


        return x, y