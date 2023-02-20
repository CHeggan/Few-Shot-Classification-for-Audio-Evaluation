"""
An largely enclosed few-shot classification class with accompanying methods. Sets 
    up the few-shot classification task loader (i.e easy/avg hard) and loops 
    over for avg classification results. Also deals with the majority of setup 
    around classifiers, and task samplers
"""
import sys
import torch
import numpy as np
from tqdm import tqdm
from classifiers_.NearestCentroid import NCC
from classifiers_.sklLinear import skLinear
from dataset_.dataset_classes.FastDataLoader import FastDataLoader
from samplers_.KShotTaskSampler import KShotTaskSampler
from samplers_.HardTaskSampler import HardTaskSampler
from samplers_.AnyWayShotSampler import AnyWayShotSampler
from dataset_.prep_batch_fns import prep_batch_fixed, prep_var_eval_1d
from dataset_.dataset_utils import variable_collate

"""
takes:
    - classifier
    - n
    - k
    - q
    - num tasks

    - fixed or variable length (can optimise if fixed)
    - dataset
    - batch function
    - difficulty
    - variable

to do:
    - create relevant dataloader:
        - dataset
        - num_workers
        - 
    - instantiates the classifier 
    - creates the task loader
    - loop over generated tasks
    - obtain metrics over tasks
"""

class FewShotClassification():
    def __init__(self, dataset, params, model_fc_out, device, variable, hardness, additional_fn):

        self.device = device
        self.params = params
        self.variable = variable
        self.additional_fn = additional_fn

        self.classifier = self.setup_classifier(params,
            model_fc_out=model_fc_out, 
            device=device)

        self.batch_fn = self.setup_batching(params=params, 
            variable=variable,
            device=device)

        self.task_loader = self.setup_taskloader(params=params, 
            dataset=dataset, 
            hardness=hardness, 
            classifier=self.classifier,
            variable=variable,
            device=device)



    def setup_classifier(self, params, model_fc_out, device):
        # Classifier selection, we use either a linear model or a NCC. 

        if params['classifier']['type'] == 'NCC':
            # Uses the originally proposed l2 NCC variant
            classifier = NCC(n_way=params['task']['n_way'],
                k_shot=params['task']['k_shot'],
                q_queries=params['task']['q_queries'],
                device=device)

        elif params['classifier']['type'] == 'sklin':
            # We use an sklearn lin model as it is faster to train generally
            classifier = skLinear(input_features=model_fc_out,
                adapt_steps=params['classifier']['adapt_steps'],
                n_way=params['task']['n_way'],
                k_shot=params['task']['k_shot'],
                q_queries=params['task']['q_queries'],
                device=device)
        
        return classifier


    def setup_batching(self, params, variable, device):
        if variable:
            prep_batch = prep_var_eval_1d(n_way=params['task']['n_way'],
                k_shot=params['task']['k_shot'],
                q_queries=params['task']['q_queries'],
                device=device,
                trans=False)

        else:
            prep_batch = prep_batch_fixed(n_way=params['task']['n_way'],
                k_shot=params['task']['k_shot'],
                q_queries=params['task']['q_queries'],
                device=device,
                trans=False)

        return prep_batch



    def setup_taskloader(self, params, dataset, hardness, classifier, variable, device):

        if variable:
            col_fn = variable_collate
            # For variable sets, much higher chance that we break through memory
            #   We take some hit in speed by setting batch size =1 in this case
            #   but it prevents crashes on smaller systems
            self.params['task']['batch_size'] = 1

        else:
            col_fn = None


        if hardness == 'avg':
            # Sets up the few-shot task loader for validation set
            task_loader = FastDataLoader(
                    dataset, 
                    batch_sampler= KShotTaskSampler(dataset=dataset,
                        batch_size=params['task']['batch_size'],
                        n_way=params['task']['n_way'],
                        k_shot=params['task']['k_shot'],
                        q_queries=params['task']['q_queries'],
                        num_tasks=params['task']['num_tasks']),
                    num_workers=params['base']['num_workers'],
                    collate_fn=col_fn)

        else:
            task_loader = FastDataLoader(
                    dataset, 
                    batch_sampler= HardTaskSampler(dataset=dataset,
                        batch_size=params['task']['batch_size'],
                        n_way=params['task']['n_way'],
                        k_shot=params['task']['k_shot'],
                        q_queries=params['task']['q_queries'],
                        classifier=classifier,
                        device=device, 
                        diff=hardness,
                        num_tasks=params['task']['num_tasks']),
                    num_workers=params['base']['num_workers'],
                    collate_fn=col_fn)
        
        return task_loader


    def eval_w_feats(self):
        # Track the testing phase
        loop = tqdm(total=len(self.task_loader), desc='Task Loop') 

        all_accs = []

        for batch_index, batch in enumerate(self.task_loader):
            # Prep batch and move to GPU
            if self.variable:
                x_support, x_queries, query_sample_nums, y = self.batch_fn(batch, self.params['task']['batch_size'])

                accs, losses = self.classifier.variable_length(x_support=x_support, 
                    x_query=x_queries, 
                    q_num=query_sample_nums, 
                    y=y)

                all_accs = all_accs + accs

            elif not self.variable:
                x, y = self.batch_fn(batch, self.params['task']['batch_size'])
                accs, losses = self.classifier.fixed_length(x, y)
                all_accs = all_accs + accs
            


            loop.update(1)

        return all_accs

    def eval_w_samples(self, model, additional_batch_fn, extra_model_params={}):
        # Track the testing phase
        loop = tqdm(total=len(self.task_loader), desc='Task Loop') 

        all_accs = []

        for batch_index, batch in enumerate(self.task_loader):
            # Prep batch and move to GPU
            if self.variable:
                x_support, x_queries, query_sample_nums, y = self.batch_fn(batch, self.params['task']['batch_size'])

                x_support = x_support.unsqueeze(2)
                x_queries = x_queries.unsqueeze(1)

                all_x_support = []

                og_support_shape = x_support.shape
                x_support = x_support.reshape(x_support.shape[0]*x_support.shape[1], x_support.shape[-2], x_support.shape[-1])

                # for idm, _ in enumerate(x_support):
                #     mini_support = x_support[idm]

                #     mini_support = additional_batch_fn(mini_support, ft_params=self.params['ft_params'])
                #     mini_support = model.forward(mini_support, **extra_model_params)

                #     all_x_support.append(mini_support)

                # x_support = torch.stack(all_x_support)

                x_support = additional_batch_fn(x_support, ft_params=self.params['ft_params'])
                x_support = model.forward(x_support, **extra_model_params)

                x_support = x_support.reshape(og_support_shape[0], og_support_shape[1], -1)

                x_queries = additional_batch_fn(x_queries, ft_params=self.params['ft_params'])
                x_queries = model.forward(x_queries, **extra_model_params)

                accs, losses = self.classifier.variable_length(x_support=x_support, 
                    x_query=x_queries, 
                    q_num=query_sample_nums, 
                    y=y)

                all_accs = all_accs + accs

            elif not self.variable:
                x, y = self.batch_fn(batch, self.params['task']['batch_size'])
                x = x.unsqueeze(2)

                og_x_shape = x.shape
                x = x.reshape(og_x_shape[0]*og_x_shape[1], og_x_shape[-2], og_x_shape[-1])
                x = additional_batch_fn(x, ft_params=self.params['ft_params'])

                x = model.forward(x, **extra_model_params)

                x = x.reshape(og_x_shape[0], og_x_shape[1], -1)

                accs, losses = self.classifier.fixed_length(x, y)
                all_accs = all_accs + accs
            


            loop.update(1)

        return all_accs


