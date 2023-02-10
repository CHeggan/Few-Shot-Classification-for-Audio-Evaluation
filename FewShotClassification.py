"""
An largely enclosed few-shot classification class with accompanying methods. Sets 
    up the few-shot classification task loader (i.e easy/avg hard) and loops 
    over for avg classification results. Also deals with the majority of setup 
    around classifiers, and task samplers
"""
import numpy as np
from classifiers_.NearestCentroid import NCC
from classifiers_.Linear import Linear
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
    def __init__(self, dataset, params, model_fc_out, prep_batch, device, variable, hardness):

        self.classifier = self.setup_classifier(params,
            model_fc_out=model_fc_out, 
            device=device)

        self.task_loader = self.setup_taskloader(params=params, 
            dataset=dataset, 
            hardness=hardness, 
            classifier=self.classifier,
            variable=variable,
            device=device)


    def setup_classifier(self, params, model_fc_out, device):
        # Classifier selection, we use either a linear model or a NCC
        if params['classifier']['type'] == 'linear':
            classifier = Linear(input_features=model_fc_out.full_set.shape[-1],
                adapt_steps=params['classifier']['adapt_steps'],
                lr=params['classifier']['lr'],
                n_way=params['task']['n_way'],
                k_shot=params['task']['k_shot'],
                q_queries=params['task']['q_queries'],
                device=device)

        elif params['classifier']['type'] == 'NCC':
            # Initialises the classifier we are going to use for running our N-way k-shot
            #   tasks. For this experiment we keep things simple and only look at l2 NCC
            classifier = NCC(n_way=params['task_setup']['n_way'],
                k_shot=params['task']['k_shot'],
                q_queries=params['task']['q_queries'],
                device=device)

        elif params['classifier']['type'] == 'sklin':
            # Initialises the classifier we are going to use for running our N-way k-shot
            #   tasks. For this experiment we keep things simple and only look at l2 NCC
            classifier = skLinear(input_features=model_fc_out,
                adapt_steps=params['classifier']['adapt_steps'],
                n_way=params['task']['n_way'],
                k_shot=params['task']['k_shot'],
                q_queries=params['task']['q_queries'],
                device=device)
        
        return classifier

    def setup_batching(self, variable):
        pass


    def setup_taskloader(self, params, dataset, hardness, classifier, variable, device):

        if variable:
            col_fn = variable_collate
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
                        num_tasks=params['task']['eval_tasks']),
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
                        num_tasks=params['task']['eval_tasks']),
                    num_workers=params['base']['num_workers'],
                    collate_fn=col_fn)
        
        return task_loader


    def fixed_length_run(self):
        pass

    def var_length_run(self):
        pass

    def run(self):
        pass


