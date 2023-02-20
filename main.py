"""
Includes the main experiment call as well as:
    - experiment param loading
    - model param loading
"""
################################################################################
# IMPORTS
################################################################################
import gc
import os
import sys
import time
import glob

import yaml
import torch
import argparse
import numpy as np
import pandas as pd

from utils import set_seed
from single_set import single_dataset_run

################################################################################
# MAIN CALL & ARGPARSE  
################################################################################
if __name__ == '__main__':

    #########################
    # PARAM LOADING
    #########################
    # Loads in other expeirment params
    with open("config.yaml") as stream:
        params = yaml.safe_load(stream)

    #########################
    # DEVICE SETTING
    #########################
    # Setting of cuda device
    if params['base']['cuda'] == False:
        device = torch.device('cpu')
    elif params['base']['cuda']:
        device = torch.device('cuda')
    else: 
        raise ValueError('Unclear use of CUDA vs CPU')

    #########################
    # SEED SETTING
    #########################
    if params['base']['seed'] == 'random':
        seed = np.random.randint(low=0, high=1000)
        params['base']['seed'] = seed
    else:
        seed = params['base']['seed']
        
    set_seed(seed)


    #########################
    # LOAD/CREATE NEW DF
    #########################
    # We allow for a restart situation if we have models leftover (i.e crash etc)
    results_path = params['base']['results_path']

    if os.path.isfile(results_path):
        results_df = pd.read_csv(results_path, index_col=0)
        models_already_tested = results_df['model_name'].values
        print(f'Loaded previous results file: {results_path}')
    else:
        results_df = pd.DataFrame()
        models_already_tested = []
        print(f'Could not load old results file: {results_path}')


    #########################
    # LOOP OVER MODELS
    #########################
    model_files = glob.glob(params['model']['model_dir'] + '/' +"*.pt")
    for idx, model_file_path in enumerate(model_files):
        trained_model_name = model_file_path.split('.')[0].split('/')[-1]

        print(f'\n Testing on Model Named {idx, trained_model_name} \n')

        #########################
        # LOOP OVER DATASETS
        #########################
        # Grabs all available dataset config files
        data_params_file_list = os.listdir(params['base']['path_to_configs'])

        print(data_params_file_list)

        all_results = []
        for idx, config_file in enumerate(data_params_file_list):
            # Grabs the dataset specific configs 
            with open(os.path.join(params['base']['path_to_configs'], config_file)) as stream:
                data_params = yaml.safe_load(stream)

            # If we are interested in validation splits, we pass those which are tets ony
            if params['task']['split'] in ['train', 'val']:
                if data_params['target_data']['all_test'] == True:
                    continue

            # # If we have done this model dataset combination, we can skip it
            # if (trained_model_name in models_already_tested):
            #     if data_params['target_data']['name'] in results_df[results_df['model_name'] == trained_model_name]['dataset'].values:
            #         continue

            print(config_file)
            # if config_file != 'kaggle_config.yaml':
            #     continue

            # # Fast ones for quick eval testing
            # if ('esc' not in config_file) and ('kaggle' not in config_file) and ('cremad' not in config_file) and ('saa' not in config_file):
            #     continue

            # if 'esc' not in config_file:
            #     continue

            # if 'common' not in config_file:
            #     continue

            # If we are extracting fix length data, we are less likely to exceed memory
            if data_params['target_data']['variable'] == False:
                params['extraction']['batch_size'] = params['extraction']['batch_size']*2
            else:
                params['extraction']['batch_size'] = 1

            with torch.no_grad():
                results = single_dataset_run(params=params, 
                    data_params=data_params,
                    model_file_path=model_file_path,
                    device=device)

                all_results = all_results + results

            if params['base']['cuda']:
                torch.cuda.empty_cache()
                gc.collect()


        model_df = pd.DataFrame(all_results)
        print(model_df)

        

            

            
