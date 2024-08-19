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
import math
import time
import glob

import yaml
import torch
import argparse
import numpy as np
import pandas as pd

from fs_utils import set_seed
from single_set import single_dataset_run

################################################################################
# MAIN CALL & ARGPARSE  
################################################################################
if __name__ == '__main__':

    #########################
    # ARGPARSE
    #########################
    parser = argparse.ArgumentParser(description='Few-Shot Audio Evaluation Codebase')

    # Model Controls
    parser.add_argument('--model_name', required=True, type=str,
        help='What model to use. Refer to Models_/encoder_selection.py  to see full options')

    parser.add_argument('--adapter', required=True, type=str,
        help='Type of adapter to use for MTL approach', choices=['None', 'split', 'bn', 'series', 'parallel', 'og_adapter', 'adaptformer_series', 'adaptformer_parallel'])
    
    parser.add_argument('--num_splits', required=True, type=int,
            help='Number f splits in final backbone layer. Cannot be <2 with adapter type split')

    parser.add_argument('--model_fc_out', required=True, type=int,
        help='Size of the final fully-connected head in backbone model, assumes we merge task heads, per head we do total/num heads')
    
    parser.add_argument('--model_dir', required=True, type=str,
        help='Directory of trained models to use for evaluation')

    # Results file name
    parser.add_argument('--results_file', required=True, type=str,
        help='Name of results file to create for this set of models')


    # Few-Shot Task Setup
    parser.add_argument('--num_tasks', required=True, type=int,
        help='Number of few-shot tasks to generate and solve')

    parser.add_argument('--split', required=True, type=str, 
        help='Split of datasets to use for evaluation', choices=['val', 'test'])
    
    parser.add_argument('--classifier', required=True, type=str, 
        help='Type of classifier to use', choices=['lin', 'sklin', 'NCC'])

    parser.add_argument('--n_way', required=False, type=float, default=5,
        help='Number of classes present in few-shot task')

    parser.add_argument('--k_shot', required=False, type=int, default=1,
        help='Number of input channels data has to model')
    
    parser.add_argument('--q_queries', required=False, type=int, default=1,
        help='Number of input channels data has to model')
    

    # GPU selection
    parser.add_argument('--gpu', type=str, required=False, default='yeet',
        help='CUDA object to pass model onto')


    ## Data format input
    parser.add_argument('--dims', required=False, type=int, default=2,
        help='Number of dimensions input data has', choices=[1, 2])

    parser.add_argument('--in_channels', required=False, type=int, default=3,
        help='Number of input channels data has to model', choices=[1, 3])
    
    parser.add_argument('--rep_length', required=False, type=float, default=5,
        help='Representation length of training data in seconds. How large should the samples be for training. Default \
        uses the og_rep_length for the specific data specified in config file.\
        If given value is 0, we pad sequences per batch and use whole samples. Note this does not work with AST models')

    args = vars(parser.parse_args())

    #########################
    # PARAM LOADING
    #########################
    # Loads in other expeirment params
    with open("config.yaml") as stream:
        params = yaml.safe_load(stream)

    #########################
    # COMBINE ARGUMENTS
    #########################
    params['adapters']['task_mode'] = args['adapter']
    params['adapters']['num_splits'] = args['num_splits']

    params['model']['dims'] = args['dims']
    params['model']['in_channels'] = args['in_channels']

    params['model']['encoder_fc_dim'] = args['model_fc_out']
    params['model']['name'] = args['model_name']
    params['model']['model_dir'] = args['model_dir']

    params['task']['split'] = args['split']
    params['task']['n_way'] = args['n_way']
    params['task']['k_shot'] = args['k_shot']
    params['task']['q_queries'] = args['q_queries']
    params['task']['num_tasks'] = args['num_tasks']
    params['classifier']['type'] = args['classifier']


    params['base']['results_path'] = 'RESULT FILES/' + args['results_file'] + '.csv'

    # GPU assignment
    if args['gpu'] == 'yeet':
        pass
    else:
        params['base']['cuda'] = args['gpu']

    

    # The number of samples expected in each training input sample
    # If we dont give a value, we default to value in argparse. This is different 
    # from training where we default to dataset og length, as we dont have real notion of that here
    params['data']['sample_rep_length'] = int(args['rep_length'] * params['ft_params']['sample_rate'])


    # Auto calculate the number of time frames our spectrogram has. Can calc frame rate as SR/Hop
    # Can reduce to (rep length * sr) / hop
    params['model']['input_tdim'] = math.ceil( params['data']['sample_rep_length']/params['ft_params']['hop_length'])

    #########################
    # DEVICE SETTING
    #########################
    # Setting of cuda device
    if params['base']['cuda'] == True:
        device = torch.device('cuda')

    elif params['base']['cuda'] == 'cpu':
        device = torch.device('cpu')

    else:
        cuda_int = params['base']['cuda']
        device = torch.device('cuda:' + str(cuda_int))

    print(device)

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
    all_results = []
    model_files = glob.glob(params['model']['model_dir'] + '/' +"*.pt")
    print('yo', model_files)
    for idx, model_file_path in enumerate(model_files):
        trained_model_name = model_file_path.split('.')[0].split('/')[-1]

        print(f'\n Testing on Model Named {idx, trained_model_name} \n')

        #########################
        # LOOP OVER DATASETS
        #########################
        # Grabs all available dataset config files
        data_params_file_list = os.listdir(params['base']['path_to_configs'])

        print(data_params_file_list)

        #all_results = []
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

            # if 'nsynth' not in config_file:
            #     continue

            # if 'sc' not in config_file:
            #     continue

            # If rep length is given as 0 (meaning we pad), we force datasets to operate as variable so that they can be collated properly
            if int(args['rep_length']) == 0:
                data_params['target_data']['variable'] = False

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
    model_df.to_csv(params['base']['results_path'])

        

            

            
