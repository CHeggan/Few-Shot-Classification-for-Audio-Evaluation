
###############################################################################
# IMPORTS
###############################################################################
import os

import torch
import numpy as np

from dataset_.prep_batch_fns import basic_flexible_prep_batch
from dataset_.dataset_classes.FastDataLoader import FastDataLoader
from dataset_.dataset_classes.PathFinderSet import PathFinderSet
from dataset_.dataset_classes.FullSetWrapper import FullSetWrapper
from dataset_.dataset_utils import variable_collate, batch_to_log_mel_spec, \
    batch_to_log_mel_spec_plus_stft, nothing_func, variable_collate
from FewShotClassification import FewShotClassification

from models_.s3prl.ssl_model_loader import ssl_model_select, pase_forward, s3prl_forward, wavlm_forward, ssast_forward

###############################################################################
# SINGLE DATASET RUN
###############################################################################
def s3prl_single_dataset_run(params, data_params, model_name, device):

    #########################
    # INITIAL DATASET HANDLING 
    #########################
    data_name = data_params['target_data']['name']
    train_classes, val_classes, test_classes = np.load(data_params['target_data']['fixed_path'],
        allow_pickle=True)
    
    # Selects the classes of interest (coi) based on split specified
    if params['task']['split'] == 'val':
        coi = val_classes
    elif params['task']['split'] == 'train':
        coi = train_classes
    elif params['task']['split'] == 'test':
        coi = test_classes


    # If we want to use global stats, we have to load them in
    if params['data']['norm'] == 'global':
        data_files_path = os.path.join('dataset_', params['data']['source_dataset'])
        stats = np.load(params['data']['stats_file'])
    else:
        stats = None

    dataset = PathFinderSet(classes=coi,
        class_root_dir=params['data_paths'][data_name],
        norm=params['data']['norm'],
        stats=stats,
        sample_rep_length=params['data']['sample_rep_length'],
        variable=data_params['target_data']['variable'],
        ext= params['data']['ext'])

    #########################
    # MODEL SELECTION
    #########################

    model = ssl_model_select(model_name, t_dim=params['model']['input_tdim']).to(device)
    model.eval()

    if model_name == 'paseplus':
        mod_forward_func = pase_forward
    elif model_name == 'wavlm':
        mod_forward_func = wavlm_forward
    elif model_name == 'ssast':
        mod_forward_func = ssast_forward
    else:
        mod_forward_func = s3prl_forward

    # No extra params needed for forward pass
    extra_params = {}

    #########################
    # FINE-TUNING DECISION
    #########################
    if params['eval']['fine_tune']:
        model.train()
        # We use a work around here and enforce we use gradients
        torch.set_grad_enabled(True)
    elif not params['eval']['fine_tune']:
        model.eval()
        torch.set_grad_enabled(False)
    else:
        raise ValueError('Cant decide whether to fine-tune or not')

    #########################
    # ADDITIONAL DATA FUNCS
    #########################
    if params['data']['in_dims'] == 1 and params['data']['in_channels'] ==1:
        extra_batch_work = nothing_func
    elif params['data']['in_dims'] == 2 and params['data']['in_channels'] ==1:
        extra_batch_work = batch_to_log_mel_spec
    elif params['data']['in_dims'] == 2 and params['data']['in_channels'] ==3:
        extra_batch_work = batch_to_log_mel_spec_plus_stft
    else:
        raise ValueError('Thaaaaaaanks, an incorrect configuration file')

    #########################
    # COLLATION FUNCTION
    #########################
    # for variable length we need a different collaction function
    # Defines the datasets to be used
    if data_params['target_data']['variable']:
        col_fn = variable_collate
    else:
        col_fn = None

    #########################
    # FEW-SHOT PROBLEM
    #########################
    fs_dataset = dataset

    results = []
    for hardness in params['task']['hardness']:

        model_fc_out = params['model']['encoder_fc_dim']

        with torch.no_grad():
            fs_class = FewShotClassification(dataset=fs_dataset,
                params=params,
                model_fc_out=model_fc_out,
                device=device,
                variable=data_params['target_data']['variable'],
                hardness=hardness,
                additional_fn=extra_batch_work
                )


            if params['use_feats'][data_name] == True:
                accs = fs_class.eval_w_feats()

            else:
                accs = fs_class.eval_w_samples(model=model,
                    additional_batch_fn=extra_batch_work,
                    extra_model_params=extra_params,
                    mod_forward_func=mod_forward_func)


        mean = np.mean(accs)
        std = np.std(accs)
        ci_95 = (1.96 / np.sqrt(int(params['task']['num_tasks']))) * std *100


        result_dict = {'model_name': model_name,
            'dataset': data_params['target_data']['name'], 
            'hardness': hardness,
            'mean': mean,
            'std': std,
            'CI_95': ci_95}

        results.append(result_dict)

        print('\n',data_params['target_data']['name'], mean, '\n')
    return results
