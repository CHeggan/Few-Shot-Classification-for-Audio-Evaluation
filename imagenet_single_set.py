
###############################################################################
# IMPORTS
###############################################################################
import os

import torch
import numpy as np
from torchvision import transforms

from dataset_.prep_batch_fns import basic_flexible_prep_batch
from dataset_.dataset_classes.FastDataLoader import FastDataLoader
from dataset_.dataset_classes.PathFinderSet import PathFinderSet
from dataset_.dataset_classes.FullSetWrapper import FullSetWrapper
from dataset_.dataset_utils import variable_collate, batch_to_log_mel_spec, \
    batch_to_log_mel_spec_plus_stft, nothing_func, variable_collate, min_max_scale
from FewShotClassification import FewShotClassification

# Load in pre-trained model evals
from models_.imagenet_pretrained.supervised import load_supervised_torch_model
# Ericsson et al rn50 models
from models_.imagenet_pretrained.ericsson_unsupervised import ResNetBackbone
# Dino v2 models (transformers)
from models_.imagenet_pretrained.others import load_dinov2, load_swav, load_barlowtwins, load_dino
# Load simclr v1 and v2 models
from models_.imagenet_pretrained.simclrv1 import load_simclrv1
from models_.imagenet_pretrained.simclrv2 import load_simclrv2

from models_.imagenet_pretrained.simsiam import load_simsiam

###############################################################################
# MODIFY FORWARD FUNC FOR IMAGES
###############################################################################
def modified_forward_func(transforms):
    def modified_forward_func(model, data):
        data = data.squeeze()
        data = transforms(data)
        data = model.forward(data)
        return data
    return modified_forward_func


###############################################################################
# SINGLE DATASET RUN
###############################################################################
def imagenet_single_dataset_run(params, data_params, model_name, device):

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

    # Load supervised pre-trained model
    if model_name in ['alexnet', 'inception_v3', 'googlenet', 'maxvit_t',
                        'convnext_small', 'convnext_tiny', 'convnext_base', 'convnext_large',
                        'densenet121', 'densenet161', 'densenet169', 'densenet201',
                        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                        'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
                        'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l',
                        'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3',
                        'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
                        'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 
                        'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf',
                        'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf',
                        'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf',
                        'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50',
                        'resnext50_32x4d', 'resnext101_32x8d',
                        'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
                        'squeezenet1_0', 'squeezenet1_1',
                        'swin_t', 'swin_b', 'swin_s', 'swin_v2_t', 'swin_v2_s', 'swin_v2_b',
                        'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 
                        'vgg11_bn', 'vgg11', 'vgg13_bn', 'vgg13', 'vgg16_bn', 'vgg16', 'vgg19_bn', 'vgg19',
                        'wide_resnet50_2', 'wide_resnet101_2']:
        
        model, _ = load_supervised_torch_model(model_name)

    # load self-supervised pre-trained model (RN50) used in Ericsson et al 
    elif model_name in ['insdis', 'moco-v1', 'pcl-v1', 'pirl', 'pcl-v2', 'simclr-v1', 'moco-v2', 'simclr-v2', 
                      'sela-v2', 'infomin', 'byol', 'deepcluster-v2', 'swav']:
        model = ResNetBackbone(model_name=model_name)

    # Load unsupervised dino v2 model (transformers)
    elif model_name in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14']:
        model = load_dinov2(model_name)

    # Load swav models
    elif model_name in ['swav_resnet50', 'swav_resnet50w2', 'swav_resnet50w4', 'swav_resnet50w5']:
        model = load_swav(model_name)

    # Load barlow twins models
    elif model_name in ['bt_resnet50']:
        model = load_barlowtwins(model_name)

    # Load simclr v1 models 
    elif model_name in ['simclr-v1-resnet50-1x', 'simclr-v1-resnet50-2x', 'simclr-v1-resnet50-4x']:
        model = load_simclrv1(model_name)

    # Load simclr v2 models
    elif model_name in ['simclr-v2-r50_1x_sk0', 'simclr-v2-r50_1x_sk1', 'simclr-v2-r50_2x_sk1', 'simclr-v2-r101_1x_sk1', 'simclr-v2-r101_2x_sk1', 
                        'simclr-v2-r152_1x_sk1', 'simclr-v2-r152_2x_sk1', 'simclr-v2-r152_3x_sk1']:
        model = load_simclrv2(model_name)


    elif model_name in ['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8', 'dino_resnet50']:
        model = load_dino(model_name)

    elif model_name in ['simsiam-rn50']:
        model = load_simsiam(model_name)

    else:
        raise ValueError(f'{model_name} not recognised ')

    model = model.to(device)
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
    # MODIFIED FORWARD FUNC
    #########################

    pipeline = []
    
    if params['data']['in_resize'] == True:
        pipeline.append(
            transforms.Resize([int(224*1.15), int(224*1.15)])
        )
        pipeline.append(
            transforms.CenterCrop(224)
        )
    
    if params['data']['in_norm'] == True:
        pipeline.append(min_max_scale)
        pipeline.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        )

    pipeline = transforms.Compose(pipeline)

    print(pipeline)

    mod_forward_func = modified_forward_func(pipeline)


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
        ci_95 = (1.96 / np.sqrt(int(params['task']['num_tasks']) * params['task']['q_queries'])) * std *100

        print(mean, std, ci_95)

        result_dict = {'model_name': model_name,
            'dataset': data_params['target_data']['name'], 
            'hardness': hardness,
            'mean': mean,
            'std': std,
            'CI_95': ci_95}

        results.append(result_dict)

        print('\n',data_params['target_data']['name'], mean, '\n')
    return results
