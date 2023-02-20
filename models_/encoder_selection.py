"""
Functions to select backbone encoder
"""
###############################################################################
# IMPORTS
###############################################################################
import torch
import torch.nn as nn

from models_.resnets import resnet18, resnet34, resnet50, resnet101, resnet152
from models_.residual_adapter import adater_resnet34, adapter_resnet18
from models_.split_resnet import split_resnet18, split_resnet34, split_resnet50, split_resnet101, split_resnet152

###############################################################################
# RESNET SELECTION
###############################################################################
def resnet_selection(dims, fc_out, in_channels, model_name='resnet18'):
    if model_name == 'resnet18':
        model = resnet18(dims, fc_out, in_channels)
    elif model_name == 'resnet34':
        model = resnet34(dims, fc_out, in_channels)
    elif model_name == 'resnet50':
        model = resnet50(dims, fc_out, in_channels)
    elif model_name == 'resnet101':
        model = resnet101(dims, fc_out, in_channels)
    elif model_name == 'resnet152':
        model = resnet152(dims, fc_out, in_channels)
    else:
        raise ValueError('ResNet name not recognised')

    return model


###############################################################################
# ADAPTER RESNET SELECTION
###############################################################################
def adapter_resnet_selection(dims, fc_out_list, in_channels, task_mode, num_tasks, model_name):
    if model_name == 'adapter_resnet18':
        model = adapter_resnet18(dims=dims, fc_out=fc_out_list, in_channels=in_channels,
            task_mode=task_mode, num_tasks=num_tasks)
    else:
        raise ValueError('Adapter ResNet name not recognised')

    return model 

# model = resnet_selection(dims=1, fc_out=1000, model_name='resnet18')


###############################################################################
# SPLIT RESNET SELECTION
###############################################################################
def split_resnet_selection(dims, fc_out, in_channels, model_name='split_resnet18'):
    if model_name == 'split_resnet18':
        model = split_resnet18(dims, fc_out, in_channels)
    elif model_name == 'split_resnet34':
        model = split_resnet34(dims, fc_out, in_channels)
    elif model_name == 'split_resnet50':
        model = split_resnet50(dims, fc_out, in_channels)
    elif model_name == 'split_resnet101':
        model = split_resnet101(dims, fc_out, in_channels)
    elif model_name == 'split_resnet152':
        model = split_resnet152(dims, fc_out, in_channels)
    else:
        raise ValueError('Split ResNet name not recognised')

    return model