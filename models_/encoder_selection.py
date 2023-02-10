"""
Functions to select backbone encoder
"""
###############################################################################
# IMPORTS
###############################################################################
import torch
import torch.nn as nn

from models_.resnets import resnet18, resnet34, resnet50, resnet101, resnet152

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


# model = resnet_selection(dims=1, fc_out=1000, model_name='resnet18')