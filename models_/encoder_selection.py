"""
Functions to select backbone encoder
"""
###############################################################################
# IMPORTS
###############################################################################
import torch
import torch.nn as nn

# Import all resnet related models
from models_.resnets.residual_adapter import adapter_resnet18, adapter_resnet34
from models_.resnets.resnets import resnet18, resnet34, resnet50, resnet101, resnet152
from models_.resnets.split_resnet import split_resnet18, split_resnet34, split_resnet50, split_resnet101, split_resnet152


# Import all asts and vit models
from models_.transformers.base_asts import ast_tiny, ast_small, ast_base
from models_.transformers.split_asts import split_ast_tiny, split_ast_small, split_ast_base
from models_.transformers.adapter_asts import adapter_ast_tiny, adapter_ast_small, adapter_ast_base


###############################################################################
# RESNET SELECTIONS
###############################################################################
########################################
# NORMAL RESNETS
########################################
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


########################################
# ADAPTER RESNETS
########################################
def adapter_resnet_selection(dims, fc_out_list, in_channels, task_mode, num_tasks, model_name='adapter_resnet18'):
    if model_name == 'adapter_resnet18':
        model = adapter_resnet18(dims=dims, fc_out=fc_out_list, in_channels=in_channels,
            task_mode=task_mode, num_tasks=num_tasks)
    elif model_name == 'adapter_resnet34':
        model = adapter_resnet34(dims=dims, fc_out=fc_out_list, in_channels=in_channels,
            task_mode=task_mode, num_tasks=num_tasks)

    else:
        raise ValueError('Adapter ResNet name not recognised')

    return model 


########################################
# SPLIT RESNETS
########################################
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


###############################################################################
# AUDIO SPECTROGRAM TRANSFORMER SELECTION SELECTIONS
###############################################################################
########################################
# NORMAL ASTS
########################################
def ast_selection(fc_out, in_channels, input_tdim, model_name):
    # vit_core in {tiny224, small224, base224}
    if model_name == 'ast_tiny':
        model = ast_tiny(in_channels=in_channels, fc_out=fc_out, input_tdim=input_tdim)
    elif model_name == 'ast_small':
        model = ast_small(in_channels=in_channels, fc_out=fc_out, input_tdim=input_tdim)
    elif model_name == 'ast_base':
        model = ast_base(in_channels=in_channels, fc_out=fc_out, input_tdim=input_tdim)
    else:
        raise ValueError('Base AST name not recognised')
    return model


########################################
# SPLIT ASTS
########################################
def split_ast_selection(fc_out, in_channels, input_tdim, model_name):
    # vit_core in {tiny224, small224, base224}
    if model_name == 'split_ast_tiny':
        model = split_ast_tiny(in_channels=in_channels, fc_out=fc_out, input_tdim=input_tdim)
    elif model_name == 'split_ast_small':
        model = split_ast_small(in_channels=in_channels, fc_out=fc_out, input_tdim=input_tdim)
    elif model_name == 'split_ast_base':
        model = split_ast_base(in_channels=in_channels, fc_out=fc_out, input_tdim=input_tdim)
    else:
        raise ValueError('Split AST name not recognised')
    return model

########################################
# ADAPTER ASTS
########################################
def adapter_ast_selection(fc_out, in_channels, input_tdim, model_name, num_tasks, adapter_type):
    # vit_core in {tiny224, small224, base224}
    if model_name == 'adapter_ast_tiny':
        model = adapter_ast_tiny(in_channels=in_channels, fc_out=fc_out, input_tdim=input_tdim,
                                num_tasks=num_tasks, adapter_type=adapter_type)
    elif model_name == 'adapter_ast_small':
        model = adapter_ast_small(in_channels=in_channels, fc_out=fc_out, input_tdim=input_tdim,
                                num_tasks=num_tasks, adapter_type=adapter_type)
    elif model_name == 'adapter_ast_base':
        model = adapter_ast_base(in_channels=in_channels, fc_out=fc_out, input_tdim=input_tdim,
                                num_tasks=num_tasks, adapter_type=adapter_type)
    else:
        raise ValueError('Adapter AST name not recognised')
    return model