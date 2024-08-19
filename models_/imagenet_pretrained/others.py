"""
Loading options for dino v2, swav and barlow twins models
"""

import torch

def load_dinov2(model_name):
    if model_name in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14']:
        model = torch.hub.load('facebookresearch/dinov2', model_name)
    else:
        raise ValueError(f'{model_name} not recognised')
    return model



def load_swav(model_name):
    if model_name in ['swav_resnet50', 'swav_resnet50w2', 'swav_resnet50w4', 'swav_resnet50w5']:
        model = torch.hub.load('facebookresearch/swav:main', model_name.split('_')[1])
    else:
        raise ValueError(f'{model_name} not recognised')
    return model
    # model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    # rn50w2 = torch.hub.load('facebookresearch/swav:main', 'resnet50w2')
    # rn50w4 = torch.hub.load('facebookresearch/swav:main', 'resnet50w4')
    # rn50w5 = torch.hub.load('facebookresearch/swav:main', 'resnet50w5')


def load_dino(model_name):
    if model_name in ['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8', 'dino_resnet50']:
        model = torch.hub.load('facebookresearch/dino:main', model_name)
    else:
        raise ValueError(f'{model_name} not recognised')
    return model

def load_barlowtwins(model_name):
    if model_name in ['bt_resnet50']:
        model = torch.hub.load('facebookresearch/barlowtwins:main', model_name.split('_')[1])
    else:
        raise ValueError(f'{model_name} not recognised')
    return model