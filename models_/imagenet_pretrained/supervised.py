"""
We utilise pre-trained models from pytorch (https://pytorch.org/vision/stable/models.html)

These include a wide variety of architecture types:
    - AlexNet
    - ConvNeXt
    - DenseNet
    - EfficientNet
    - EfficientNetV2
    - GoogLeNet
    - Inception V3
    - MaxVit
    - MNASNet
    - MobileNet V2
    - MobileNet V3
    - RegNet
    - ResNet
    - ResNeXt
    - ShuffleNet V2
    - SqueezeNet
    - SwinTransformer
    - VGG
    - VisionTransformer
    - Wide ResNet
"""
################################################################################
# IMPORTS
################################################################################
import builtins

import torchvision
import torch.nn as nn


################################################################################
# LOWERCASE ATTRIBUTE MATCHER and IDENTITY FUNCTION CLASS
################################################################################
orig_getattr = builtins.getattr
def igetattr(obj, attr):
    for a in dir(obj):
        if a.lower() == attr.lower():
            return orig_getattr(obj, a)
        

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

################################################################################
# PRE-TRAINED MODEL SELECTION
################################################################################
def load_supervised_torch_model(name, weights_name='IMAGENET1K_V1', features=True):
    # Grabs the actual modle function from torchvision
    model_func = getattr(torchvision.models, name.lower())
    # Grabs the weights class for our model using lowercase getattr
    weights_class = igetattr(torchvision.models, name + '_weights')
    
    # If a selected model class has undated/upgraded weights, we use them instead
    if hasattr(weights_class, 'IMAGENET1K_V2'):
        weights_name = 'IMAGENET1K_V2'

    # Lods the wieghts into model function
    model = model_func(weights=weights_name)
    # Grabs the weights object we need
    weights_object = getattr(weights_class, weights_name)
    # Extracts transformation pipeline from weights object
    preprocess = weights_object.transforms()

    if features:
        model.fc = Identity()

    return model, preprocess

################################################################################
# LIST OF MODELS
################################################################################
models = ['alexnet', 'inception_v3', 'googlenet', 'maxvit_t',
          
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
    'wide_resnet50_2', 'wide_resnet101_2']