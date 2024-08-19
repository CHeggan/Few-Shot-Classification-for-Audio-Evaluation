"""
Controls loading of unsupervised resnet50 models from:
    -> https://github.com/linusericsson/ssl-transfer/tree/main

Models supported = InsDis, MoCo-v1, PCL-v1, PIRL, PCL-v2, SimCLR-v1, MoCo-v2, SimCLR-v2,
    SeLa-v2, InfoMin, BYOL, DeepCluster-v2, SwAV
"""

import os

import torch
import torch.nn as nn
from torchvision import models, datasets


class ResNetBackbone(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

        self.model = models.resnet50(pretrained=False)
        del self.model.fc

        state_dict = torch.load(os.path.join('X:/Trained_model_storage/SSL Images/checkpoints/rn50', self.model_name + '.pth'))
        if self.model_name == 'simclr-v1':
            state_dict = state_dict['state_dict']
            del state_dict['fc.weight']
            del state_dict['fc.bias']
        self.model.load_state_dict(state_dict)

        self.model.train()
        print("num parameters:", sum(p.numel() for p in self.model.parameters()))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        return x


    

