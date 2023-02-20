"""
ResNet model modules and classes with adapters. Contains modified code that allows for 
    exact same parameterised resnets to be created for 1d and 2d data samples. We 
    include the original series and parallel convolutions adapters included in this work:
        https://openaccess.thecvf.com/content_cvpr_2018/papers/Rebuffi_Efficient_Parametrization_of_CVPR_2018_paper.pdf
"""
###############################################################################
# IMPORTS
###############################################################################
import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# COUNT MODEL PARAMS
###############################################################################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

###############################################################################
# ADAPTER SEQUENTIAL WORKAROUND
###############################################################################
class mySequential(nn.Sequential):
    def forward(self, x, task_int):
        for module in self._modules.values():
            x = module(x=x, task_int=task_int)
        return x

###############################################################################
# GENERALISED CONV PARTS
###############################################################################
def conv3x3(in_planes, out_planes, dims, stride=1, groups=1, dilation=1):
    """3x3 or 9x1 convolution with padding"""
    if dims == 1:
        return nn.Conv1d(in_planes, out_planes, kernel_size=9, stride=stride,
                     padding=int(np.floor(9/2)), groups=groups, bias=False, dilation=dilation)
    elif dims == 2: 
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    else:
        raise ValueError('Dims not recognised') 


# def conv1x1(in_planes, out_planes, dims, stride=1):
#     """1x1 or 1x1 convolution"""
#     if dims == 1:
#         return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#     elif dims == 2:
#         return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#     else:
#         raise ValueError('Dims not recognised')

###############################################################################
# ADAPTER CONVOLUTIONS
###############################################################################
def conv1x1_fonc(in_planes, dims, out_planes=None, stride=1, bias=False):

    if dims == 1:
        if out_planes is None:
            return nn.Conv1d(in_planes, in_planes, kernel_size=1, stride=stride, padding=0, bias=bias)
        else:
            return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

    elif dims == 2:
        if out_planes is None:
            return nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, padding=0, bias=bias)
        else:
            return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

    else:
        raise ValueError('Dims not recognised')


###############################################################################
# 1x1 CLASS FOR ADAPTER CONVOLUTION
###############################################################################
class conv1x1(nn.Module):
    
    def __init__(self, planes, dims, task_mode, out_planes=None, stride=1):
        super(conv1x1, self).__init__()

        self.task_mode = task_mode

        # For series adaptors, we layer a relevant batch norm and 1x1 convolution,
        #   the original input is merged with output of adaptor
        if task_mode == 'series':
            if dims == 1:
                self.conv = nn.Sequential(nn.BatchNorm1d(planes), conv1x1_fonc(planes, dims=dims))
            elif dims == 2:
                self.conv = nn.Sequential(nn.BatchNorm2d(planes), conv1x1_fonc(planes, dims=dims))
            else:
                raise ValueError('Dims not recognised')

        # For parallel adaptors, we only apply the convolution with no skip
        elif task_mode == 'parallel':
            self.conv = conv1x1_fonc(planes, dims=dims, out_planes=out_planes, stride=stride) 
        elif task_mode == 'bn':
            self.conv = conv1x1_fonc(planes, dims=dims)
        else:
            raise ValueError(f'Adapter type not recognised: {task_mode}')


    def forward(self, x):
        y = self.conv(x)
        # The skip connection for the series adaptor
        if self.task_mode == 'series':
            y += x
        return y






###############################################################################
# CONVOLUTIONAL TASK CLASS FOR ADAPTERS
###############################################################################
class conv_task(nn.Module):
    
    def __init__(self, in_planes, planes, dims, task_mode, num_tasks, stride=1, is_proj=1, second=0, drop_1=0, drop_2=0):
        super(conv_task, self).__init__()
        self.is_proj = is_proj
        self.second = second
        self.drop1 = drop_1
        self.drop2 = drop_2

        self.task_mode = task_mode

        self.conv = conv3x3(in_planes, planes, dims=dims, stride=stride)

        if dims == 1:
            batch_func = nn.BatchNorm1d
        elif dims ==2:
            batch_func = nn.BatchNorm2d
        else:
            raise ValueError('Dims not recognised')
        
        # We either use series, paralled or batch_fn adapters. Note: if task_mode is anything other than series or parallel, 
        #   module defaults to using batch norm adapters
        if task_mode == 'series' and is_proj:
            self.bns = nn.ModuleList([nn.Sequential(conv1x1(planes, dims=dims, task_mode=task_mode), batch_func(planes)) for i in range(num_tasks)])

        elif task_mode == 'parallel' and is_proj:
            self.parallel_conv = nn.ModuleList([conv1x1(in_planes, dims=dims, out_planes=planes, stride=stride, task_mode=task_mode) for i in range(num_tasks)])
            self.bns = nn.ModuleList([batch_func(planes) for i in range(num_tasks)])
        # Use batch norm adapters
        elif task_mode == 'bn':
            self.bns = nn.ModuleList([batch_func(planes) for i in range(num_tasks)])
        else:
            raise ValueError(f'Adapter type not recognised: {task_mode}')

    
    def forward(self, x, task_int):
        y = self.conv(x)

        if self.second == 0:
            if self.drop1:
                x = F.dropout2d(x, p=0.5, training = self.training)
        else:
            if self.drop2:
                x = F.dropout2d(x, p=0.5, training = self.training)

        if self.task_mode == 'parallel' and self.is_proj:
            y = y + self.parallel_conv[task_int](x)


        y = self.bns[task_int](y)

        return y


###############################################################################
# ADAPTER BASIC BLOCK
###############################################################################
# No projection: identity shortcut
class BasicBlockAdapater(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, dims, task_mode, stride=1, shortcut=0, num_tasks=1, is_proj=1):
        super(BasicBlockAdapater, self).__init__()

        self.conv1 = conv_task(in_planes, planes, dims=dims, stride=stride, 
            num_tasks=num_tasks, task_mode=task_mode, is_proj=is_proj)

        self.con_2_norm = nn.ReLU(True)
        self.conv2 = conv_task(planes, planes, dims=dims, stride=1, 
            num_tasks=num_tasks,task_mode=task_mode, is_proj=is_proj, second=1)

        self.shortcut = shortcut
        if self.shortcut == 1:
            if dims == 1:
                self.avgpool = nn.AvgPool1d(2)
            elif dims ==2:
                self.avgpool = nn.AvgPool2d(2)
            else:
                raise ValueError('Dims not recognised')
        
    def forward(self, x, task_int):
        residual = x

        y = self.conv1(x, task_int)

        y = self.con_2_norm(y)
        y = self.conv2(y, task_int)

        if self.shortcut == 1:
            pool_func = nn.AdaptiveAvgPool2d((y.shape[-2], y.shape[-1]))
            residual = pool_func(x)

            residual = torch.cat((residual, residual*0),1)

        y += residual
        y = F.relu(y)
        
        return y



###############################################################################
# GENERALISED BASIC BLOCK
###############################################################################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, dims, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()

        # Generalise to dim = 1 or 2
        if norm_layer is None:
            if dims == 1:
                norm_layer = nn.BatchNorm1d
            elif dims == 2:
                norm_layer = nn.BatchNorm2d
            else:
                raise ValueError('Dims not recognised')

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, dims=dims, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dims=dims)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

###############################################################################
# BOTTLENECK
###############################################################################
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, dims, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            if dims == 1:
                norm_layer = nn.BatchNorm1d
            elif dims == 2:
                norm_layer = nn.BatchNorm2d
            else:
                raise ValueError('Dims not recognised')

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, dims=dims)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, dims, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, dims=dims)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


###############################################################################
# ADAPTER RESNET CLASS
###############################################################################
class AdapterResNet(nn.Module):

    def __init__(self, block, layers, dims, out_dim_list, num_tasks, task_mode, in_channels=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(AdapterResNet, self).__init__()

        self.num_tasks = num_tasks
        self.num_outs = len(out_dim_list)

        # Create exception for weird input combinations
        if dims == 1 and in_channels!=1:
            raise ValueError('1-d input only supports 1 input channel')

        if norm_layer is None:
            if dims == 1:
                norm_layer = nn.BatchNorm1d
            elif dims == 2:
                norm_layer = nn.BatchNorm2d
            else:
                raise ValueError('Dims not recognised')

        self._norm_layer = norm_layer

        self.inplanes = 64

        if dims == 1:
            self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=49, stride=2, padding=3,
                                bias=False)
            self.maxpool = nn.MaxPool1d(kernel_size=9, stride=2, padding=1)         
        
        elif dims == 2:
            self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError('Dims not recognised')

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], dims=dims, 
                                        num_tasks=1, task_mode=task_mode)
        self.layer2 = self._make_layer(block, 128, layers[1], dims=dims, stride=2,
                                       num_tasks=num_tasks, task_mode=task_mode)
        self.layer3 = self._make_layer(block, 256, layers[2], dims=dims, stride=2,
                                       num_tasks=num_tasks, task_mode=task_mode)
        self.layer4 = self._make_layer(block, 512, layers[3], dims=dims, stride=2,
                                       num_tasks=num_tasks, task_mode=task_mode)

        if dims == 1:
            self.end_bns = nn.ModuleList([nn.Sequential(nn.BatchNorm1d(int(512)),nn.ReLU(True)) for i in range(num_tasks)])
            self.avgpool = nn.AdaptiveAvgPool1d(1) 

        if dims == 2:
            self.end_bns = nn.ModuleList([nn.Sequential(nn.BatchNorm2d(int(512)),nn.ReLU(True)) for i in range(num_tasks)])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        else:
            raise ValueError('Dims not recognised')


        if len(out_dim_list) != num_tasks:
            if len(out_dim_list) != 1:
                raise ValueError('Different number of output dims compared to number of specificed tasks')


        self.fc_layers = nn.ModuleList()
        # Create fully connected layers
        for out_dim in out_dim_list:
            self.fc_layers.append(  nn.Linear(512 * block.expansion, out_dim) )
    

        if dims == 1:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        elif dims == 2:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            raise ValueError('Dims not recognised')

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)





    def _make_layer(self, block, planes, nblocks, dims, task_mode, num_tasks, stride=1):
        shortcut = 0
        if stride != 1 or self.inplanes != planes * block.expansion:
            shortcut = 1

        layers = []
        layers.append(block(self.inplanes, planes, dims=dims, stride=stride, shortcut=shortcut, 
            task_mode=task_mode, num_tasks=num_tasks))

        self.inplanes = planes * block.expansion

        for i in range(1, nblocks):
            layers.append(block(self.inplanes, planes, dims=dims, task_mode=task_mode, num_tasks=num_tasks))

        return mySequential(*layers)



    def _forward_impl(self, x, task_int):
        if task_int == 'all':

            if self.num_outs == 1:
                raise ValueError('Requested all linear outputs concat but there is only 1 defined')

            all_outputs = []

            for val in range(self.num_tasks):
                y = self._forward_impl(x, task_int=val)
                all_outputs.append(y)

            x = torch.concat(all_outputs, dim=1)

        else:
            # See note [TorchScript super()]
            self.task_int = task_int
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            # For 1st layer we dont use multiple adaptors 
            x = self.layer1(x, 0)
            x = self.layer2(x, task_int)
            x = self.layer3(x, task_int)
            x = self.layer4(x, task_int)

            x = self.end_bns[task_int](x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            
            if self.num_outs == 1:
                x = self.fc_layers[0](x)
            else:
                x = self.fc_layers[task_int](x)

        return x


    def forward(self, x, task_int):
        return self._forward_impl(x, task_int=task_int)


###############################################################################
# GENERALISED RESNET CLASS
###############################################################################
class ResNet(nn.Module):

    def __init__(self, block, layers, dims, out_dim, in_channels=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()

        self.output_dim = out_dim

        # Create exception for weird input combinations
        if dims == 1 and in_channels!=1:
            raise ValueError('1-d input only supports 1 input channel')

        if norm_layer is None:
            if dims == 1:
                norm_layer = nn.BatchNorm1d
            elif dims == 2:
                norm_layer = nn.BatchNorm2d
            else:
                raise ValueError('Dims not recognised')
        self._norm_layer = norm_layer


        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if dims == 1:
            self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=49, stride=2, padding=3,
                                bias=False)
            self.maxpool = nn.MaxPool1d(kernel_size=9, stride=2, padding=1)         
        
        elif dims == 2:
            self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError('Dims not recognised')

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)



        self.layer1 = self._make_layer(block, 64, layers[0], dims=dims)
        self.layer2 = self._make_layer(block, 128, layers[1], dims=dims, stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], dims=dims, stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], dims=dims, stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        if dims == 1:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
        elif dims == 2:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError('Dims not recognised')

        self.fc = nn.Linear(512 * block.expansion, out_dim)

        if dims == 1:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        elif dims == 2:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            raise ValueError('Dims not recognised')

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, dims, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, dims, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, dims, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dims, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

###############################################################################
# RESNET CALL FUNCTIONS
###############################################################################
"""
All resnets are designed so that they can take any of the following:
    -> 1-d signal w/ 1 input channel 
    -> 2-d signal w/1 input channel (mel-spectrogram)
    -> 2-signal w/3 input channels (mel-spec + STFT real and imaginary)
"""

def adapter_resnet18(dims, fc_out, in_channels, task_mode, num_tasks):
    return AdapterResNet(block=BasicBlockAdapater, layers=[2, 2, 2, 2], dims=dims, 
        out_dim_list=fc_out, in_channels=in_channels, task_mode=task_mode, num_tasks=num_tasks)

def adater_resnet34(dims, fc_out, in_channels, task_mode, num_tasks):
    return AdapterResNet(block=BasicBlockAdapater, layers=[3, 4, 6, 3], dims=dims, 
        out_dim_list=fc_out, in_channels=in_channels, task_mode=task_mode, num_tasks=num_tasks)



# def resnet50(dims, fc_out, in_channels):
#     return ResNet(block=Bottleneck, layers=[3, 4, 6, 3], dims=dims, out_dim=fc_out, in_channels=in_channels)

# def resnet101(dims, fc_out, in_channels):
#     return ResNet(block=Bottleneck, layers=[3, 4, 23, 3], dims=dims, out_dim=fc_out, in_channels=in_channels)

# def resnet152(dims, fc_out, in_channels):
#     return ResNet(block=Bottleneck, layers=[3, 8, 36, 3], dims=dims, out_dim=fc_out, in_channels=in_channels)



###############################################################################
# EXAMPLE CALL
###############################################################################

# num_tasks = 10

# model = adapter_resnet18(dims=2, in_channels=1, fc_out=[1000], task_mode='parallel', num_tasks=num_tasks)

# data = torch.rand(10, 1, 128, 313)

# for i in range(num_tasks):
#     print(model.forward(data, task_int=i).shape)


# # print(model.forward(data, task_int='all').shape)

# print(count_parameters(model))