"""
ResNet model modules and classes. Contains modified code that allows for 
    exact same parameterised resnets to be created for 1d and 2d data samples
"""
###############################################################################
# IMPORTS
###############################################################################
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


def conv1x1(in_planes, out_planes, dims, stride=1):
    """1x1 or 1x1 convolution"""
    if dims == 1:
        return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    elif dims == 2:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    else:
        raise ValueError('Dims not recognised')

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
# GENERALISED RESNET CLASS
###############################################################################
class SeqResNet(nn.Module):

    def __init__(self, block, layers, dims, feat_dim, in_channels=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(SeqResNet, self).__init__()

        self.dims = dims
        self.feat_dim = feat_dim

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
            stride = 3
            self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=49, stride=stride, padding=3,
                                bias=False)
            self.maxpool = nn.MaxPool1d(kernel_size=9, stride=stride, padding=1)         
        
        elif dims == 2:
            stride = (2, 1)
            self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=stride, padding=3,
                                bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError('Dims not recognised')

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # An adaptive average that reduces feature dim to what we want
        #   Need this for the bottleneck layer models, as they have expansion
        self.adaptive_avg = nn.AdaptiveAvgPool1d(feat_dim)

        self.linear_downsample = nn.Linear(feat_dim*block.expansion, feat_dim)
        print(self.linear_downsample)

        self.layer1 = self._make_layer(block, 64, layers[0], dims=dims)
        self.layer2 = self._make_layer(block, 128, layers[1], dims=dims, stride=stride,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], dims=dims, stride=stride,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, feat_dim, layers[3], dims=dims, stride=stride,
                                       dilate=replace_stride_with_dilation[2])
        
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
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)
        if self.dims == 1:
            x = self.maxpool(x)
        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)

        # print('4', x.shape)

        # x = self.avgpool(x)
        # x = F.adaptive_avg_pool2d(x, (x.shape[0], x.shape[1], 1, x.shape[3]))
        if self.dims == 2:
            x = x.mean(dim=2)

        # As we remove the flattening and fc layerto keep sequential, we have to 
        #   manually reduce 
        if x.shape[1] > self.feat_dim:
            # print('yo')
            # Need to have feature dim as last 
            # So transform from (batch, feats, t_seq) -> (batch, t_seq, feats)
            x = x.transpose(1, 2)
            # Reduces the size of feature dimension to that requested upon model creation (self.feat_dim)
            x = self.linear_downsample(x)
            # Transpose data back to original dimension order
            x = x.transpose(2, 1)
        
        # print(x.shape)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

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

def seq_resnet18(dims, feat_dim, in_channels):
    return SeqResNet(block=BasicBlock, layers=[2, 2, 2, 2], dims=dims, feat_dim=feat_dim, in_channels=in_channels)

def seq_resnet34(dims, feat_dim, in_channels):
    return SeqResNet(block=BasicBlock, layers=[3, 4, 6, 3], dims=dims, feat_dim=feat_dim, in_channels=in_channels)

def seq_resnet50(dims, feat_dim, in_channels):
    return SeqResNet(block=Bottleneck, layers=[3, 4, 6, 3], dims=dims, feat_dim=feat_dim, in_channels=in_channels)



def resnet101(dims, fc_out, in_channels):
    return ResNet(block=Bottleneck, layers=[3, 4, 23, 3], dims=dims, out_dim=fc_out, in_channels=in_channels)

def resnet152(dims, fc_out, in_channels):
    return ResNet(block=Bottleneck, layers=[3, 8, 36, 3], dims=dims, out_dim=fc_out, in_channels=in_channels)



###############################################################################
# EXAMPLE CALL
###############################################################################
# model = resnet34(dims=2, fc_out=100, in_channels=1)

# data = torch.rand(10, 1, 128, 313)

# print(model.forward(data).shape)
# print(count_parameters(model))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# model = seq_resnet34(dims=2, feat_dim=500, in_channels=3).to('cuda')
# print(f'resnet18: {count_parameters(model)}')


# # Can use hop length = 256 to obtain a downsample rate ~250

# data = torch.rand((10, 3, 128, 63)).to('cuda')
# # # data = torch.rand((10, 1, 16000)).to('cuda')
# # out = model(data)
# # print(out.shape)




# from seq_simclr import Seq_SimCLR_V1
# simclr = Seq_SimCLR_V1(model, t_dim=63, reduce_over='t_steps', gru_hidden=512).to('cuda')


# out = simclr.forward(data, data)
# print(out)








# x = out

# # (batch, t_seq, feats)
# x = x.transpose(1, 2)
# print(x.shape)

# # takes (batch, t_seq, feats)
# gru = nn.GRU(x.shape[2], 1000 , num_layers=1, batch_first=True).to('cuda')
# gru_out, hidden = gru(x)

# print(gru_out.shape)



# class DotProductAttention(nn.Module):
#     def forward(self, query, key, value):
#         # Calculate attention scores by taking the dot product of the query and key
#         attention_scores = torch.matmul(query, key.transpose(1, 2))
        
#         # Apply softmax to normalize the scores and get attention weights
#         attention_weights = torch.softmax(attention_scores, dim=-1)
        
#         # Calculate the weighted sum using the attention weights
#         attended_values = torch.bmm(attention_weights, value)
        
#         return attended_values, attention_weights

# # Example usage:
# # Assuming you have query, key, and value tensors of shape [batch_size, sequence_length, hidden_dim]
# # Initialize the attention module
# attention = DotProductAttention()



# # linear = nn.Linear(1000, 500).to('cuda')



# # Apply attention mechanism to get attended values
# attended_values, attention_weights = attention(gru_out, gru_out, gru_out)
# print(attended_values.shape, attention_weights.shape)




# from nt_xentloss import NT_XentLoss


# attended_values = attended_values.transpose(1, 2)
# fc = nn.Linear(x.shape[1], 1).to('cuda')

# x = fc(attended_values).squeeze()
# print(x.shape)

# loss = NT_XentLoss(x, x, temperature=0.07)
# print(loss)


# # model = resnet34(dims=2, fc_out=1000, in_channels=3)
# # print(f'resnet34: {count_parameters(model)}')

# # model = resnet50(dims=2, fc_out=1000, in_channels=3)
# # print(f'resnet50: {count_parameters(model)}')

# # model = resnet101(dims=2, fc_out=1000, in_channels=3)
# # print(f'resnet101: {count_parameters(model)}')

# # model = resnet152(dims=2, fc_out=1000, in_channels=3)
# # print(f'resnet152: {count_parameters(model)}')