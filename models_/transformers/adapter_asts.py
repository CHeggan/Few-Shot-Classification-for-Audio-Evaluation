"""
Adapter enabled implementation of Audio Spectrogram Transformers. 

Main differences from basic vision transformers is the patch embedding overwrite 
    to allow usage with variable length spectrograms.

Adapters are embedded in the core vision transformer used. 
"""

###############################################################################
# IMPORTS
###############################################################################
import timm
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple,trunc_normal_

from .model_utils import count_parameters
from .adapter_vits import adapter_vit_tiny_patch16_224, adapter_vit_small_patch16_224, adapter_vit_base_patch16_224

###############################################################################
# AUDIO SPECTROGRAM TRANSFORMER PARTS
###############################################################################
########################################
# PATCH EMBED OVERWRITE
########################################
# override the timm package to relax the input shape constraint.
class AST_PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=False):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    



###############################################################################
# AUDIO SPECTROGRAM TRANSFORMER (AST) MAIN MODEL
###############################################################################
class AdapterASTModel(nn.Module):
    def __init__(
            self, 
            in_channels,
            out_dim_list,
            num_tasks,
            adapter_type,
            vit_core,
            fstride=10, 
            tstride=10, 
            input_fdim=128, 
            input_tdim=1024, 
            verbose=True
    ):
        """An adapter and split final layer implementation of the Audio Spectrogram Transformer. 
            Modified to use adapter based vit core,  multiple fc heads, use different channel numbers, and 
            is no longer as dependant on timm package - still somewhat. 

        Args:
            in_channels (int): Number of in_channels for the data. {1, 3}
            out_dim_list (list): List of ints for sizes of output fc linear heads
            num_tasks (int): Number of adapter modules ot create at each point. Should be = len(out_dim_list)
            adapter_type (str): Type of adapter module to use. {adaptformer_series, adaptformer_parallel, og_adapter}
            vit_core (str): Name of vision transformer core to use. {tiny224, small224, base224}
            fstride (int, optional): Frequency stride to take. Defaults to 10.
            tstride (int, optional): Time stride to take. Defaults to 10.
            input_fdim (int, optional): Frequency input dimensionality. Defaults to 128.
            input_tdim (int, optional): Time input dimensionality. Defaults to 1024.
            verbose (bool, optional): Whether to print output/updates. Defaults to True.

        Raises:
            Exception: Raises error if an unknown vit core name is passes
        """

        super(AdapterASTModel, self).__init__()

        self.num_tasks = num_tasks
        self.num_outs = len(out_dim_list)

        #assert timm.__version__ == '0.8.23.dev0', 'Please use timm == 0.8.23.dev0, the code requires use of layers module.'

        # Replace the patch embed class used by the vit code  - we passed in as argument to avoid this
        #timm.layers.PatchEmbed = PatchEmbed

        # Doesnt matter what we give as fc out as we are replacing head anyway, we
        #   pass the same list to avoid exception. The created heads are never used
        if vit_core == 'tiny224':
            self.vit_core = adapter_vit_tiny_patch16_224(fc_out=out_dim_list, num_tasks=num_tasks, 
                                                         adapter_type=adapter_type, embed_layer=AST_PatchEmbed)
        elif vit_core == 'small224':
            self.vit_core = adapter_vit_small_patch16_224(fc_out=out_dim_list, num_tasks=num_tasks, 
                                                         adapter_type=adapter_type, embed_layer=AST_PatchEmbed)
        elif vit_core == 'base224':
            self.vit_core = adapter_vit_base_patch16_224(fc_out=out_dim_list, num_tasks=num_tasks, 
                                                         adapter_type=adapter_type, embed_layer=AST_PatchEmbed)
        else:
            raise Exception('Model must be one of tiny224, small224, base224')

        self.original_num_patches = self.vit_core.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.vit_core.pos_embed.shape[2]

        # Replace the single mlp head with a list of fully connected layers
        self.fc_layers = nn.ModuleList()
        # Create fully connected layers
        for out_dim in out_dim_list:
            self.fc_layers.append(  nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, out_dim)) )


        # Automatically get the intermediate shape
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.vit_core.patch_embed.num_patches = num_patches
        if verbose == True:
            print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
            print('number of patches={:d}'.format(num_patches))

        # the linear projection layer. This layer changes what dimensionality/num channels of input we take
        new_proj = torch.nn.Conv2d(in_channels, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        self.vit_core.patch_embed.proj = new_proj

        # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
        # Removed + 2 from 2nd term as we dont use a distilled ViT (which has two additional things concat to x before pos embed)
        new_pos_embed = nn.Parameter(torch.zeros(1, self.vit_core.patch_embed.num_patches, self.original_embedding_dim))
        self.vit_core.pos_embed = new_pos_embed
        trunc_normal_(self.vit_core.pos_embed, std=.02)


    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
    

    def forward(self, x, task_int):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        if x.ndim < 4:
            x = x.unsqueeze(1)

        x = x.transpose(2, 3)


        # If we want all, we just go over forward multiple times
        if task_int == 'all':

            if self.num_outs == 1:
                raise ValueError('Requested all linear outputs concat but there is only 1 defined')

            all_outputs = []

            for val in range(self.num_tasks):
                y = self.forward(x, task_int=val)
                all_outputs.append(y)

            x = torch.concat(all_outputs, dim=1)
        
        else:
            x = self.vit_core.patch_embed(x)
            x = x + self.vit_core.pos_embed

            x = self.vit_core.patch_drop(x)
            x = self.vit_core.norm_pre(x)

            # Have to pass through blocks individually due to sequential/forward argument issue
            for i in range(len(self.vit_core.blocks)):
                x = self.vit_core.blocks[i](x, task_int=task_int)

            x = self.vit_core.norm(x)



            # Gets the pre-logits scores out, to go through the new heads here
            # Do this to keep consistency with splitASTs
            x = self.vit_core.split_forward_head(x, task_int='doesnt matter', pre_logits=True)
            x = self.fc_layers[task_int](x)


        return x

    


###############################################################################
# AST CALL FUNCTION
###############################################################################
"""
AST takes an input of of either [batch, in_channels, input_tdim, n_mels] where 
    in_channels is in {1, 3}. 

Variant is a choice from {tiny224, small224, base224}
"""

def adapter_ast_tiny(fc_out, in_channels, input_tdim, num_tasks, adapter_type):
    return AdapterASTModel(in_channels=in_channels, vit_core='tiny224', out_dim_list=fc_out, 
                           input_tdim=input_tdim, num_tasks=num_tasks, adapter_type=adapter_type)

def adapter_ast_small(fc_out, in_channels, input_tdim, num_tasks, adapter_type):
    return AdapterASTModel(in_channels=in_channels, vit_core='small224', out_dim_list=fc_out,  
                           input_tdim=input_tdim, num_tasks=num_tasks, adapter_type=adapter_type)

def adapter_ast_base(fc_out, in_channels, input_tdim, num_tasks, adapter_type):
    return AdapterASTModel(in_channels=in_channels, vit_core='base224', out_dim_list=fc_out,  
                           input_tdim=input_tdim, num_tasks=num_tasks, adapter_type=adapter_type)


###############################################################################
# EXAMPLE PASS THROUGH
###############################################################################

# input_tdim = 157
# in_channels = 1

# model = adapter_ast_small([500, 500], in_channels=in_channels, input_tdim=input_tdim,
#                         num_tasks=2, adapter_type='adaptformer_series')

# test_input = torch.rand([10, in_channels, input_tdim, 128])

# test_output = model.forward(test_input, 0)

# print(test_output.shape)
# print(count_parameters(model))
