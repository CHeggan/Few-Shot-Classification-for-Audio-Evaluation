"""
Base implementation of Audio Spectrogram Transformers. 

Main differences from basic vision transformers is the patch embedding overwrite 
    to allow usage with variable length spectrograms.

Modified to take 3 channel input again.

Main code format from: https://github.com/YuanGongND/ast/blob/master/src/models/ast_models.py
"""

###############################################################################
# IMPORTS
###############################################################################
import timm
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple,trunc_normal_


from .model_utils import count_parameters
from .vits import VisionTransformer
from .vits import vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224

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
class ASTModel(nn.Module):
    def __init__(
            self, 
            in_channels, # we allow either 1 or 3 channel input as found in CLAR work
            fc_out, #replace label dim by a generic output dimensionality
            vit_core, 
            fstride=10, 
            tstride=10, 
            input_fdim=128, 
            input_tdim=1024, 
            verbose=True
    ):
        super(ASTModel, self).__init__()
        #assert timm.__version__ == '0.8.23.dev0', 'Please use timm == 0.8.23.dev0, the code requires use of layers module.'

        # Replace the patch embed class used by the vit code  - we passed in as argument to avoid this
        #timm.layers.PatchEmbed = PatchEmbed

        if vit_core == 'tiny224':
            self.vit_core = vit_tiny_patch16_224(fc_out=fc_out, embed_layer=AST_PatchEmbed)
        elif vit_core == 'small224':
            self.vit_core = vit_small_patch16_224(fc_out=fc_out, embed_layer=AST_PatchEmbed)
        elif vit_core == 'base224':
            self.vit_core = vit_base_patch16_224(fc_out=fc_out, embed_layer=AST_PatchEmbed)
        else:
            raise Exception('Model must be one of tiny224, small224, base224')

        self.original_num_patches = self.vit_core.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.vit_core.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, fc_out))

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
    

    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        if x.ndim < 4:
            x = x.unsqueeze(1)

        x = x.transpose(2, 3)

        x = self.vit_core.patch_embed(x)
        x = x + self.vit_core.pos_embed

        x = self.vit_core.patch_drop(x)
        x = self.vit_core.norm_pre(x)

        x = self.vit_core.blocks(x)
        x = self.vit_core.norm(x)

        # Gets the pre-logits scores out, to go through the new head here
        x = self.vit_core.forward_head(x, pre_logits=True)

        # Puts features through new head
        x = self.mlp_head(x)

        return x
    


###############################################################################
# AST CALL FUNCTION
###############################################################################
"""
AST takes an input of of either [batch, in_channels, input_tdim, n_mels] where 
    in_channels is in {1, 3}. 

Variant is a choice from {tiny224, small224, base224}
"""

def ast_tiny(fc_out, in_channels, input_tdim):
    return ASTModel(in_channels=in_channels, vit_core='tiny224', fc_out=fc_out, input_tdim=input_tdim)

def ast_small(fc_out, in_channels, input_tdim):
    return ASTModel(in_channels=in_channels, vit_core='small224', fc_out=fc_out, input_tdim=input_tdim)

def ast_base(fc_out, in_channels, input_tdim):
    return ASTModel(in_channels=in_channels, vit_core='base224', fc_out=fc_out, input_tdim=input_tdim)


###############################################################################
# EXAMPLE PASS THROUGH
###############################################################################

# input_tdim = 157
# in_channels = 1

# model = ast_tiny(fc_out=1000, in_channels=in_channels, input_tdim=input_tdim)

# test_input = torch.rand([10, in_channels, input_tdim, 128])

# test_output = model.forward(test_input)

# print(test_output.shape)
