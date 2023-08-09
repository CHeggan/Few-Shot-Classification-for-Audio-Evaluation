"""
Vision transformers with adapters for multi-task learning.

Base of model Code from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

Implements following adapter variants:
    - AdaptFormer (https://github.com/ShoufaChen/AdaptFormer/blob/main/models/adapter.py)
    - OG Adapter (https://github.com/rabeehk/compacter/tree/main/seq2seq/adapters)
    - Compacter (https://github.com/rabeehk/compacter/tree/main/seq2seq/adapters)

Paper references can be found from code links. 
"""

################################################################################
# IMPORTS
################################################################################
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.functional as F

from functools import partial
# layers only appears in higher level packages (pip install timm==0.8.23.dev0)
from timm.layers import use_fused_attn, Mlp, DropPath, PatchEmbed, PatchDropout, \
    trunc_normal_

from .model_utils import count_parameters
from .adapters.AdaptFormer import AdapterFormer
from .adapters.BasicAdapter import BasicAdapter
from .adapters.Compacter import HyperComplexAdapter


################################################################################
# VISION TRANSFORMER PARTS
################################################################################

########################################
# ATTENTION
########################################
class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

########################################
# LAYER SCALING
########################################
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

########################################
# BASIC BLOCK
########################################
class AdapterBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            adapter_type, # Type of adapter we want to apply
            num_tasks, # NUmber of tasks/adapters to deploy within the model
            mlp_ratio=4,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):

        super().__init__()

        self.norm1 = norm_layer(dim)

        self.adapter_type = adapter_type

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # DropPath drops an entire sample from the batch, which results in 'stochastic depth'
        # When using with residual connections like here, it is effectively a residual dropout, 
        #   where output is then just x instead of x + residual
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # AdaptFormer adapter setup: Only one module after mlp
        if adapter_type in ['adaptformer_series', 'adaptformer_parallel']:
            self.adaptmlp = nn.ModuleList([AdapterFormer(d_model=dim) for i in range(num_tasks)])

        # Original Adapter setup: One after attn and one after mlp
        elif adapter_type == 'og_adapter':
            self.after_attn = nn.ModuleList([BasicAdapter(d_model=dim) for i in range(num_tasks)])
            self.after_mlp = nn.ModuleList([BasicAdapter(d_model=dim) for i in range(num_tasks)])
    
        else:
            raise Exception('Adapter selection not recognised')

    def forward(self, x, task_int):
        # Path through norm, attn, scale
        x_res = self.norm1(x)
        x_res = self.attn(x_res)

        # Adapter after the attention layer, ends up: x + attn_x + adapt_x
        if self.adapter_type == 'og_adapter':
            adapt_x = self.after_attn[task_int](x)
            x = x + adapt_x


        # Merge path after attention
        x = x + self.ls1(self.drop_path1(x_res))
        


        x_res = self.norm2(x)
        x_res = self.mlp(x_res)


        # Add in adaptformer parallel residual
        if self.adapter_type == 'adaptformer_parallel':
            adapt_x = self.adaptmlp[task_int](x, residual=False)
            x = x + adapt_x

        # Add in adaptformer series adapter
        if self.adapter_type == 'adaptformer_series':
            x_res = self.adaptmlp[task_int](x_res, residual=False)

        elif self.adapter_type == 'og_adapter':
            adapt_x = self.after_mlp[task_int](x)
            x = x + adapt_x

        

        # Final merge path after mlp
        x = x + self.ls2(self.drop_path2(x_res))

        return x


################################################################################
# MAIN VISION TRANSFORMER CLASS (ViT)
################################################################################
class AdapterVisionTransformer(nn.Module):
    def __init__(
            self,
            out_dim_list, 
            num_tasks,
            adapter_type, 
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            global_pool: str = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[Callable] = None,
            act_layer: Optional[Callable] = None,
            block_fn: Callable = AdapterBlock,
            mlp_layer: Callable = Mlp,
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.

            out_dim_list: List of final linear layer output sizes
            num_tasks: Number of tasks we are working with, the same number of adpaters per spot are loaded
            adapter_type: Options are adaptformer_series, adaptformer_parallel and og_adapter

            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """

        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # A list of fully connected output head sizes
        self.num_outs = len(out_dim_list)

        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                num_tasks=num_tasks, # number of adapter tasks
                adapter_type=adapter_type, # type of adpater to use
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)])
    

        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)

        self.fc_layers = nn.ModuleList()
        # Create fully connected layers
        for out_dim in out_dim_list:
            self.fc_layers.append(  nn.Linear(self.embed_dim, out_dim) )

        # Handles error where we have num tasks != num outputs
        if len(out_dim_list) != num_tasks:
            if len(out_dim_list) != 1:
                raise ValueError('Different number of output dims compared to number of specificed tasks')
            

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)


    def forward_features(self, x, task_int):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        # Have to pass through blocks individually due to sequential/forward argument issue
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, task_int=task_int)

        x = self.norm(x)
        return x

    def split_forward_head(self, x, task_int, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)

        if pre_logits:
            return x 
        
        else:
            x = self.fc_layers[task_int](x)
            return x


    def forward(self, x, task_int):
        # didn't implement an 'all' option for task int here as it likely isn't useful 
        # If ever needed, can take code snippet from AdapterResNet
        if task_int == 'all':
            raise Exception('No all function currently implemented for main feats')
        
        x = self.forward_features(x, task_int)
        x = self.split_forward_head(x, task_int=task_int)

        return x



################################################################################
# VIT CALL FUNCTIONS
################################################################################
"""
All vision transformers take a standard 3 x 224 x 224 input
"""
def adapter_vit_tiny_patch16_224(fc_out, num_tasks, adapter_type, embed_layer=PatchEmbed):
    return AdapterVisionTransformer(out_dim_list=fc_out, num_tasks=num_tasks, adapter_type=adapter_type, 
                                    patch_size=16, embed_dim=192, depth=12, num_heads=3, embed_layer=embed_layer)

def adapter_vit_small_patch16_224(fc_out, num_tasks, adapter_type, embed_layer=PatchEmbed):
    return AdapterVisionTransformer(out_dim_list=fc_out, num_tasks=num_tasks, adapter_type=adapter_type,
                                     patch_size=16, embed_dim=384, depth=12, num_heads=6, embed_layer=embed_layer)

def adapter_vit_base_patch16_224(fc_out, num_tasks, adapter_type, embed_layer=PatchEmbed):
    return AdapterVisionTransformer(out_dim_list=fc_out, num_tasks=num_tasks, adapter_type=adapter_type,
                                     patch_size=16, embed_dim=768, depth=12, num_heads=12, embed_layer=embed_layer)



# ################################################################################
# # EXAMPLE DATA THROUGH
# ################################################################################
# data_batch = torch.rand(size=(100, 3, 224, 224)).to('cuda')

# model = vit_tiny_patch16_224(1000).to('cuda')

# print(model)
# print(count_parameters(model))

# data = model.forward(data_batch)

# print(data.shape)


# data_batch = torch.rand(size=(100, 3, 224, 224)).to('cuda')
# model = adapter_vit_small_patch16_224([500, 500], num_tasks=2, adapter_type='adaptformer_series').to('cuda')
# print(model.forward(data_batch, 1).shape)

# print(count_parameters(model))

