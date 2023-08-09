"""
Implementation of the original proposed transformer adapter. Code taken and modified
    from:
        - https://github.com/rabeehk/compacter/blob/main/seq2seq/adapters/adapter_modeling.py
"""

################################################################################
# IMPORTS
################################################################################
import math
import torch
import torch.nn as nn

################################################################################
# CLASSIC ADAPTER
################################################################################

class BasicAdapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized.
    
    We modify the code to exclude config considerations and directly replace params"""

    def __init__(self, d_model):
        super().__init__()
        self.input_dim = d_model
        # Reduction factor is 32
        self.down_sample_size = self.input_dim // 32

        # Compacter paper used gelu activation
        self.activation = nn.GELU()
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size) 
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim) 

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output