
###############################################################################
# IMPORTS
###############################################################################
import torch
import random
import numpy as np
from collections import OrderedDict

###############################################################################
# SET SEED
###############################################################################
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

###############################################################################
# LOAD MODELS
###############################################################################
def load_backbone(model, path, verbose=False):
    """
    Loads some arbitrary models' state dict from path
    """
    state_dict = torch.load(path, map_location='cpu')

    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[2:] # Removes number at beginning 
    #     new_state_dict[name] = v

    model.load_state_dict(state_dict)
    if verbose:
        print(f'Successfully Loaded Model from: {path}')
    return model




