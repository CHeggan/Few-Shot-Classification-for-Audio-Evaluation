###############################################################################
# IMPORTS
###############################################################################
import torch
import numpy as np
from tqdm import tqdm

###############################################################################
# FEATURE EXTRACTOR
###############################################################################
class FeatureExtractor:
    def __init__(self, dataloader, model, prep_batch, additional_fn, variable):
        """FeatureExtract class takes in some flat dataloader (just meaning not 
            one which create tasks)along with relevant processing functions and 
            the model of interest. It then loops over the data object, storing
            extracted features from the model.

        Args:
            dataloader (iterable data object): pyTorch dataloader, expect it to 
                be flat (i.e generated plain batches of actual samples (no augs etc))
            model (nn module): Torch nn module through which we can pass the data
            prep_batch (func): An instantiated batching function, takes in batch 
                and returns a new prepared batch (moved to device etc). For this,
                we require a prep batch which keeps original labels (either numeric
                or string)
            additional_fn (func): An additional function maybe needed to process 
                the data, i.e raw to mel-spec
            variable (bool): Whether or not the input data is of variable length
        """
        self.model = model
        self.prep_batch = prep_batch
        self.dataloader = dataloader
        self.additional_fn = additional_fn

        self.variable = variable

    def generate(self, ft_params, verbose, extra_params={}):

        if verbose:
            loop = tqdm(total=len(self.dataloader), desc='Feat Gen')
        
        all_feats = []
        all_labels = []

    
        for idx, batch in enumerate(self.dataloader):

            if self.variable:
                x, y, lengths = self.prep_batch(batch)
            else:
                x, y = self.prep_batch(batch)

            # if x.shape[0] == 1:
            #     x = torch.concat((x, x))
            #     print(x.shape)
            x = self.additional_fn(x, ft_params=ft_params)

            feats = self.model.forward(x, **extra_params).detach().cpu()

            if self.variable:
                feat_list = list(torch.split(feats, lengths))
                all_feats = all_feats + feat_list

            else:
                all_feats.append(feats)

            all_labels.append(y)

            if verbose:
                loop.update(1)

        if self.variable == False:
            all_feats = torch.concat(all_feats, dim=0)
            print('fixed feats', all_feats.shape)

        all_labels = torch.concat(all_labels)

        return all_feats, all_labels
