###############################################################################
# IMPORTS
###############################################################################
import torch
import numpy as np

###############################################################################
# PREP BATCH FUNCTION GENERATOR
###############################################################################
class CreatePrepBatch():
    pass


# need a variable length conventional batcher - with a suitable collation function
    # would be nice to have a uniform collate and prep batch across both

def basic_flexible_prep_batch(device, data_type='float', input_size=None, variable=False):
    """Is the parent function for batch prepping. Returns an initialised function 

    Args:
        device (torch CUDA object): The CUDA device we want to load data to
        data_type (str, optional): The type to convert the input data to, can be 
            float or double
        input_size (int, optional): The expected size of input to the model. This allows us 
            to split up data and stack it if is needed. Kind of dynamic splitting
        variable (boolean, optional): Whether the data incoming is variable length 
            samples or not
    """
    def inner_prep_batch(batch, labels=True):
        """The child prep batch function. Takes some batch and processes it
            into proper tasks before moving it to a GPU for calculations.
        Depending on whether the data is variable length or not, batch data will
            come in as lists not tensors

        Args:
            batch (Tuple of lists or Tensors): The unformatted batch of data and tasks
            labels: (boolean): Whether or not we are actually using labels here or not

        Returns:
            Tensor, Tensor: The formatted x, y tensor pairs
        """
        x, y = batch

        # If variable length, we split up and stack on top of one another
        if variable:
            all_tensors = []
            lengths = []
            for sample_set in x:
                if sample_set.ndim < 3:
                    sample_set = sample_set.unsqueeze(1)
                # Track the number of sub-samples per clip
                lengths.append(sample_set.shape[0])
                all_tensors.append(sample_set)

            x = torch.cat(all_tensors)
            # If we are using labels, we do proper formatting
            if labels:
                y = torch.stack(y)
        
            if data_type == 'float':
                x = x.float().to(device)
                y = y.float().to(device)
            elif data_type == 'double':
                x = x.double().to(device)
                y = y.double().to(device)
            else:
                raise ValueError('data type not recognised')

            return x, y, lengths

        else:
            if data_type == 'float':
                x = x.float().to(device)
                y = y.float().to(device)
            elif data_type == 'double':
                x = x.double().to(device)
                y = y.double().to(device)
            else:
                raise ValueError('data type not recognised')

            return x, y
    return inner_prep_batch


###############################################################################
# FIXED LENGTH BATCHING FUNCTION (TRAIN AND EVAL)
###############################################################################
def prep_batch_fixed(n_way, k_shot, q_queries, device, trans):
    """Is the parent function for batch prepping. Returns an initialised function 

    Args:
        n_way (int)): The number of ways in classification task
        k_shot (int): Number of support vectors in a given task
        q_queries (int): Number of query vectors in a given task
        device (torch CUDA object): The CUDA device we want to load data to
        trans (boolean): Whether to apply transformer specific changes to data batching
    """
    def prep_batch_fixed(batch, meta_batch_size):
        """The child prep batch fucntion. Takes some batch and processes it
            into prper tasks before moving it to a GPU for calculations.

        Args:
            batch (Tensor): The unformatted batch of data and tasks
            meta_batch_size (int): The expected batch size

        Returns:
            Tensor, Tensor: The formatted x, y tensor pairs
        """
        x, y = batch

        x = x.squeeze()

        if x.ndim > 2:
            x = x.reshape(meta_batch_size, (n_way*k_shot + n_way*q_queries),
                                x.shape[-2], x.shape[-1])
        elif x.ndim == 2:
            x = x.reshape(meta_batch_size, (n_way*k_shot + n_way*q_queries),
                     x.shape[-1])

        # If transformer batch, we transpose and dont unsqueeze for channel 1
        if trans:
            x = torch.transpose(x, 2, 3)
            x = x.float().to(device)
        else:
            x = x.float().to(device)

        y_tr = torch.arange(0, n_way, 1/k_shot)
        y_val = torch.arange(0, n_way, 1/q_queries)
        y = torch.cat((y_tr, y_val))

        # Creates a batch dimension and then repeats across it
        y = y.unsqueeze(0).repeat(meta_batch_size, 1)

        y = y.long().to(device)

        return x, y
    return prep_batch_fixed

###############################################################################
# VARIABLE LENGTH EVAL BATCH FUNCTION (1D INPUT SIGNALS )
###############################################################################
def prep_var_eval_1d(n_way, k_shot, q_queries, device, trans):
    """Is the parent function for batch prepping. Returns an initialised function 

    Args:
        n_way (int)): The number of ways in classification task
        k_shot (int): Number of support vectors in a given task
        q_queries (int): Number of query vectors in a given task
        device (torch CUDA object): The CUDA device we want to load data to
        trans (boolean): Whether to apply transformer specific changes to data batching
    """
    def prep_var_eval(batch, meta_batch_size):
        """The child prep batch fucntion. Takes some batch and processes it
            into prper tasks before moving it to a GPU for calculations. Works
            for teh varibale length sets at eval/test time

        Args:
            batch (Tensor): The unformatted batch of data and tasks
            meta_batch_size (int): The expected batch size

        Returns:
            Tensor, Tensor: The formatted x, y tensor pairs
        """
        x, y = batch

        end_index = 0
        supports, queries = [], []
        for i in range(meta_batch_size):
            for idx, samples in enumerate( x[ end_index : end_index + (n_way*k_shot) ] ):
                supports.append(samples)
            end_index += n_way*k_shot
            for idx, samples in enumerate( x[ end_index : end_index + (n_way*q_queries) ] ):
                queries.append(samples)
            end_index += n_way*q_queries

        x_support = torch.zeros(meta_batch_size*(n_way*k_shot), x[0].shape[-1])
        for idx, samples in enumerate(supports):
            ind = np.random.choice(samples.shape[0])
            x_support[idx] = samples[ind]
        x_support = x_support.reshape(meta_batch_size, (n_way*k_shot), x[0].shape[-1])

        x_queries = torch.zeros(1, x[0].shape[-1])
        query_sample_nums = []
        for idx, samples in enumerate(queries):
            #print(samples.shape)
            query_sample_nums.append(samples.shape[0])
            for j, samp in enumerate(samples):
                if samp.ndim == 1:
                    samp = samp.unsqueeze(0)
                x_queries = torch.cat((x_queries, samp), 0)

        # Cuts the first sample as it was just zeros
        x_queries = x_queries[1:]

        # Generates and sets up the y value arary
        y_tr = torch.arange(0, n_way, 1/k_shot)
        y_val = torch.arange(0, n_way, 1/q_queries)
        y = torch.cat((y_tr, y_val))
        # Creates a batch dimension and then repeats across it
        y = y.unsqueeze(0).repeat(meta_batch_size, 1)

        # Changes data type and moves to passed device(CUDA)
        y = y.long().to(device)

        if trans:
            x_support = torch.transpose(x_support, 2, 3)
            x_queries = torch.transpose(x_queries, 1, 2)
            x_support = x_support.float().to(device)
            x_queries = x_queries.float().to(device)
        else:
            x_support = x_support.float().to(device)
            x_queries = x_queries.float().to(device)

        return x_support, x_queries, query_sample_nums, y
    return prep_var_eval