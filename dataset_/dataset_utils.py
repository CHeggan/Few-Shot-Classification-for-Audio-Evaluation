"""
Contains a variety of helpful functions for the dataset side of things
"""
##############################################################################
# IMPORTS
##############################################################################
import sys
import torch
import torchaudio

##############################################################################
# VARIABLE LENGTH COLLATE FUNCTION
###############################################################################
def variable_collate(batch):
    """Dataloader collation function for variable length inputs 

    Args:
        batch (tuple): (x, y) where x has variable lengths between samples

    Returns:
        list: List of xs and ys. Contained in list as sample length agnostic
    """
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]

##############################################################################
# NORMALISATION/SCALING FUNCTIONS
##############################################################################
def nothing_func(x, extra_params=None):
    return x

def per_sample_scale(x):
    return (x- x.mean()) / x.std()

def given_stats_scale(x, mu, sigma):
    return (x - mu) / sigma


##############################################################################
# CIRCULAR PADDING AND ENFORCED LENGTH CROPPING FUNCTIONS
##############################################################################
def circular_padding_1d_v1(input, desired_length):
    """Padding function for data which uses rollback or circular behavior. The
        two different branches of this function are designed with speed in mind. 
        This is the reason for instance that branch 1 (ratio < 2) does not use 
        repeat and slice (it is slow and in-efficient for samples which only 
        need a few extra data points).

    Args:
        input (Tensor): The sample to be padded
        desired_length (int): Length to which the sample should be padded 

    Returns:
        Tensor: Final padded sample
    """
    # First calculate the length ratios ot decide what to do
    ratio = torch.tensor(desired_length / input.shape[1] )
    # If ratio is less than 2 (desired length < 2*current length), we simply
    #   replicate a beginning portion of the signal and append to end 
    if ratio < 2:
        diff = desired_length - input.shape[-1]
        to_add_to_end = input[:, 0:diff]
        final = torch.cat((input, to_add_to_end), axis=1)
    # If we have to multiply signal up my multiple times, we take a different 
    #   approach and simply repeat the signal up before trimming down
    else:
        num_repeats = int(torch.ceil(ratio))
        repeated = input.repeat((1, num_repeats))
        final = repeated[:, 0:desired_length]
    return final

def enforce_length(sample, desired_length):
    """Function to enforce that a sample is of specific length. This is done by 
        either padding (we opt for circular) or clipping, is CUDA friendly

    Args:
        sample (Tensor): Input sample to have its length enforced
        desired_length (int): Length sample should have by end of process

    Returns:
        Tensor: The newly length enforced sample
    """
    size_ratio = sample.shape[-1] / (desired_length)
    if size_ratio < 1:
        new_sample = circular_padding_1d_v1(sample, int(desired_length))
    elif size_ratio > 1:
        new_sample = sample[..., 0:int(desired_length)]
    else:
        new_sample = sample
    return new_sample


##############################################################################
# CONTRASTIVE TRANSFORMATIONS
##############################################################################
class ContrastiveTransformations(object):
    def __init__(self, base_transforms, n_views=2):
        """Applies transforms n_views times to input data. creates correlated
            views

        Args:
            base_transforms (Composed List): Transforms object that can x passed 
                directly through it
            n_views (int, optional): Number of unique views of x data to create.
                 Defaults to 2.
        """
        self.base_transforms = base_transforms
        self.n_views = n_views
        
    def __call__(self, x):
        # Had to add ['samples'] as newest version(not released) of torch-aug uses dicts
        return [self.base_transforms(x)['samples'] for i in range(self.n_views)]


################################################################################
# BATCH OF SIGNAL OT MEL-SPEC
################################################################################
def batch_to_log_mel_spec(samples, ft_params):
    transform = torchaudio.transforms.MelSpectrogram(**ft_params).to(samples.device)
    mel_specs = transform(samples)
    log_mel_specs = 20.0 / ft_params['power'] * torch.log10(mel_specs + sys.float_info.epsilon)
    #log_mel_specs = log_mel_specs.squeeze()
    return log_mel_specs

################################################################################
# BATCH OF SIGNAL TO MEL-SPEC + STFT
################################################################################
def batch_get_mag_phase(stft_data):
    for i in range(stft_data.shape[0]):
        sub_data = stft_data[i]
        phase = torch.atan2(sub_data[..., 1], sub_data[..., 0])
        magnitudes = torch.sqrt(torch.pow(sub_data[..., 0], 2) + torch.pow(sub_data[..., 1], 2))
        stft_data[i, ..., 0] = magnitudes
        stft_data[i, ..., 1] = phase
    return stft_data


def batch_to_log_mel_spec_plus_stft(samples, ft_params):
    # Do we have to change back dims at end of code, yay or nay
    # If we start with 4 dim input then yay
    og_shape = samples.shape
    need_to_change = False
    if samples.ndim == 4:
        need_to_change = True
        samples = samples.reshape(samples.shape[0]*samples.shape[1], samples.shape[-2], samples.shape[-1])

    mel_specs = batch_to_log_mel_spec(samples, ft_params)
    mel_specs = mel_specs.squeeze()

    # Need to remove channel dimension
    samples = samples.squeeze()

    stft_data = torch.stft(samples, n_fft=ft_params['n_fft'],
        hop_length=ft_params['hop_length'], return_complex=False)

    # Transform to project stft bins to mel scale (n bins)
    transform = torchaudio.transforms.MelScale(n_mels=ft_params['n_mels'], 
        sample_rate=ft_params['sample_rate'], n_stft=stft_data.shape[1]).to(samples.device)

    trans_real = transform(stft_data[..., 0])
    trans_img = transform(stft_data[..., 1])

    trans_stft_data = torch.stack((trans_real, trans_img), dim=-1)

    # Get the phase and magnitude of the stft data
    mag_phase = batch_get_mag_phase(trans_stft_data)
    magnitude = mag_phase[..., 0]
    phase = mag_phase[..., 1]

    # Converts amplitude spectrogram to log scale
    log_mel_scale_stft_amplitude = 20.0 / ft_params['power'] * torch.log10(magnitude + sys.float_info.epsilon)

    # Finally stacks the 3 channels of input data together along new channel di
    final_samples = torch.stack((mel_specs, log_mel_scale_stft_amplitude, phase), dim=1)

    if need_to_change:
        final_samples = final_samples.reshape(og_shape[0], og_shape[1], final_samples.shape[-3],
            final_samples.shape[-2], final_samples.shape[-1])

    return final_samples








# import matplotlib.pyplot as plt
# MEL_SPEC_PARAMS = {'sample_rate': 16000,
#                 'n_mels':128,
#                 'n_fft':1024,
#                 'hop_length':512,
#                 'power':2}

# # data = torch.rand(10, 1, 160000)

# data = torch.load('C:/Users/user/Documents/Datasets/FSD50k/FSD50k_train_val_pt_with_labels/63.pt')[0].to('cuda')
# # plt.plot(data[4][0])
# # plt.show()
# #data = torch.load('C:/Users/user/Documents/Datasets/AudioSet/AudioSet_Small_Example_Set_full_train_pt_w_labels/YXsa_RAELNR0.pt')[0]
# # data = data.repeat(10, 1)
# print(data.shape)
# final_data = batch_to_log_mel_spec_plus_stft(data, MEL_SPEC_PARAMS)
# final_data = final_data.cpu()
# for i in range(final_data.shape[0]):
#     # rows, columns, ordering
#     plt.subplot(1, 3, 1)
#     plt.imshow(final_data[i][0])
#     plt.colorbar()

#     plt.subplot(1, 3, 2)
#     plt.imshow(final_data[i][1])
#     plt.colorbar()

#     plt.subplot(1, 3, 3)
#     plt.imshow(final_data[i][2])
#     plt.colorbar()

#     plt.show()