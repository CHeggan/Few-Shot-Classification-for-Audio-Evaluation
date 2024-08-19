"""
Script to load all ssl audio models. SThese come from a variety of sources:
    -> From the normal and functioning parts of s3prl 
        - baseline with normal features
        - wav2vec
        - wav2vec2
        - hubert
        - decoar
        - decoar2

    -> Some work arounds for s3prl
        - mocking jay

    -> From their original github repos
        - pase 

Quite a lot of modules and packages have to be installed for this to work so this 
    script should go along with a requirements file.
"""

"""
Due to the diversity here, it is actually impossible to run all of these models 
    off of a single environment.

More specifically, we require 3 unique envs to cover this selection of models:
    -> Most models can be run with a cuda enabled s3prl env
    -> PASE+ requires a cuda non-enables env with QRNN module
    -> SSAST requires an env with timm=0.4.5

These specifications unfortunately mean that imports have to be function 
    wrapped to avoid constant manual intervention.
"""
################################################################################
# IMPORTS
################################################################################
import os
import sys

import torch

################################################################################
# COUNT MODEL PARAMS
################################################################################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

################################################################################
# MODEL LOADING
################################################################################

#####################################
# MODIFIED FORWARD FUNCTION
#####################################

def s3prl_forward(model, x):
    x = x.squeeze()
    x = model(x)
    x = x['last_hidden_state']
    # Average over time dim
    x = x.mean(dim=1)
    return x

def pase_forward(model, x):
    x = model(x)
    # Average over time dim
    x = x.mean(dim=2)
    return x

def wavlm_forward(model, x):
    x = x.squeeze()
    x = model.extract_features(x)[0]
    # Average over time dim
    x = x.mean(dim=1)
    return x

def ssast_forward(model, x):
    x = x.squeeze()
    # x = model(x, task='ft_cls')
    x = model(x, task='ft_avgtok')
    # x = model(x, task='pretrain_mpc')
    
    return x 

#####################################
# GET MODELS USING S3PRL DIRECTLY
#####################################
def get_wav2vec():
    import s3prl.hub as hub
    model = getattr(hub, 'wav2vec')()
    return model

def get_wav2vec2():
    import s3prl.hub as hub
    model = getattr(hub, 'wav2vec2')()
    return model

def get_hubert():
    import s3prl.hub as hub
    model = getattr(hub, 'hubert')()
    return model

def get_decoar():
    import s3prl.hub as hub
    model = getattr(hub, 'decoar')()
    return model

def get_decoar2():
    import s3prl.hub as hub
    model = getattr(hub, 'decoar2')()
    return model


#####################################
# HACKED S3PRL MODELS + OTHERS
#####################################
def get_mockingjay():
    from s3prl.upstream.mockingjay.hubconf import mockingjay_local

    model = mockingjay_local(ckpt="models_/s3prl/checkpoints/mj_960.ckpt")
    return model


def get_paseplus():
    from pase.models.frontend import wf_builder
    model = wf_builder('models_/s3prl/configs/PASE+.cfg').eval()
    model.load_pretrained('models_/pase_master/pase.ckpt', load_last=True, verbose=True)
    # data = torch.randn(10, 1, 160000)
    # out = model(data)
    # print(out.shape)
    return model 


def get_wavlm():
    from models_.wavlm.WavLM import WavLM, WavLMConfig
    checkpoint = torch.load('models_/s3prl/checkpoints/WavLM-Base.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    return model


def get_ssast(t_dim, f_dim=128):
    from models_.ssast.ast_models import ASTModel
    model = ASTModel(label_dim=527,
         fshape=16, tshape=16,fstride=10, tstride=10,
         input_fdim=f_dim, input_tdim=t_dim, model_size='base',
         pretrain_stage=False, load_pretrained_mdl_path='models_/s3prl/checkpoints/audio_model_librispeech.pth').cuda()
    return model


def get_tera():
    from s3prl.upstream.tera.hubconf import tera_local
    model = tera_local(ckpt='models_/s3prl/checkpoints/tera_logMelBase_T_F_M_AdamW_b32_1m_960hr_drop1.ckpt')
    return model


def get_npc():
    from s3prl.upstream.npc.hubconf import npc_local
    model = npc_local(ckpt='models_/s3prl/checkpoints/npc_960hr.ckpt')
    return model


def get_vqwav2vec():
    from s3prl.upstream.vq_wav2vec.hubconf import wav2vec2_local
    model = wav2vec2_local(ckpt='models_/s3prl/checkpoints/vq-wav2vec.pt')
    return model


def get_distilhubert():
    from s3prl.upstream.distiller.hubconf import distiller_local
    model = distiller_local(ckpt='models_/s3prl/checkpoints/disilhubert_ls960_4-8-12.ckpt')
    return model


def get_apc():
    from s3prl.upstream.apc.hubconf import apc_local
    model = apc_local(ckpt='models_/s3prl/checkpoints/apc_960hr.ckpt')
    return model 


def get_vqapc():
    from s3prl.upstream.apc.hubconf import apc_local
    model = apc_local(ckpt='models_/s3prl/checkpoints/vq_apc_960hr.ckpt')
    return model 

#####################################
# MODEL SELECTION
#####################################
def ssl_model_select(name, t_dim=157):
    # Unless another env is recommended, models require s3prl based env
    if name == 'wav2vec':
        return get_wav2vec()
    elif name == 'wav2vec2':
        return get_wav2vec2()
    elif name == 'hubert':
        return get_hubert()
    elif name == 'decoar':
        return get_decoar()
    elif name == 'decoar2':
        return get_decoar2()
    elif name == 'mockingjay':
        return get_mockingjay()
    
    
    elif name == 'tera':
        return get_tera()
    elif name == 'npc':
        return get_npc()
    elif name == 'vqwav2vec':
        return get_vqwav2vec()
    elif name == 'distilhubert':
        return get_distilhubert()
    elif name == 'apc':
        return get_apc()
    elif name == 'vqapc':
        return get_vqapc()
    
    # PASE+ requires specialised env with no cuda due to qrnn module
    elif name == 'paseplus':
        return get_paseplus()
    elif name == 'wavlm':
        return get_wavlm()
    # SSAST requires specialised env with timm==0.4.5
    elif name == 'ssast':
        return get_ssast(t_dim)
    else:
        raise ValueError(f'Model name "{name}" not recognised')


################################################################################
# MODEL LOAD TESTING
################################################################################
# get_paseplus()
# model = get_mockingjay().cuda()

# data = torch.rand(size=(1, 80000)).cuda()

# out = s3prl_forward(model, data)
# print(out.shape)

# model_funcs = [get_paseplus, get_wavlm, get_wav2vec, get_wav2vec2, get_hubert, get_decoar, get_decoar2, get_mockingjay]

# for get_func in model_funcs:
#     print(get_func)
#     model = get_func().eval()
#     device='cpu'

#     if get_func.__name__ == 'get_paseplus':
#         print('yo')
#         model = model.to(device)
#         # x = torch.randn(size=(10, 1, 160000), dtype=torch.float).to(device)
#         # out = model(x)

#     elif get_func.__name__ == 'get_wavlm':
#         model = model.to(device)

#     else:
#         x = torch.randn(size=(10, 160000), dtype=torch.float).to(device)

#         out = s3prl_forward(model, x)
#     # print(out.shape)


#     print(get_func, f'Params:{count_parameters(model)}')



# model = ssl_model_select('ssast', t_dim=312).cuda()
# data = torch.rand((10, 128, 312)).cuda()
# print(data.shape)

# print(count_parameters(model))