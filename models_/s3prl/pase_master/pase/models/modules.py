import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.distributions import Binomial
from torch.nn.utils.spectral_norm import spectral_norm
from torch.nn.utils.weight_norm import weight_norm
import numpy as np
import json
import os
try:
    from torchqrnn import QRNN
except ImportError:
    QRNN = None

def format_frontend_chunk(batch, device='cpu'):
    if type(batch) == dict:
        if 'chunk_ctxt' and 'chunk_rand' in batch:
            keys = ['chunk', 'chunk_ctxt', 'chunk_rand', 'cchunk']
            # cluster all 'chunk's, including possible 'cchunk'
            batches = [batch[k] for k in keys if k in batch]
            x = torch.cat(batches, dim=0).to(device)
            # store the number of batches condensed as format
            data_fmt = len(batches)
        else:
            x = batch['chunk'].to(device)
            data_fmt = 1
    else:
        x = batch
        data_fmt = 0
    return x, data_fmt

def format_frontend_output(y, data_fmt, mode):
    #assert data_fmt >= 0
    if data_fmt > 1:
        embedding = torch.chunk(y, data_fmt, dim=0)
        chunk = embedding[0]
        return embedding, chunk
    elif data_fmt == 1:
        chunk = embedding = y
        return embedding, chunk
    else:
        return select_output(y, mode=mode)

def build_rnn_block(in_size, rnn_size, rnn_layers, rnn_type,
                    bidirectional=True,
                    dropout=0, use_cuda=True):
    if (rnn_type.lower() == 'qrnn') and QRNN is not None:
        if bidirectional:
            print('WARNING: QRNN ignores bidirectional flag')
            rnn_size = 2 * rnn_size
        rnn = QRNN(in_size, rnn_size, rnn_layers, dropout=dropout, window=2,
                   use_cuda=use_cuda)
    elif rnn_type.lower() == 'lstm' or rnn_type.lower() == 'gru':
        rnn = getattr(nn, rnn_type.upper())(in_size, rnn_size, rnn_layers,
                                            dropout=dropout,
                                            bidirectional=bidirectional)
    else:
        raise TypeError('Unrecognized rnn type: ', rnn_type)
    return rnn
        
def select_output(h, mode=None):
    if mode == "avg_norm":
        return h - torch.mean(h, dim=2, keepdim=True)
    elif mode == "avg_concat":
        global_avg = torch.mean(h, dim=2, keepdim=True).repeat(1, 1, h.shape[-1])
        return torch.cat((h, global_avg), dim=1)
    elif mode == "avg_norm_concat":
        global_avg = torch.mean(h, dim=2, keepdim=True)
        h = h - global_avg
        global_feature = global_avg.repeat(1, 1, h.shape[-1])
        return torch.cat((h, global_feature), dim=1)
    else:
        return h


def build_norm_layer(norm_type, param=None, num_feats=None):
    if norm_type == 'bnorm':
        return nn.BatchNorm1d(num_feats)
    elif norm_type == 'snorm':
        spectral_norm(param)
        return None
    elif norm_type == 'bsnorm':
        spectral_norm(param)
        return nn.BatchNorm1d(num_feats)
    elif norm_type == 'lnorm':
        return nn.LayerNorm(num_feats)
    elif norm_type == 'wnorm':
        weight_norm(param)
        return None
    elif norm_type == 'inorm':
        return nn.InstanceNorm1d(num_feats, affine=False)
    elif norm_type == 'affinorm':
        return nn.InstanceNorm1d(num_feats, affine=True)
    elif norm_type is None:
        return None
    else:
        raise TypeError('Unrecognized norm type: ', norm_type)

def forward_norm(x, norm_layer):
    if norm_layer is not None:
        if isinstance(norm_layer, nn.LayerNorm):
            x = x.transpose(1, 2)
        x = norm_layer(x)
        if isinstance(norm_layer, nn.LayerNorm):
            x = x.transpose(1, 2)
        return x
    else:
        return x

def build_activation(activation, params, init=0):
    if activation == 'prelu' or activation is None:
        return nn.PReLU(params, init=init)
    if isinstance(activation, str):
        return getattr(nn, activation)()
    else:
        return activation

def forward_activation(activation, tensor):
    if activation == 'glu':
        # split tensor in two in channels dim
        z, g = torch.chunk(tensor, 2, dim=1)
        y = z * torch.sigmoid(g)
        return y
    else:
        return activation(tensor)

def get_padding(kwidth, dilation):
    return (kwidth // 2) * dilation

class NeuralBlock(nn.Module):

    def __init__(self, name='NeuralBlock'):
        super().__init__()
        self.name = name

	# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/5
    def describe_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        print('-' * 10)
        print(self)
        print('Num params: ', pp)
        print('-' * 10)
        return pp

class Saver(object):

    def __init__(self, model, save_path, max_ckpts=5, optimizer=None, prefix=''):
        self.model = model
        self.save_path = save_path
        self.ckpt_path = os.path.join(save_path, '{}checkpoints'.format(prefix)) 
        self.max_ckpts = max_ckpts
        self.optimizer = optimizer
        self.prefix = prefix

    def save(self, model_name, step, best_val=False):
        save_path = self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        ckpt_path = self.ckpt_path
        if os.path.exists(ckpt_path):
            with open(ckpt_path, 'r') as ckpt_f:
                # read latest checkpoints
                ckpts = json.load(ckpt_f)
        else:
            ckpts = {'latest':[], 'current':[]}

        model_path = '{}-{}.ckpt'.format(model_name, step)
        if best_val: 
            model_path = 'best_' + model_path
        model_path = '{}{}'.format(self.prefix, model_path)
        
        # get rid of oldest ckpt, with is the frst one in list
        latest = ckpts['latest']
        if len(latest) > 0:
            todel = latest[0]
            if self.max_ckpts is not None:
                if len(latest) > self.max_ckpts:
                    try:
                        print('Removing old ckpt {}'.format(os.path.join(save_path, 
                                                            'weights_' + todel)))
                        os.remove(os.path.join(save_path, 'weights_' + todel))
                        latest = latest[1:] 
                    except FileNotFoundError:
                        print('ERROR: ckpt is not there?')

        latest += [model_path]

        ckpts['latest'] = latest
        ckpts['current'] = model_path

        with open(ckpt_path, 'w') as ckpt_f:
            ckpt_f.write(json.dumps(ckpts, indent=2))

        st_dict = {'step':step,
                   'state_dict':self.model.state_dict()}

        if self.optimizer is not None: 
            st_dict['optimizer'] = self.optimizer.state_dict()
        # now actually save the model and its weights
        #torch.save(self.model, os.path.join(save_path, model_path))
        torch.save(st_dict, os.path.join(save_path, 
                                          'weights_' + \
                                           model_path))

    def read_latest_checkpoint(self):
        ckpt_path = self.ckpt_path
        print('Reading latest checkpoint from {}...'.format(ckpt_path))
        if not os.path.exists(ckpt_path):
            print('[!] No checkpoint found in {}'.format(self.save_path))
            return None
        else:
            with open(ckpt_path, 'r') as ckpt_f:
                ckpts = json.load(ckpt_f)
            curr_ckpt = ckpts['current'] 
            return curr_ckpt

    #def load(self):
    #    save_path = self.save_path
    #    ckpt_path = self.ckpt_path
    #    print('Reading latest checkpoint from {}...'.format(ckpt_path))
    #    if not os.path.exists(ckpt_path):
    #        raise FileNotFoundError('[!] Could not load model. Ckpt '
    #                                '{} does not exist!'.format(ckpt_path))
    #    with open(ckpt_path, 'r') as ckpt_f:
    #        ckpts = json.load(ckpt_f)
    #    curr_ckpt = ckpts['curent'] 
    #    st_dict = torch.load(os.path.join(save_path, curr_ckpt))
    #    return 

    def load_weights(self):
        save_path = self.save_path
        curr_ckpt = self.read_latest_checkpoint()
        if curr_ckpt is None:
            print('[!] No weights to be loaded')
            return False
        else:
            st_dict = torch.load(os.path.join(save_path,
                                              'weights_' + \
                                              curr_ckpt))
            if 'state_dict' in st_dict:
                # new saving mode
                model_state = st_dict['state_dict']
                self.model.load_state_dict(model_state)
                if self.optimizer is not None and 'optimizer' in st_dict:
                    self.optimizer.load_state_dict(st_dict['optimizer'])
            else:
                # legacy mode, only model was saved
                self.model.load_state_dict(st_dict)
            print('[*] Loaded weights')
            return True

    def load_ckpt_step(self, curr_ckpt):
        ckpt = torch.load(os.path.join(self.save_path,
                                       'weights_' + \
                                       curr_ckpt),
                          map_location='cpu')
        step = ckpt['step']
        return step

    def load_pretrained_ckpt(self, ckpt_file, load_last=False, load_opt=True,
                             verbose=True):
        model_dict = self.model.state_dict() 
        st_dict = torch.load(ckpt_file, 
                             map_location=lambda storage, loc: storage)
        if 'state_dict' in st_dict:
            pt_dict = st_dict['state_dict']
        else:
            # legacy mode
            pt_dict = st_dict
        all_pt_keys = list(pt_dict.keys())
        if not load_last:
            # Get rid of last layer params (fc output in D)
            allowed_keys = all_pt_keys[:-2]
        else:
            allowed_keys = all_pt_keys[:]
        # Filter unnecessary keys from loaded ones and those not existing
        pt_dict = {k: v for k, v in pt_dict.items() if k in model_dict and \
                   k in allowed_keys and v.size() == model_dict[k].size()}
        if verbose:
            print('Current Model keys: ', len(list(model_dict.keys())))
            print('Current Pt keys: ', len(list(pt_dict.keys())))
            print('Loading matching keys: ', list(pt_dict.keys()))
        if len(pt_dict.keys()) != len(model_dict.keys()):
            raise ValueError('WARNING: LOADING DIFFERENT NUM OF KEYS')
            print('WARNING: LOADING DIFFERENT NUM OF KEYS')
        # overwrite entries in existing dict
        model_dict.update(pt_dict)
        # load the new state dict
        self.model.load_state_dict(model_dict)
        for k in model_dict.keys():
            if k not in allowed_keys:
                print('WARNING: {} weights not loaded from pt ckpt'.format(k))
        if self.optimizer is not None and 'optimizer' in st_dict and load_opt:
            self.optimizer.load_state_dict(st_dict['optimizer'])


class Model(NeuralBlock):

    def __init__(self, max_ckpts=5, name='BaseModel'):
        super().__init__()
        self.name = name
        self.optim = None
        self.max_ckpts = max_ckpts

    def save(self, save_path, step, best_val=False, saver=None):
        model_name = self.name

        if not hasattr(self, 'saver') and saver is None:
            self.saver = Saver(self, save_path,
                               optimizer=self.optim,
                               prefix=model_name + '-',
                               max_ckpts=self.max_ckpts)

        if saver is None:
            self.saver.save(model_name, step, best_val=best_val)
        else:
            # save with specific saver
            saver.save(model_name, step, best_val=best_val)

    def load(self, save_path):
        if os.path.isdir(save_path):
            if not hasattr(self, 'saver'):
                self.saver = Saver(self, save_path, 
                                   optimizer=self.optim,
                                   prefix=self.name + '-',
                                   max_ckpts=self.max_ckpts)
            self.saver.load_weights()
        else:
            print('Loading ckpt from ckpt: ', save_path)
            # consider it as ckpt to load per-se
            self.load_pretrained(save_path)

    def load_pretrained(self, ckpt_path, load_last=False, verbose=True):
        # tmp saver
        saver = Saver(self, '.', optimizer=self.optim)
        saver.load_pretrained_ckpt(ckpt_path, load_last, verbose=verbose)


    def activation(self, name):
        return getattr(nn, name)()

    def parameters(self):
        return filter(lambda p: p.requires_grad, super().parameters())

    def get_total_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def describe_params(self):
        pp = 0
        if hasattr(self, 'blocks'):
            for b in self.blocks:
                p = b.describe_params()
                pp += p
        else:
            print('Warning: did not find a list of blocks...')
            print('Just printing all params calculation.')
        total_params = self.get_total_params()
        print('{} total params: {}'.format(self.name,
                                           total_params))
        return total_params


class GConv1DBlock(NeuralBlock):

    def __init__(self, ninp, fmaps,
                 kwidth, stride=1, norm_type=None,
                 act='prelu',
                 name='GConv1DBlock'):
        super().__init__(name=name)
        if act is not None and act == 'glu':
            Wfmaps = 2 * fmaps
        else:
            Wfmaps = fmaps
        self.conv = nn.Conv1d(ninp, Wfmaps, kwidth, stride=stride)
        self.norm = build_norm_layer(norm_type, self.conv, fmaps)
        self.act = build_activation(act, fmaps)
        self.kwidth = kwidth
        self.stride = stride

    def forward(self, x):
        if self.stride > 1 or self.kwidth % 2 == 0:
            P = (self.kwidth // 2 - 1,
                 self.kwidth // 2)
        else:
            P = (self.kwidth // 2,
                 self.kwidth // 2)
        x_p = F.pad(x, P, mode='reflect')
        h = self.conv(x_p)
        h = forward_activation(self.act, h)
        h = forward_norm(h, self.norm)
        #h = self.act(h)
        return h

class PatternedDropout(nn.Module):
    def __init__(self, emb_size, p=0.5, 
                 dropout_mode=['fixed_rand'],
                 ratio_fixed=None,
                 range_fixed=None,
                 drop_whole_channels=False):
        """Applies a fixed pattern of dropout for the whole training
        session (i.e applies different only among pre-specified dimensions)
        """
        super(PatternedDropout, self).__init__()

        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

        if self.p > 0:

            d_modes = ['std', 'fixed_rand', 'fixed_given']
            assert dropout_mode in d_modes, (
                "Expected dropout mode in {}, got {}".format(d_modes, dropout_mode)
            )
            self.drop_whole_channels = drop_whole_channels
            self.dropout_fixed = False

            if dropout_mode == 'fixed_rand':

                self.dropout_fixed = True
                assert ratio_fixed is not None, (
                    "{} needs 'ratio_fixed' arg set.".format(dropout_mode)
                )
                if ratio_fixed <= 0  or ratio_fixed > 1:
                    raise ValueError("{} mode needs 'ratio_fixed' to be"\
                                    " set and in (0, 1) range, got {}"\
                                    .format(dropout_mode, ratio_fixed))
                self.ratio_fixed = ratio_fixed
                self.dropped_dimsize = int(emb_size - emb_size*ratio_fixed)
                tot_idx = np.arange(emb_size)
                sel_idx = np.sort(np.random.choice(tot_idx, 
                                size=self.dropped_dimsize, replace=False))

            elif dropout_mode == 'fixed_given':

                self.dropout_fixed = True
                if range_fixed is None or \
                    not isinstance(range_fixed, str) or \
                    len(range_fixed.split(":")) < 2:
                    raise ValueError("{} mode needs 'range_dropped' to be"\
                                    " set (i.e. 10:20)".format(dropout_mode))
                rng = range_fixed.split(":")
                beg = int(rng[0])
                end = int(rng[1])
                assert beg < end and end <= emb_size, (
                    "Incorrect range {}".format(range_fixed)
                )
                self.dropped_dimsize = int(emb_size - (end -beg))
                tot_idx = np.arange(emb_size)
                fixed_idx = np.arange(beg, end, 1)
                sel_idx = np.setdiff1d(tot_idx, fixed_idx, assume_unique=True)

            if self.dropout_fixed:
                assert len(sel_idx) > 0, (
                    "Asked for fixed dropout, but sel_idx {}".format(sel_idx)
                )
                print ("Enabled dropout mode: {}. p={}, drop channels {}. Selected indices to apply dropout are: {}"\
                        .format(dropout_mode, self.p, drop_whole_channels, sel_idx))
                self.dindexes = torch.LongTensor(sel_idx)
                self.p = p
                self.p_scale = 1. / (1. - self.p)
            else:
                # notice, it is up to the user to make sure experiments between
                # fixed dropout and regular one are comparabe w.r.t p (i.e. for
                # fixed mode we only keep droping a subset of units (i.e. 50%),
                # thus p is effectively lower when compared to regular dropout)
                self.p =  p
                print ("Using std dropout with p={}".format(self.p))
        else:
            print ('Dropout at the inputs disabled, as p={}'.format(self.p))

    def forward(self, x):

        if self.p == 0 or not self.training:
            return x

        if self.dropout_fixed and self.training:
            self.dindexes = self.dindexes.to(x.device)
            assert len(x.size()) == 3, (
                "Expected to get 3 dimensional tensor, got {}"\
                   .format(len(x.size()))
            )
            bsize, emb_size, tsize = x.size()
            #print (bsize, esize, tsize)
            if self.drop_whole_channels:
                batch_mask = torch.full(size=(bsize, emb_size), fill_value=1.0, device=x.device)
                probs = torch.full(size=(bsize, self.dropped_dimsize),
                                  fill_value=1.-self.p)
                b = Binomial(total_count=1, probs=probs)
                mask = b.sample()
                mask = mask.to(x.device)
                batch_mask[:,self.dindexes] *= (mask * self.p_scale)
                #print ('mask dc', mask)
                #print ('maks dcv', mask.view(bsize, self.dropped_dimsize, -1))
                #x[:,self.dindexes,:] = x[:,self.dindexes,:].clone() * self.p_scale\
                #                         * mask.view(bsize, self.dropped_dimsize, -1)
                x = x * batch_mask.view(bsize, emb_size, -1)
            else:
                batch_mask = torch.ones_like(x, device=x.device)
                probs = torch.full(size=(bsize, self.dropped_dimsize, tsize), 
                                 fill_value=1.-self.p)
                b = Binomial(total_count=1, probs=probs)
                mask = b.sample()
                mask = mask.to(x.device)
                batch_mask[:,self.dindexes,:] *= (mask * self.p_scale)
                x = x * batch_mask
                #xx = x.data.clone()
                #x[:,self.dindexes,:] = x[:,self.dindexes,:].clone() * mask * self.p_scale
            return x
        else:
            return F.dropout(x, p=self.p, training=self.training)

class MLPBlock(NeuralBlock):

    def __init__(self, ninp, fmaps, din=0, dout=0, context=1, 
                 tie_context_weights=False, name='MLPBlock',
                 ratio_fixed=None, range_fixed=None, 
                 dropin_mode='std', drop_channels=False, emb_size=100):
        super().__init__(name=name)
        self.ninp = ninp
        self.fmaps = fmaps
        self.tie_context_weights = tie_context_weights
        assert context % 2 != 0, context
        if tie_context_weights:
            self.W = nn.Conv1d(ninp, fmaps, 1)
            self.pool = nn.AvgPool1d(kernel_size=context, stride=1,
                                      padding=context//2, count_include_pad=False)
        else:
            self.W = nn.Conv1d(ninp, fmaps, context, padding=context//2)

        self.din = PatternedDropout(emb_size=emb_size, p=din, 
                                    dropout_mode=dropin_mode,
                                    range_fixed=range_fixed,
                                    ratio_fixed=ratio_fixed,
                                    drop_whole_channels=drop_channels)
        self.act = nn.PReLU(fmaps)
        self.dout = nn.Dropout(dout)

    def forward(self, x, device=None):
        if self.tie_context_weights:
            return self.dout(self.act(self.pool(self.W(self.din(x)))))
        return self.dout(self.act(self.W(self.din(x))))

class GDeconv1DBlock(NeuralBlock):

    def __init__(self, ninp, fmaps,
                 kwidth, stride=4, norm_type=None,
                 act=None,
                 bias=True,
                 name='GDeconv1DBlock'):
        super().__init__(name=name)
        if act is not None and act == 'glu':
            Wfmaps = 2 * fmaps
        else:
            Wfmaps = fmaps
        pad = max(0, (stride - kwidth)//-2)
        self.deconv = nn.ConvTranspose1d(ninp, Wfmaps,
                                         kwidth, 
                                         stride=stride,
                                         padding=pad, 
                                         bias=bias)
        self.norm = build_norm_layer(norm_type, self.deconv,
                                     Wfmaps)
        self.act = build_activation(act, fmaps)
        self.kwidth = kwidth
        self.stride = stride

    def forward(self, x):
        h = self.deconv(x)
        if (self.stride % 2 != 0 and self.kwidth % 2 == 0) or \
           (self.stride % 2 == 0 and self.kwidth % 2 != 0): # and self.stride > self.kwidth:
            h = h[:, :, :-1]
        h = forward_norm(h, self.norm)
        h = forward_activation(self.act, h)
        return h

class ResBasicBlock1D(NeuralBlock):
    """ Adapted from
        https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    expansion = 1

    def __init__(self, inplanes, planes, kwidth=3, 
                 dilation=1, norm_layer=None, name='ResBasicBlock1D'):
        super().__init__(name=name)
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        # compute padding given dilation factor
        P  = (kwidth // 2) * dilation
        self.conv1 = nn.Conv1d(inplanes, planes, kwidth,
                               stride=1, padding=P,
                               bias=False, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kwidth,
                               padding=P, dilation=dilation,
                               bias=False)
        self.bn2 = norm_layer(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResDilatedModule(NeuralBlock):

    def __init__(self, ninp, fmaps,
                 res_fmaps,
                 kwidth, dilation,
                 norm_type=None,
                 act=None,
                 causal=False,
                 name='ResDilatedModule'):
        super().__init__(name=name)
        assert kwidth % 2 != 0
        self.causal = causal
        self.dil_conv = nn.Conv1d(ninp, fmaps,
                                  kwidth, dilation=dilation)
        if act is not None:
            self.act = getattr(nn, act)()
        else:
            self.act = nn.PReLU(fmaps, init=0)
        self.dil_norm = build_norm_layer(norm_type, self.dil_conv,
                                         fmaps)
        self.kwidth = kwidth
        self.dilation = dilation
        # skip 1x1 convolution
        self.conv_1x1_skip = nn.Conv1d(fmaps, ninp, 1)
        self.conv_1x1_skip_norm = build_norm_layer(norm_type, 
                                                   self.conv_1x1_skip,
                                                   ninp)
        # residual 1x1 convolution
        self.conv_1x1_res = nn.Conv1d(fmaps, res_fmaps, 1)
        self.conv_1x1_res_norm = build_norm_layer(norm_type, 
                                                  self.conv_1x1_res,
                                                  res_fmaps)

    def forward(self, x):
        if self.causal:
            kw__1 = self.kwidth - 1
            P = kw__1 + kw__1 * (self.dilation - 1)
            # causal padding
            x_p = F.pad(x, (P, 0))
        else:
            kw_2 = self.kwidth // 2
            P = kw_2 * self.dilation
            x_p = F.pad(x, (P, P))
        # dilated conv
        h = self.dil_conv(x_p)
        # normalization if applies
        h = forward_norm(h, self.dil_norm)
        # activation
        h = self.act(h)
        a = h
        # conv 1x1 to make residual connection
        h = self.conv_1x1_skip(h)
        # normalization if applies
        h = forward_norm(h, self.conv_1x1_skip_norm)
        # return with skip connection
        y = x + h
        # also return res connection (going to further net point directly)
        sh = self.conv_1x1_res(a)
        sh = forward_norm(sh, self.conv_1x1_res_norm)
        return y, sh

# SincNet conv layer
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def sinc(band,t_right, cuda=False):
    y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
    y_left= flip(y_right,0)

    ones = torch.ones(1)
    if cuda:
        ones = ones.to('cuda')
    y=torch.cat([y_left, ones, y_right])

    return y
    
    
# Modified from https://github.com/mravanelli/SincNet
class SincConv(nn.Module):

    def __init__(self, N_filt, Filt_dim, fs, stride=1,
                 padding='VALID', pad_mode='reflect'):
        super(SincConv, self).__init__()

        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) \
                                         / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, 
                                 N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10 ** (mel_points / 2595) - 1)) # Convert Mel to Hz
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (fs / 2) - 100
                
        self.freq_scale=fs * 1.0
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1/self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy((b2-b1)/self.freq_scale))

        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs
        self.padding = padding
        self.stride =stride
        self.pad_mode = pad_mode
        
    def forward(self, x):
        cuda = x.is_cuda
        filters=torch.zeros((self.N_filt, self.Filt_dim))
        N=self.Filt_dim
        t_right=torch.linspace(1, (N - 1) / 2, 
                               steps=int((N - 1) / 2)) / self.fs
        if cuda:
            filters = filters.to('cuda')
            t_right = t_right.to('cuda')
        
        min_freq=50.0;
        min_band=50.0;
        filt_beg_freq = torch.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (torch.abs(self.filt_band) + \
                                         min_band / self.freq_scale)
        n = torch.linspace(0, N, steps = N)
        # Filter window (hamming)
        window=(0.54 - 0.46 * torch.cos(2 * math.pi * n / N)).float()
        if cuda:
            window = window.to('cuda')
        for i in range(self.N_filt):
            low_pass1 = 2 * filt_beg_freq[i].float()* \
                    sinc(filt_beg_freq[i].float() * self.freq_scale, 
                         t_right, cuda)
            low_pass2 = 2 * filt_end_freq[i].float()* \
                    sinc(filt_end_freq[i].float() * self.freq_scale, 
                         t_right, cuda)
            band_pass=(low_pass2 - low_pass1)
            band_pass=band_pass/torch.max(band_pass)
            if cuda:
                band_pass = band_pass.to('cuda')

            filters[i,:]=band_pass * window
        if self.padding == 'SAME':
            if self.stride > 1:
                x_p = F.pad(x, (self.Filt_dim // 2 - 1,
                                self.Filt_dim // 2), mode=self.pad_mode)
            else:
                x_p = F.pad(x, (self.Filt_dim // 2,
                                self.Filt_dim // 2), mode=self.pad_mode)
        else:
            x_p = x
        out = F.conv1d(x_p, filters.view(self.N_filt, 1, self.Filt_dim),
                       stride=self.stride)
        return out

class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='VALID', pad_mode='reflect',
                 dilation=1, bias=False, groups=1,
                 sample_rate=16000, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.pad_mode = pad_mode
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)
        

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        # (kernel_size, 1)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

 


    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)
        
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
		# Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). 
        # I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_  
        band_pass_center = 2*band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])
        
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)

        
        band_pass = band_pass / (2*band[:,None])
        

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        x = waveforms 

        if self.padding == 'SAME':
            if self.stride > 1:
                x_p = F.pad(x, (self.kernel_size // 2 - 1,
                                self.kernel_size // 2), mode=self.pad_mode)
            else:
                x_p = F.pad(x, (self.kernel_size // 2,
                                self.kernel_size // 2), mode=self.pad_mode)
        else:
            x_p = x

        return F.conv1d(x_p, self.filters, stride=self.stride,
                        padding=0, dilation=self.dilation,
                        bias=None, groups=1) 

class FeResBlock(NeuralBlock):

    def __init__(self, num_inputs,
                 fmaps, kwidth, 
                 dilations=[1, 2],
                 downsample=1,
                 pad_mode='constant',
                 act=None,
                 norm_type=None,
                 name='FeResBlock'):
        super().__init__(name=name)
        if act is not None and act == 'glu':
            Wfmaps = 2 * fmaps
        else:
            Wfmaps = fmaps
        self.num_inputs = num_inputs
        self.fmaps = fmaps
        self.kwidth = kwidth
        downscale = 1. / downsample
        self.downscale = downscale
        # stride is ignored for no downsampling is
        # possible in FeResBlock
        self.stride = 1
        #self.dilation = dilation
        dilation = dilations[0]
        self.conv1 = nn.Conv1d(num_inputs,
                               Wfmaps, kwidth,
                               dilation=dilation,
                               padding=get_padding(kwidth, dilation))
        self.norm1 = build_norm_layer(norm_type,
                                      self.conv1,
                                      fmaps)
        self.act1 = build_activation(act, fmaps)
        dilation = dilations[1]
        self.conv2 = nn.Conv1d(fmaps, Wfmaps,
                               kwidth,
                               dilation=dilation,
                               padding=get_padding(kwidth, dilation))
        #assert self.norm2 is not None
        self.norm2 = build_norm_layer(norm_type,
                                      self.conv2,
                                      fmaps)
        self.act2 = build_activation(act, fmaps)
        if self.num_inputs != self.fmaps:
            # build projection layer
            self.resproj = nn.Conv1d(self.num_inputs,
                                     self.fmaps, 1)

    def forward(self, x):
        """
        # compute pad factor
        if self.kwidth % 2 == 0:
            if self.dilation > 1:
                raise ValueError('Not supported dilation with even kwdith')
            P = (self.kwidth // 2 - 1,
                 self.kwidth // 2)
        else:
            pad = (self.kwidth // 2) * (self.dilation - 1) + \
                    (self.kwidth // 2)
            P = (pad, pad)
        """
        identity = x
        #x = F.pad(x, P, mode=self.pad_mode)
        if self.downscale < 1:
            x = F.interpolate(x, scale_factor=self.downscale)
        x = self.conv1(x)
        x = forward_norm(x, self.norm1)
        x = forward_activation(self.act1, x)
        x = self.conv2(x)
        x = forward_activation(self.act2, x)
        if hasattr(self, 'resproj'):
            identity = self.resproj(identity)
        if self.downscale < 1:
            identity = F.interpolate(identity, scale_factor=self.downscale)
        x = x + identity
        x = forward_norm(x, self.norm2)
        return x

class FeBlock(NeuralBlock):

    def __init__(self, num_inputs,
                 fmaps, kwidth, stride,
                 dilation,
                 pad_mode='reflect',
                 act=None,
                 norm_type=None,
                 sincnet=False,
                 sr=16000,
                 name='FeBlock'):
        super().__init__(name=name)
        if act is not None and act == 'glu':
            Wfmaps = 2 * fmaps
        else:
            Wfmaps = fmaps
        self.num_inputs = num_inputs
        self.fmaps = fmaps
        self.kwidth = kwidth
        self.stride = stride
        self.dilation = dilation
        self.pad_mode = pad_mode
        self.sincnet = sincnet
        if sincnet:
            # only one-channel signal can be analyzed
            assert num_inputs == 1, num_inputs
            self.conv = SincConv_fast(1, Wfmaps,
                                      kwidth, 
                                      sample_rate=sr,
                                      padding='SAME',
                                      stride=stride,
                                      pad_mode=pad_mode)
        else:
            self.conv = nn.Conv1d(num_inputs,
                                  Wfmaps,
                                  kwidth,
                                  stride,
                                  dilation=dilation)
        if not (norm_type == 'snorm' and sincnet):
            self.norm = build_norm_layer(norm_type,
                                         self.conv,
                                         Wfmaps)
        self.act = build_activation(act, fmaps)

    def forward(self, x):
        if self.kwidth > 1 and not self.sincnet:
            # compute pad factor
            if self.stride > 1 or self.kwidth % 2 == 0:
                if self.dilation > 1:
                    raise ValueError('Cannot make dilated convolution with '
                                     'stride > 1')
                P = (self.kwidth // 2 - 1,
                     self.kwidth // 2)
            else:
                pad = (self.kwidth // 2) * (self.dilation - 1) + \
                        (self.kwidth // 2)
                P = (pad, pad)
            x = F.pad(x, P, mode=self.pad_mode)
        h = self.conv(x)
        if hasattr(self, 'norm'):
            h = forward_norm(h, self.norm)
        h = forward_activation(self.act, h)
        #h = self.act(h)
        return h


class VQEMA(nn.Module):
    """ VQ w/ Exp. Moving Averages,
        as in (https://arxiv.org/pdf/1711.00937.pdf A.1).
        Partly based on
        https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
    """
    def __init__(self, emb_K, emb_dim, beta,
                 gamma, eps=1e-5):
        super().__init__()
        self.emb_K = emb_K
        self.emb_dim = emb_dim
        self.emb = nn.Embedding(self.emb_K,
                                self.emb_dim)
        self.emb.weight.data.normal_()
        self.beta = beta
        self.gamma = gamma
        self.register_buffer('ema_cluster_size', torch.zeros(emb_K))
        self.ema_w = nn.Parameter(torch.Tensor(emb_K, emb_dim))
        self.ema_w.data.normal_()

        self.gamma = gamma
        self.eps = eps

    def forward(self, inputs):
        # convert inputs [B, F, T] -> [BxT, F]
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.emb_dim)
        device = 'cuda' if inputs.is_cuda else 'cpu'

        # TODO: UNDERSTAND THIS COMPUTATION
        # compute distances
        dist = (torch.sum(flat_input ** 2, dim=1, keepdim=True) + \
                torch.sum(self.emb.weight ** 2, dim=1) - \
                2 * torch.matmul(flat_input, self.emb.weight.t()))

        # Encoding
        enc_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        enc = torch.zeros(enc_indices.shape[0], self.emb_K).to(device)
        enc.scatter_(1, enc_indices, 1)
        
        # Use EMA to update emb vectors
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.gamma + \
                    (1 - self.gamma) * torch.sum(enc, 0)
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.eps) / \
                (n + self.emb_K * self.eps) * n
            )
            dw = torch.matmul(enc.t(), flat_input)
            self.ema_w = nn.Parameter(self.ema_w * self.gamma + \
                                      (1 - self.gamma) * dw)
            self.emb.weight = nn.Parameter(self.ema_w / \
                                           self.ema_cluster_size.unsqueeze(1))

        # Quantize and reshape
        Q = torch.matmul(enc, self.emb.weight).view(input_shape)

        # Loss 
        e_latent_loss = torch.mean((Q.detach() - inputs) ** 2)
        loss = self.beta * e_latent_loss

        Q = inputs + (Q - inputs).detach()
        avg_probs = torch.mean(enc, dim=0)
        # perplexity
        PP = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return loss, Q.permute(0, 2, 1).contiguous(), PP, enc

class SimpleResBlock1D(nn.Module):
    """ Based on WaveRNN a publicly available WaveRNN implementation:
        https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py
    """

    def __init__(self, dims):
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class MelResNet(nn.Module):
    """ Based on WaveRNN a publicly available WaveRNN implementation:
        https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py
    """

    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims, pad):
        super().__init__()
        k_size = pad * 2 + 1
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=k_size, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks):
            self.layers.append(SimpleResBlock1D(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers: x = f(x)
        x = self.conv_out(x)
        return x

class Stretch2d(nn.Module):
    """ Based on WaveRNN a publicly available WaveRNN implementation:
        https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py
    """

    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        # y_scale is feat dim, x_scale is time
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)

class UpsampleNetwork(nn.Module):
    """ Based on WaveRNN a publicly available WaveRNN implementation:
        https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py
    """

    def __init__(self, feat_dims, upsample_scales=[4, 4, 10], compute_dims=128,
                 res_blocks=10, res_out_dims=128, pad=2):
        super().__init__()
        self.num_outputs = res_out_dims
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims, pad)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1. / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers: m = f(m)
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)



if __name__ == '__main__':
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    # 800 samples @ 16kHz is 50ms
    T = 800
    # n = 20 z time-samples per frame
    n = 20
    zgen = ZGen(n, T // n, 
                z_amp=0.5)
    all_z = None
    for t in range(0, 200, 5):
        time_idx = torch.LongTensor([t])
        z_ten = zgen(time_idx)
        print(z_ten.size())
        z_ten = z_ten.squeeze()
        if all_z is None:
            all_z = z_ten
        else:
            all_z = np.concatenate((all_z, z_ten), axis=1)
    N = 20
    for k in range(N):
        plt.subplot(N, 1, k + 1)
        plt.plot(all_z[k, :], label=k)
        plt.ylabel(k)
    plt.show()

    # ResBlock
    resblock = ResBlock1D(40, 100, 5, dilation=8)
    print(resblock)
    z = z_ten.unsqueeze(0)
    print('Z size: ', z.size())
    y = resblock(z)
    print('Y size: ', y.size())

    x = torch.randn(1, 1, 16) 
    deconv = GDeconv1DBlock(1, 1, 31)
    y = deconv(x)
    print('x: {} -> y: {} deconv'.format(x.size(),
                                         y.size()))
    conv = GConv1DBlock(1, 1, 31, stride=4)
    z = conv(y)
    print('y: {} -> z: {} conv'.format(y.size(),
                                       z.size()))
    """
    #x = torch.randn(1, 1, 16384)
    #sincnet = SincConv(1024, 251, 16000, padding='SAME')
    #feblock = FeBlock(1, 100, 251, 1)
    #y = feblock(x)
    #print('y size: ', y.size())
    #vq = VQEMA(50, 100, 0.25, 0.99)
    #x = torch.randn(10, 100, 160)
    #_, Q, PP , _ = vq(x)
#    conv = SincConv_fast(1, 10,
#                         251, 
#                         sample_rate=16000,
#                         padding='SAME',
#                         stride=160)
    #conv = GConv1DBlock(1, 10, 21, 1)
    #conv = FeResBlock(1, 10, 3, 1, 1)
    #conv = ResDilatedModule(1, 50, 256, 3, 1024)
    #x = torch.randn(1, 1, 16000)
    #print(conv)
    #y, res = conv(x)
    #print(y.size())
    upnet = UpsampleNetwork(100, [4, 4, 10], 256, 10, 100, 2)
    print(upnet)
    x = torch.randn(1, 100, 200)
    z, y = upnet(x)
    print(x.shape, z.shape, y.shape)




