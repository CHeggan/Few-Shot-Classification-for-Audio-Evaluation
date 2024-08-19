from .minions import *
from ..losses import *
from ..utils import AuxiliarSuperviser, get_grad_norms
from ..log import *
#from tensorboardX import SummaryWriter
import soundfile as sf
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import random
import timeit
import os


class Waveminionet(Model):

    def __init__(self, frontend=None, frontend_cfg=None,
                 minions_cfg=None, z_minion=True,
                 z_cfg=None, adv_loss='BCE',
                 num_devices=1, pretrained_ckpts=None,
                 name='Waveminionet'):
        super().__init__(name=name)
        # augmented wav processing net
        # it trains simultaneously with many tasks
        # forcing a hierarchy of abstraction to distill # the contents within waveforms 
        if minions_cfg is None or len(minions_cfg) < 1:
            raise ValueError('Please specify a stack of minions'
                             ' config with at least 1 minion. '
                             'GIMME SOMETHING TO DO.')
        if frontend is not None:
            self.frontend = frontend
        else:
            if frontend_cfg is None:
                # default params
                self.frontend = WaveFe()
            else:
                self.frontend = WaveFe(**frontend_cfg)
        if self.frontend.quantizer is not None:
            self.vq = True
        else:
            self.vq = False
        # -------- MINION STACK --------
        self.minions = nn.ModuleList()
        self.mi_fwd = False
        ninp = self.frontend.emb_dim
        self.min2idx = {}
        for minion_cfg in minions_cfg:
            if 'mi' in minion_cfg['name'] and not self.mi_fwd:
                # add additional code for pair (either CMI or MI)
                # (just once, thus use mi_fwd flag)
                ninp += self.frontend.emb_dim
            minion_cfg['num_inputs'] = ninp
            minion = minion_maker(minion_cfg)
            self.minions.append(minion)
            self.min2idx[minion.name] = len(self.min2idx) 
            if hasattr(minion, 'skip') and minion.skip:
                nouts = minion.hidden_size
                # acumulate num of inputs (concat skip connection)
                ninp += nouts
            if 'mi' in minion.name:
                # if MI minion is present, multi chunk forward
                # is needed (3 chunks are fwd)
                self.mi_fwd = True
        if z_minion:
            # Make the minion enforcing the shape of the latent space
            # to be like some prior z_gen enforced in the loss
            # This minion is disconnected from others, just enforcing
            # frontend's output to follow Z, but no skip,
            # and it always backprops even in random backprop selection
            # as it acts as a regularizer
            if z_cfg is None:
                z_cfg = {
                    'num_inputs':self.frontend.emb_dim,
                    'num_outputs':1,
                    'hidden_layers':3,
                    'hidden_size':1024,
                    'norm_type':'bnorm',
                    'dropout':0.,
                    'kwidths':[31,11,5],
                    'name':'z',
                    'grad_reverse':False,
                    'skip':False
                }
            self.z_cfg = z_cfg
            self.z_adv_loss = adv_loss
            #self.z_minion = minion_maker(z_cfg)
            #self.z_minion.loss.register_DNet(self.z_minion)
        if pretrained_ckpts is not None:
            self.load_checkpoints(pretrained_ckpts)
        if num_devices > 1:
            self.frontend_dp = nn.DataParallel(self.frontend)
            self.minions_dp = nn.ModuleList([nn.DataParallel(m) for m in \
                                             self.minions])

    def build_z_minion(self, cfg):
        print('Built regularizer Z minion')
        device = 'cuda' if next(self.parameters()).is_cuda else 'cpu'
        self.z_cfg['loss'] = ZAdversarialLoss(loss=self.z_adv_loss,
                                              batch_acum=cfg['batch_acum'])
        # Build the regularizer Z minion
        self.z_minion = minion_maker(self.z_cfg)
        self.z_minion.loss.register_DNet(self.z_minion)
        self.z_minion.to(device)

    def forward(self, x):
        raise NotImplementedError
        fe_h = self.frontend(x)
        #print('front-end inference: ', fe_h.size())
        h = fe_h
        outs = {}
        for mi, minion in enumerate(self.minions, start=1):
            y, h_ = minion(h)
            if minion.skip:
                h_c = torch.cat((h, h_), dim=1)
                h = h_c
            else:
                h = h
            outs[minion.name] = y
        return outs, h

    def join_skip(self, x, skip):
        if skip is None:
            return x
        else:
            return torch.cat((x, skip), dim=1)

    def load_checkpoints(self, load_path):
        # create each savers first for all net components
        savers = [Saver(self.frontend, load_path, 
                        prefix='PASE-')]
        if hasattr(self, 'z_minion'):
            savers.append(Saver(self.z_minion, load_path,
                           prefix='Zminion-'))
        for mi, minion in enumerate(self.minions, start=1):
            savers.append(Saver(minion, load_path, 
                                prefix='M-{}-'.format(minion.name)))
        # now load each ckpt found
        giters = 0
        for saver in savers:
            # try loading all savers last state if not forbidden is active
            try:
                state = saver.read_latest_checkpoint()
                giter_ = saver.load_ckpt_step(state)
                print('giter_ found: ', giter_)
                # assert all ckpts happened at last same step
                if giters == 0:
                    giters = giter_
                else:
                    assert giters == giter_, giter_
                saver.load_pretrained_ckpt(os.path.join(load_path,
                                                        'weights_' + state), 
                                           load_last=True)
            except TypeError:
                break

    def forward_chunk(self, frontend, batch, chunk_name, device):
        if self.vq:
            vq_loss, fe_Q, \
            vq_pp, vq_idx = frontend(batch[chunk_name].to(device))
            return fe_Q
        else:
            return frontend(batch[chunk_name].to(device))

    def train_(self, dloader, cfg, device='cpu', va_dloader=None):
        epoch = cfg['epoch']
        bsize = cfg['batch_size']
        batch_acum = cfg['batch_acum']
        save_path = cfg['save_path']
        log_freq = cfg['log_freq']
        sup_freq = cfg['sup_freq']
        grad_keys = cfg['log_grad_keys']
        if cfg['sup_exec'] is not None:
            aux_save_path = os.path.join(cfg['save_path'],
                                         'sup_aux')
            if not os.path.exists(aux_save_path):
                os.makedirs(aux_save_path)
            self.aux_sup = AuxiliarSuperviser(cfg['sup_exec'], aux_save_path)
        # Adversarial auto-encoder hyperparams
        warmup_epoch = cfg['warmup']
        zinit_weight = cfg['zinit_weight']
        zinc = cfg['zinc']
        zweight = 0
        if hasattr(self, 'frontend_dp'):
            frontend = self.frontend_dp
        else:
            frontend = self.frontend
        # Build the regularizer minion
        self.build_z_minion(cfg)
        # Make the log writer(s)
        writer = LogWriter(save_path, log_types=cfg['log_types'])
        bpe = cfg['bpe'] if 'bpe' in cfg else len(dloader)
        print('=' * 50)
        print('Beginning training...')
        print('Batches per epoch: ', bpe)
        # rndmin_train flag means we donly backprop one minion path        
        # per batch update, selecting the minion randomly
        rndmin_train = cfg['rndmin_train']
        print('Randomized minion training: ', rndmin_train)
        feopt = getattr(optim, cfg['fe_opt'])(self.frontend.parameters(), 
                                              lr=cfg['fe_lr'])
        # Make the saver array. Each one will refer to one model. Init
        # with frontend model and optimizer
        savers = [Saver(self.frontend, save_path, 
                        max_ckpts=cfg['max_ckpts'],
                        optimizer=feopt, prefix='PASE-')]
        lrdecay = cfg['lrdecay']
        if lrdecay > 0:
            fesched = optim.lr_scheduler.StepLR(feopt,
                                                step_size=cfg['lrdec_step'],
                                                gamma=cfg['lrdecay'])
            #fesched = optim.lr_scheduler.ReduceLROnPlateau(feopt,
            #                                               mode='min',
            #                                               factor=lrdecay,
            #                                               verbose=True)
        if hasattr(self, 'z_minion'):
            z_lr = cfg['z_lr']
            zopt = getattr(optim, cfg['min_opt'])(self.z_minion.parameters(), 
                                                  lr=z_lr)
            if lrdecay > 0:
                #zsched = optim.lr_scheduler.ReduceLROnPlateau(zopt,
                #                                              mode='min',
                #                                              factor=lrdecay,
                #                                              verbose=True)
                zsched = optim.lr_scheduler.StepLR(zopt,
                                                   step_size=cfg['lrdec_step'],
                                                   gamma=cfg['lrdecay'])
            savers.append(Saver(self.z_minion,
                                save_path, max_ckpts=cfg['max_ckpts'],
                                optimizer=zopt, prefix='Zminion-'))

        # print model components
        print(self)
        if 'min_lrs' in cfg:
            min_lrs = cfg['min_lrs']
        else:
            min_lrs = None
        minopts = {}
        minscheds = {}
        for mi, minion in enumerate(self.minions, start=1):
            min_opt = cfg['min_opt']
            min_lr = cfg['min_lr']
            if min_lrs is not None and minion.name in min_lrs:
                min_lr = min_lrs[minion.name]
                print('Applying lr {:.5f} to minion {}'.format(min_lr,
                                                               minion.name))
            minopts[minion.name] = getattr(optim, min_opt)(minion.parameters(),
                                                           lr=min_lr)
            if lrdecay > 0:
                #minsched = lr_scheduler.ReduceLROnPlateau(minopts[minion.name],
                #                                          mode='min',
                #                                          factor=lrdecay,
                #                                          verbose=True)
                minsched = lr_scheduler.StepLR(minopts[minion.name],
                                               step_size=cfg['lrdec_step'],
                                               gamma=cfg['lrdecay'])
                minscheds[minion.name] = minsched
            savers.append(Saver(minion, save_path, max_ckpts=cfg['max_ckpts'],
                                optimizer=minopts[minion.name],
                                prefix='M-{}-'.format(minion.name)))

        minions_run = self.minions
        if hasattr(self, 'minions_dp'):
            minions_run = self.minions_dp

        min_global_steps = {}
        if cfg['ckpt_continue']:
            giters = 0
            for saver in savers:
                # try loading all savers last state if not forbidden is active
                try:
                    state = saver.read_latest_checkpoint()
                    giter_ = saver.load_ckpt_step(state)
                    print('giter_ found: ', giter_)
                    # assert all ckpts happened at last same step
                    if giters == 0:
                        giters = giter_
                    else:
                        assert giters == giter_, giter_
                    saver.load_pretrained_ckpt(os.path.join(save_path,
                                                            'weights_' + state), 
                                               load_last=True)
                except TypeError:
                    break

            global_step = giters
            # redefine num epochs depending on where we left it
            epoch_beg = int(global_step / bpe)
            epoch = epoch - epoch_beg
        else:
            epoch_beg = 0
            global_step = 0

        z_losses = None
        print('Beginning step of training: ', global_step)
        print('Looping for {} epochs: '.format(epoch))

        for epoch_ in range(epoch_beg, epoch_beg + epoch):
            self.train()
            timings = []
            beg_t = timeit.default_timer()
            min_loss = {}
            if epoch_ + 1 == warmup_epoch and hasattr(self, 'z_minion'):
                zweight = zinit_weight

            iterator = iter(dloader)
            for bidx in range(1, bpe + 1):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(dloader)
                    batch = next(iterator)
                #feopt.zero_grad()
                fe_h = {}
                # accumulate all forwards of FE and concat in same batch
                fe_forwards = [batch['chunk']]
                # forward chunk (alone) through frontend
                if self.mi_fwd:
                    fe_forwards.extend([batch['chunk_ctxt'],
                                       batch['chunk_rand']])
                fe_forwards.append(batch['cchunk'])
                # build triplet batch and forward it too
                fe_forwards_b = torch.cat(fe_forwards, dim=0)
                if self.vq:
                    vq_loss, fe_Q, \
                    vq_pp, vq_idx = frontend(fe_forwards_b.to(device))
                    fe_h['all'] = fe_Q
                else:
                    fe_h['all'] = frontend(fe_forwards_b.to(device))
                # slice the tensor back in batch dimension
                all_feh = torch.chunk(fe_h['all'], len(fe_forwards), dim=0)
                fe_h['chunk'] = all_feh[0]
                fe_h['cchunk'] = all_feh[-1]
                min_h = {}
                h = fe_h['chunk']
                skip_acum = None
                for mi, minion in enumerate(minions_run, start=1):
                    min_name = self.minions[mi - 1].name
                    if 'mi' in min_name:
                        triplet_P = self.join_skip(torch.cat((all_feh[0],
                                                              all_feh[1]),
                                                             dim=1), skip_acum)
                        triplet_N = self.join_skip(torch.cat((all_feh[0],
                                                              all_feh[2]),
                                                             dim=1), skip_acum)
                        triplet_all = torch.cat((triplet_P, triplet_N), dim=0)
                        if min_name == 'cmi':
                            # average through time dimension for ChunkMI
                            triplet_all = torch.mean(triplet_all, dim=2,
                                                     keepdim=True)
                        y = minion(triplet_all)
                        bsz = y.size(0)//2
                        slen = y.size(2)
                        batch[min_name] = torch.cat((torch.ones(bsz, 1, slen),
                                                     torch.zeros(bsz, 1, slen)),
                                                    dim=0)

                    else:
                        if self.minions[mi - 1].skip:
                            y, h_ = minion(self.join_skip(h, skip_acum))
                            if skip_acum is None:
                                skip_acum = h_
                            else:
                                skip_acum = torch.cat((skip_acum, h_), dim=1)
                        else:
                            y = minion(self.join_skip(h, skip_acum))
                        if min_name == 'spc':
                            # we must create the spc labels, composed of 
                            # B ones and B zeros (future and past). It
                            # internally creates 2B samples
                            bsz = y.size(0) // 2
                            slen = y.size(2)
                            batch['spc'] = torch.cat((torch.ones(bsz, 1, slen),
                                                      torch.zeros(bsz, 1,
                                                                  slen)),
                                                     dim=0)
                    min_h[min_name] = y

                if epoch_ + 1 >= warmup_epoch and hasattr(self, 'z_minion'):
                    # First shape the hidden space as Z if needed
                    # Adversarial learning to map Fe(wav) to Z ~ prior
                    if cfg['cchunk_prior']:
                        # clean chunk inference is the 'real' reference
                        z_real = fe_h['cchunk']
                        z_true_trainable = True
                    else:
                        z_real = None
                        z_true_trainable = False
                    z_losses = self.z_minion.loss(global_step,
                                                  fe_h['chunk'],
                                                  zopt,
                                                  z_true=z_real,
                                                  z_true_trainable=z_true_trainable)
                    # TODO: weight new adversarial minion?
                    #g_loss = zweight * g_loss
                   
                    # update weight incrementally if needed still
                    zweight = min(1, zweight + zinc)

                global_step += 1
                t_loss = torch.zeros(1).to(device)
                if z_losses is not None and 'g_loss' in z_losses:
                    # aggregate adversarial loss for PASE trunk
                    t_loss += z_losses['g_loss']
                if self.vq:
                    # aggregate VQ loss
                    t_loss += vq_loss
                # backprop time
                if rndmin_train:
                    if rnd_min not in min_global_steps:
                        min_global_steps[rnd_min] = 0
                    min_names = list(min_h.keys())
                    rnd_min = random.choice(min_names)
                    #minopts[rnd_min].zero_grad()
                    y_ = min_h[rnd_min]
                    minion = minions_run[self.min2idx[rnd_min]]
                    y_lab = batch[rnd_min].to(device)
                    lweight = minion.loss_weight
                    if isinstance(minion.loss, WaveAdversarialLoss):
                        loss = minion.loss(min_global_steps[rnd_min],
                                           y_, y_lab, c_real=fe_h['chunk'])
                        d_real_loss = loss['d_real_loss']
                        d_fake_loss = loss['d_fake_loss']
                        if not '{}_Dreal'.format(rnd_min) in min_loss:
                            min_loss['{}_Dreal'.format(rnd_min)] = []
                            min_loss['{}_Dfake'.format(rnd_min)] = []
                        if not '{}_Dreal'.format(rnd_min) in min_global_steps:
                            min_global_steps['{}_Dreal'.format(rnd_min)] = 0
                            min_global_steps['{}_Dfake'.format(rnd_min)] = 0
                        min_loss['{}_Dreal'.format(rnd_min)] = d_real_loss.item()
                        min_loss['{}_Dfake'.format(rnd_min)] = d_fake_loss.item()
                        loss = loss['g_loss']
                        loss = lweight * loss
                        loss.backward()
                    else:
                        loss = minion.loss(y_, y_lab)
                        loss = lweight * loss
                        loss.backward()
                    if rnd_min not in min_loss:
                        min_loss[rnd_min] = []
                    min_loss[rnd_min].append(loss.item())
                    min_global_steps[rnd_min] += 1
                    if '{}_Dreal'.format(rnd_min) in min_global_steps:
                        min_global_steps['{}_Dreal'.format(rnd_min)] += 1
                        min_global_steps['{}_Dfake'.format(rnd_min)] += 1
                    #minopts[rnd_min].step()
                else:
                    #if hasattr(self, 'minions_dp'):
                    #    raise NotImplementedError('DataParallel to be included')
                    # Compute all minion losses
                    for min_name, y_ in min_h.items():
                        if min_name not in min_global_steps:
                            min_global_steps[min_name] = 0
                        minion = minions_run[self.min2idx[min_name]]
                        minopts[min_name].zero_grad()
                        y_lab = batch[min_name].to(device)
                        lweight = minion.loss_weight
                        if isinstance(minion.loss, WaveAdversarialLoss):
                            loss = minion.loss(min_global_steps[min_name],
                                               y_, y_lab, c_real=fe_h['chunk'])
                            d_real_loss = loss['d_real_loss']
                            d_fake_loss = loss['d_fake_loss']
                            if not '{}_Dreal'.format(min_name) in min_loss:
                                min_loss['{}_Dreal'.format(min_name)] = []
                                min_loss['{}_Dfake'.format(min_name)] = []
                            if not '{}_Dreal'.format(min_name) in min_global_steps:
                                min_global_steps['{}_Dreal'.format(min_name)] = 0
                                min_global_steps['{}_Dfake'.format(min_name)] = 0
                            min_loss['{}_Dreal'.format(min_name)].append(d_real_loss.item())
                            min_loss['{}_Dfake'.format(min_name)].append(d_fake_loss.item())
                            loss = loss['g_loss']
                            loss = lweight * loss
                        else:
                            loss = minion.loss(y_, y_lab)
                            loss = lweight * loss
                        t_loss += loss
                        #loss.backward(retain_graph=True)
                        if min_name not in min_loss:
                            min_loss[min_name] = []
                        min_loss[min_name].append(loss.item())
                        min_global_steps[min_name] += 1
                        if '{}_Dreal'.format(min_name) in min_global_steps:
                            min_global_steps['{}_Dreal'.format(min_name)] += 1
                            min_global_steps['{}_Dfake'.format(min_name)] += 1
                        #minopts[min_name].step()
                    t_loss.backward()
                if bidx % batch_acum == 0 or bidx >= bpe:
                    # first store the gradients
                    grads = get_grad_norms(self, grad_keys)
                    # update minions
                    for min_name, y_ in min_h.items():
                        minopts[min_name].step()
                        minopts[min_name].zero_grad()
                    # update frontend
                    feopt.step()
                    feopt.zero_grad()
                    if epoch_ + 1 >= warmup_epoch and hasattr(self, 'z_minion'):
                        zopt.step()
                        zopt.zero_grad()
                end_t = timeit.default_timer()
                timings.append(end_t - beg_t)
                beg_t = timeit.default_timer()
                if bidx % log_freq == 0 or bidx >= bpe:
                    print('-' * 50)
                    print('Batch {}/{} (Epoch {}):'.format(bidx, bpe, epoch_))
                    for min_name, losses in min_loss.items():
                        print('Minion {} loss: {:.3f} gidx: '
                              '{:5d} '.format(min_name, losses[-1], 
                                              min_global_steps[min_name]))
                        writer.add_scalar('train/{}_loss'.format(min_name),
                                          losses[-1], min_global_steps[min_name])
                        if min_name in min_h:
                            # Adversarial are not included
                            writer.add_histogram('train/{}'.format(min_name),
                                                 min_h[min_name].data,
                                                 bins='sturges',
                                                 global_step=min_global_steps[min_name])
                        if min_name in min_h:
                            # Adversarial are not included
                            writer.add_histogram('train/gtruth_{}'.format(min_name),
                                                 batch[min_name].data,
                                                 bins='sturges',
                                                 global_step=min_global_steps[min_name])
                    if z_losses is not None:
                        z_log = 'ZMinion '
                        if 'dfake_loss' in z_losses:
                            dfake_loss = z_losses['dfake_loss'].item()
                            z_log += 'dfake_loss: {:.3f},'.format(dfake_loss)
                            writer.add_scalar('train/dfake_loss',
                                              dfake_loss,
                                              global_step)
                        if 'dreal_loss' in z_losses:
                            dreal_loss = z_losses['dreal_loss'].item()
                            writer.add_scalar('train/dreal_loss',
                                              dreal_loss,
                                              global_step)
                            z_log += ' dreal_loss: {:.3f},'.format(dreal_loss)
                        if 'greal_loss' in z_losses:
                            greal_loss = z_losses['greal_loss'].item()
                            z_log += ', greal_loss: {:.3f},'.format(greal_loss)
                            writer.add_scalar('train/greal_loss',
                                              greal_loss,
                                              global_step)
                        if 'gfake_loss' in z_losses:
                            gfake_loss = z_losses['gfake_loss'].item()
                            z_log += ', gfake_loss: {:.3f},'.format(gfake_loss)
                            writer.add_scalar('train/gfake_loss',
                                              gfake_loss,
                                              global_step)
                        print(z_log)
                        #writer.add_scalar('train/zweight',
                        #                  zweight,
                        #                  global_step)
                        if z_true_trainable:
                            writer.add_histogram('train/z_real',
                                                 fe_h['cchunk'],
                                                 bins='sturges',
                                                 global_step=global_step)
                        writer.add_histogram('train/z_fake',
                                             fe_h['chunk'],
                                             bins='sturges',
                                             global_step=global_step)
                    if self.vq:
                        print('VQLoss: {:.3f}, VQPP: '
                              '{:.3f}'.format(vq_loss.item(), vq_pp.item()))
                        writer.add_scalar('train/vq_loss', vq_loss.item(),
                                          global_step=global_step)
                        writer.add_scalar('train/vq_pp', vq_pp.item(),
                                          global_step=global_step)
                    # --- Get gradient norms for sanity check ----
                    for kgrad, vgrad in grads.items():
                        writer.add_scalar('train/GRAD/{}'.format(kgrad),
                                          vgrad, global_step)
                    print('Total summed loss: {:.3f}'.format(t_loss.item()))


                    print('Mean batch time: {:.3f} s'.format(np.mean(timings)))
            # epoch end
            if va_dloader is not None:
                va_bpe = cfg['va_bpe']
                eloss = self.eval_(va_dloader, bsize, va_bpe, log_freq=log_freq,
                                   epoch_idx=epoch_,
                                   writer=writer, device=device)

                """
                if lrdecay > 0:
                    # update frontend lr
                    fesched.step(eloss)
                    # update Z minion lr
                    if hasattr(self, 'z_minion'):
                        zsched.step(eloss)
                    # update each minion lr
                    for mi, minion in enumerate(self.minions, start=1):
                        minscheds[minion.name].step(eloss)
                """

            if lrdecay > 0:
                # update frontend lr
                fesched.step()
                # update Z minion lr
                if hasattr(self, 'z_minion'):
                    zsched.step()
                # update each minion lr
                for mi, minion in enumerate(self.minions, start=1):
                    minscheds[minion.name].step()

            # Save plain frontend weights 
            fe_path = os.path.join(save_path, 
                                   'FE_e{}.ckpt'.format(epoch_))
            torch.save(self.frontend.state_dict(), fe_path)
            # Run through each saver to save model and optimizer
            for saver in savers:
                saver.save(saver.prefix[:-1], global_step)
            #torch.save(self.state_dict(),
            #           os.path.join(save_path,
            #                        'fullmodel_e{}.ckpt'.format(epoch_)))
            # TODO: sup. aux losses
            if (epoch_ + 1 ) % sup_freq == 0 or \
               (epoch_ + 1) >= (epoch_beg + epoch):
                if hasattr(self, 'aux_sup'):
                    self.aux_sup(epoch_, fe_path, cfg['fe_cfg'])

    def eval_(self, dloader, batch_size, bpe, log_freq,
              epoch_idx=0, writer=None, device='cpu'):
        self.eval()
        with torch.no_grad():
            bsize = batch_size
            frontend = self.frontend
            minions_run = self.minions
            print('=' * 50)
            print('Beginning evaluation...')
            timings = []
            beg_t = timeit.default_timer()
            min_loss = {}

            iterator = iter(dloader)
            for bidx in range(1, bpe + 1):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(dloader)
                    batch = next(iterator)
                # Build chunk keys to know what to encode
                chunk_keys = ['chunk']
                if self.mi_fwd:
                    chunk_keys += ['chunk_ctxt', 'chunk_rand']
                fe_h = {}
                # Forward chunk(s) through frontend
                for k in chunk_keys:
                    fe_h[k] = frontend(batch[k].to(device))
                min_h = {}
                h = fe_h['chunk']
                skip_acum = None
                for mi, minion in enumerate(minions_run, start=1):
                    min_name = self.minions[mi - 1].name
                    if 'mi' in min_name:
                        triplet_P = self.join_skip(torch.cat((fe_h['chunk'],
                                                              fe_h['chunk_ctxt']),
                                                             dim=1), skip_acum)
                        triplet_N = self.join_skip(torch.cat((fe_h['chunk'],
                                                              fe_h['chunk_rand']),
                                                             dim=1), skip_acum)
                        triplet_all = torch.cat((triplet_P, triplet_N), dim=0)
                        if min_name == 'cmi':
                            # average through time dimension for ChunkMI
                            triplet_all = torch.mean(triplet_all, dim=2,
                                                     keepdim=True)
                        y = minion(triplet_all)
                        bsz = y.size(0)//2
                        slen = y.size(2)
                        batch[min_name] = torch.cat((torch.ones(bsz, 1, slen),
                                                     torch.zeros(bsz, 1, slen)),
                                                    dim=0)
                    else:
                        if self.minions[mi - 1].skip:
                            y, h_ = minion(self.join_skip(h, skip_acum))
                            if skip_acum is None:
                                skip_acum = h_
                            else:
                                skip_acum = torch.cat((skip_acum, h_), dim=1)
                        else:
                            y = minion(self.join_skip(h, skip_acum))
                        if min_name == 'spc':
                            # we must create the spc labels, composed of 
                            # B ones and B zeros (future and past). It
                            # internally creates 2B samples
                            bsz = y.size(0) // 2
                            slen = y.size(2)
                            batch['spc'] = torch.cat((torch.ones(bsz, 1, slen),
                                                      torch.zeros(bsz, 1,
                                                                  slen)),
                                                     dim=0)
                    min_h[min_name] = y

                # Compute all minion losses
                for min_name, y_ in min_h.items():
                    y_lab = batch[min_name].to(device)
                    minion = self.minions[self.min2idx[min_name]]
                    lweight = minion.loss_weight
                    if isinstance(minion.loss, WaveAdversarialLoss):
                        loss = minion.loss(bidx,
                                           y_, y_lab, c_real=fe_h['chunk'],
                                           grad=False)
                        loss = loss['g_loss']
                    else:
                        loss = minion.loss(y_, y_lab)
                    loss = lweight * loss
                    #loss = lweight * self.minions[self.min2idx[min_name]].loss(y_, y_lab)
                    if min_name not in min_loss:
                        min_loss[min_name] = []
                    min_loss[min_name].append(loss.item())
                end_t = timeit.default_timer()
                timings.append(end_t - beg_t)
                beg_t = timeit.default_timer()
                
                if bidx % log_freq == 0 or bidx >= bpe:
                    print('-' * 50)
                    print('EVAL Batch {}/{} (Epoch {}):'.format(bidx, 
                                                                bpe,
                                                                epoch_idx))
                    for min_name, losses in min_loss.items():
                        print('Minion {} loss: {:.3f}'
                              ''.format(min_name, losses[-1]))
                    print('Mean batch time: {:.3f} s'.format(np.mean(timings)))

            # --------------------------------------------------------------
            # After all eval data, write mean values of epoch per minion
            aggregate = 0
            for min_name, losses in min_loss.items():
                mlosses = np.mean(losses)
                writer.add_scalar('eval/{}_loss'.format(min_name),
                                  mlosses, epoch_idx)
                aggregate += mlosses
            # aggregate eval loss
            writer.add_scalar('eval/total_loss', aggregate,
                              epoch_idx)
            return aggregate


    def state_dict(self):
        sdict = {}
        for k, v in super().state_dict().items():
            if '_dp.' in k:
                # skip any DataParallel wrapped thing
                continue
            sdict[k] = v
        return sdict


if __name__ == '__main__':
    wmodel = Waveminionet(
        minions_cfg=[
            {'num_outputs':1,
             'dropout':0.2,
             'name':'chunk',
             'type':'decoder',
             },
            {'num_outputs':257,
             'dropout':0.2,
             'name':'lps',
             },
            {'num_outputs':40,
             'dropout':0.2,
             'name':'mfcc'
             },
            {'num_outputs':4,
             'dropout':0.2,
             'name':'prosody'
             },
            #{'num_outputs':1,
            # 'dropout':0.2,
            # 'name':'mi',
            # 'keys':['chunk',
            #         'chunk_ctxt',
            #         'chunk_rand']
            #},
        ]
                         )
    print(wmodel)
    x = torch.randn(1, 1, 8000)
    outs, y = wmodel(x)
    for k, v in outs.items():
        print('{} : {}'.format(k, v.size()))
