import argparse
import logging
import os

import h5py
import numpy as np
import torch
from torch.utils import data

column_indices = [0, 1, 2, 5, 6, 7, 8, -1]

class Dataset(data.Dataset):
    """
        Load dataset
    """

    def __init__(self, ID, opt, lensed, rseed=934):
        self.logger = logging.getLogger(__name__)

        self.ID = ID
        self.rseed = rseed
        self.lensed = lensed
        self.dpath = opt.dpath
        self.dhead = opt.dhead
        self.train = opt.train
        self.problem = opt.problem
        self.spinning = opt.spinning
        self.lense_model = opt.lense_model
        self.f_extension = opt.f_extension
        self.norm_method = opt.norm_method
        self.triplet_class = opt.triplet_class

        self.pp, self.y = self._load_data()
    
        self.len = self.y.size()[0]

    def _get_label(self, leng):
        _value = 1 if self.lensed else 0
        if self.triplet_class and self.spinning:
            _value = 2
        _label = torch.zeros(leng, dtype=torch.long)+_value

        # unlensed: 0, lesned: 1, unlensed_precessing: 2

        return _label

    def _target_norm(self, fpath, data):
        self.logger.info("Target normalization")
        fname = 'min_max.npy'
        fmm_dir = os.path.join(fpath, fname)
        self.logger.info("Load min_max from %s"%fmm_dir)
        min_max = np.load(fmm_dir)[:, column_indices]
        self.mins, self.maxs = min_max[0], min_max[1]
        data = (data-self.mins)/(self.maxs-self.mins)
        data = data*2.-1.

        return data
    
    def target_unnorm(self, data):
        self.logger.info("Target unnormalization")
        data = (data+1.)/2.
        data = (self.maxs-self.mins)*data+self.mins
        return data

    def input_norm(self, args):
        if self.norm_method == 'feature_wise':
            self.logger.info("Perform feature wise normalization")
            means, stds = args
            for i in range(self.pp.shape[1]):
                mean = means[i]
                std = stds[i]
                self.pp[:, i] = (self.pp[:, i] - mean)/std
        elif self.norm_method == 'sample_wise':
            self.logger.info("Perform sample wise normalization")
            for i in range(self.pp.shape[0]):
                ppmin = self.pp[i].min()
                ppmax = self.pp[i].max()
                self.pp[i] = (self.pp[i]-ppmin)/(ppmax-ppmin)
        else:
            self.logger.info("Perform global wise normalization")
            ppmin, ppmax = args
            self.pp = (self.pp-ppmin)/(ppmax-ppmin)

    def _load_data(self):
        if self.lensed:
            lense_token = 'lensed'
            dname = self.dhead+'_'+self.lense_model
        else:
            lense_token = 'unlensed'
            dname = self.dhead+'_None'
            if self.spinning:
                dname += '_spinning'
        # dname += '_'+self.norm_method
        fpath = os.path.join(self.dpath, dname)

        fpp_name = '%s_pp_%s.%s' % (self.ID, lense_token, self.f_extension)
        fpp_dir = os.path.join(fpath, fpp_name)

        self.logger.info("Load spectrograms from %s" % fpp_dir)
        with h5py.File(fpp_dir, 'r') as f:
            pp = np.array(f['pp'], dtype=np.float32)

        if self.problem == 'regression':
            fpa_name = fpp_name.replace('_pp_', '_params_')
            fpa_dir = os.path.join(fpath, fpa_name)

            self.logger.info("Load parameters from %s" % fpa_dir)
            with h5py.File(fpa_dir, 'r') as f:
                params = np.array(f['parameters'], dtype=np.float32)
            
            params = params.T[column_indices].T

            params = self._target_norm(fpath, params)
            for i, p in enumerate(params):
                # print(min(p), max(p))
                if min(p) < -1 or max(p) > 1:
                    raise ValueError("Target normalization went wrong!")

            return torch.from_numpy(pp), torch.from_numpy(params)
        else:
            labels = self._get_label(len(pp))

            return torch.from_numpy(pp), labels

    def __getitem__(self, index):
        pp, y = self.pp[index], self.y[index]
        return pp, y

    def __len__(self):
        return self.len

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def Parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='VGG11', type=str, dest='model')
    parser.add_argument('--cdate', default='latest', type=str, dest='cdate')
    parser.add_argument('--dpath', default='./data/', type=str, dest='dpath')
    parser.add_argument('--dhead', default='imrppv2', type=str, dest='dhead')
    parser.add_argument('--problem', default='regression', type=str, dest='problem')
    parser.add_argument('--f_extension', default='h5', type=str, dest='f_extension')
    parser.add_argument('--test_ppath', default='Plots', type=str, dest='test_ppath')
    parser.add_argument('--test_spath', default='Results', type=str, dest='test_spath')
    parser.add_argument('--reg_loss', default='SmoothL1Loss', type=str, dest='reg_loss')
    parser.add_argument('--test_cpath', default='Checkpoint', type=str, dest='test_cpath')
    parser.add_argument('--cls_loss', default='NLLLoss', type=str, dest='cls_loss')
    parser.add_argument('--norm_method', default='feature_wise', type=str, dest='norm_method')
    parser.add_argument('--model_state', default='model_state.pt', type=str, dest='model_state')
    parser.add_argument('--test_epath', default='./Experiments_nm/', type=str, dest='test_epath')
    parser.add_argument('--trainer_state', default='trainer_state.pt', type=str, dest='trainer_state')
    parser.add_argument('--lense_model', choices=['None', 'PML', 'SIS', 'MICRO'], default='None', type=str, dest='lense_model')
    parser.add_argument('--tl_model', choices=['PML', 'SIS'], default='PML', type=str, dest='tl_model')

    parser.add_argument('--gkde', default=False, type=str2bool, dest='gkde')
    parser.add_argument('--train', default=False, type=str2bool, dest='train')
    parser.add_argument('--lensed', default=True, type=str2bool, dest='lensed')
    parser.add_argument('--resume', default=False, type=str2bool, dest='resume')
    parser.add_argument('--spinning', default=False, type=str2bool, dest='spinning')
    parser.add_argument('--danalysis', default=False, type=str2bool, dest='danalysis')
    parser.add_argument('--results_plot', default=False, type=str2bool, dest='results_plot')
    parser.add_argument('--triplet_class', default=False, type=str2bool, dest='triplet_class')

    parser.add_argument('--bsize', default=64, type=int, dest='bsize')
    parser.add_argument('--gpuid', default=0, type=int, dest='gpuid')
    parser.add_argument('--print_every', default=10, type=int, dest='print_every')
    parser.add_argument('--lr_decay_epoch', default=1, type=int, dest='lr_decay_epoch')
    parser.add_argument('--validate_every', default=100, type=int, dest='validate_every')
    parser.add_argument('--checkpoint_every', default=10000, type=int, dest='checkpoint_every')
    parser.add_argument('--max_training_epoch', default=100, type=int, dest='max_training_epoch')

    parser.add_argument('--train_drop', default=0.1, type=float, dest='train_drop')

    opt = parser.parse_args()

    opt.classification = True if opt.problem == 'classification' else False

    opt.test_epath = opt.test_epath.replace('nm', opt.norm_method)
    opt.token = '_'+opt.lense_model
    if opt.classification:
        # opt.bsize /= 2
        opt.lensed = True
        opt.token += '_None'
        if opt.triplet_class:
            opt.token += '-triplet'
            opt.spinning = False
        if opt.spinning:
            opt.token += '-spinning'
    opt.token += '_'+opt.problem+'_'+opt.model

    return opt