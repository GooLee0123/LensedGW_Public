# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 18:36:48 2018

@author: Ivan
Modified by Robin
Modified by Joongoo Lee

Generates a pickle file of PML or SIS spectrograms and parameters

"""
# from __future__ import print_function

import logging
import os
import pickle
import random
import time
import traceback
from multiprocessing import Pool, cpu_count
import sys

import h5py
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pycbc.noise
import pycbc.psd
# from utils.progress_bar import ProgressBar
import scipy.fftpack
import scipy.integrate
import tables
from matplotlib.pyplot import mlab
from pycbc.waveform import get_td_waveform
from recordtype import recordtype
from astropy.cosmology import WMAP9 as cosmo

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(
    format=LOG_FORMAT,
    level=getattr(logging, LOG_LEVEL)
)

#from collections import namedtuple
data_extension = 'hdf5'
debug = False
lensing = True if sys.argv[1].startswith('T') else False
spinning = True if sys.argv[2].startswith('T') else False
if lensing:
    spinning = False
lensing_model = sys.argv[3] if lensing else 'None'
# norm_method = sys.argv[4] # [feature_wise, sample_wise]
start_id = 0
samples = 45000        # Change number of samples and directory for saving files
sig_dur = 8            # Duration of signal in second
#output_dir = 'outputs/test/' if debug else 'outputs/imrppv2_spin/'
output_dir = './data/test/' if debug else './data/imrppv2_%s' % (lensing_model)
if spinning:
    output_dir += '_spinning'
# output_dir += '_'+norm_method
ftoken = 'lensed' if lensing else 'unlensed'
n_thread = 1 if debug else cpu_count()
# n_thread = 1

fs = 4096
f_lower = 20
f_upper = 512
#noise_signal_f_ratio = 100

# Physical constants in SI unit
G_SI    = 6.674E-11 # Gravitational constant [m^3/kg/s^2].
c_SI    = 2.998E8   # Speed of light [m/s].
p_SI    = 3.086E16  # parsec in SI unit [m].
SM_SI   = 1.989E30  # Solar mass [kg].
H0_SI   = 2.268E-18 # Hubble constant [m/s/m].
                    # This is equivalent to 70 [km/s/Mpc] 
                    # as in (Takahashi & Nakamura 2003, https://arxiv.org/abs/astro-ph/0305055)

#Physical constants fo
H    = 70  # [km/s/Mpc]
G    = 1
c    = 1
p    = 2.291458E-11
SM   = 1.476E3

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_dir(output_dir)

def loguniform(low=0, high=1, size=None, base=10):
    return np.power(base, np.random.uniform(low, high, size))

Params = recordtype('Param', 'M_L D_L D_LS D_S ETA z_L z_s MC M1 M2 spin1 spin2 y mu_plus mu_minus SNR t_delay chi_eff', default=None)
#Params.__new__.func_defaults = (None,) * len(Params._fields) #default as None

def sample_param(lensing=True, spinning=True):
    sample_configs = {'M_L': (3, 5, 'log'), # Exponent of lense mass in solar mass
                  'D': (10E6, 1000E6, 'lin'), # Distance in parsec
                  'ETA': (1e-6, 0.5, 'lin'), # Source position in parsec
                  'CM': (5, 55, 'lin'), # Component masses in solar mass
    }

    M_L = SM_SI * random_sample(sample_configs['M_L']) if lensing else 0 # Lense mass in SI
    ETA = p_SI * random_sample(sample_configs['ETA']) if lensing else 0 # Displacement of source in SI
    D_L = p_SI * random_sample(sample_configs['D']) # Lense distance in SI
    D_LS = p_SI * random_sample(sample_configs['D']) # Distance between source and lense in SI
    M1 = random_sample(sample_configs['CM']) # Prior BH mass
    sample_configs['CM2'] = (4, M1, 'lin') # The mass range of secondary BH
    M2 = random_sample(sample_configs['CM2']) # Secondary BH mass
    MC = (M1*M2)**(0.6)/(M1+M2)**(0.2)
    D_S = D_L + D_LS # Source distance in SI
    z_L = H0_SI*D_L/c_SI # lense redshift
    z_s = H0_SI*D_S/c_SI # source redshift
    M1 *= SM_SI # to SI
    M2 *= SM_SI # to SI

    spin1 = random_ball() if spinning else [0,0,0]
    spin2 = random_ball() if spinning else [0,0,0]

    chi_eff = pycbc.conversions.chi_eff(M1, M2, spin1[-1], spin2[-1])

    params = Params(M_L, D_L, D_LS, D_S, ETA, z_L, z_s, MC, M1, M2, spin1, spin2)
    params.chi_eff = chi_eff
    return params

def random_sample(config):
    if config[-1] == 'log':
        return loguniform(config[0], config[1])
    elif config[-1] == 'lin':
        return np.random.uniform(config[0], config[1])
    else:
        print('Invalid Config: {}'.format(config[-1]))
        raise Exception

def random_ball(dimension=3, radius=1):
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_direction = np.random.normal(size=(dimension,))
    random_direction /= np.linalg.norm(random_direction)
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = np.random.random() ** (1./dimension)
    # Return the list of random (direction & length) points.
    return radius * (random_direction * random_radii)

def get_waveform(params, fs, approx='IMRPhenomPv2'):
    hp, _ = get_td_waveform(approximant=approx,
                            mass1=params.M1/SM_SI, # to solar mass
                            mass2=params.M2/SM_SI, # to solar mass
                            distance=params.D_S/(p_SI*1E6), # to Mpc
                            spin1x = params.spin1[0],
                            spin1y = params.spin1[1],
                            spin1z = params.spin1[2],
                            spin2x = params.spin2[0],
                            spin2y = params.spin2[1],
                            spin2z = params.spin2[2],
                            delta_t=1.0/fs,
                            f_lower=f_lower
                            )
    return hp

def PML(hp, params):
    #amplification function in f domain
    M_lz = params.M_L*(1+params.z_L) # Redshifted lens mass [kg]
    Xi1 = 4*G_SI*params.M_L/c_SI**2 # term1 of Einstein radius [m]
    Xi2 = params.D_L*params.D_LS/params.D_S # term2 of Einstein radius  [m]
    Xi_SI = np.sqrt(Xi1*Xi2) # Einstein radius in SI [m]
    params.y = (params.ETA*params.D_L)/(Xi_SI*params.D_S) # Dimensionless source position
    if not (1 > params.y > 0.05):
        return None

    # Calculating magnification factors and time delay
    beta = np.sqrt(params.y*params.y + 4) # For simplicity of equations
    k = (params.y*params.y + 2)/(2*params.y*beta) # For simplicity of equations
    params.mu_plus  = 0.5 + k
    params.mu_minus = 0.5 - k
    params.t_delay = 4*G_SI/(c_SI**3)*M_lz*((0.5*params.y*beta) + np.log((beta + params.y)/(beta - params.y))) # Time delay

    hf = hp.to_frequencyseries()
    f_vals = hf.sample_frequencies.numpy()
    Ff1 = np.sqrt(np.abs(params.mu_plus)) # Term1 for F(f)
    Ff2 = 1j*np.sqrt(np.abs(params.mu_minus)) # Term2 for F(f)
    Ff3 = np.exp(2*np.pi*1j*f_vals*params.t_delay) # Term3 for F(f)
    Ff = Ff1-Ff2*Ff3 # F(f)
    hlf = Ff*hf
    hl = hlf.to_timeseries()
    hp = hf.to_timeseries()

    return hl

def SIS(hp, params):
    M_lz = params.M_L*(1+params.z_L) # Redshifted lens mass [kg]
    Xi1 = 4*G_SI*params.M_L/c_SI**2 # term1 of Einstein radius [m]
    Xi2 = params.D_L*params.D_LS/params.D_S # term2 of Einstein radius  [m]
    Xi_SI = np.sqrt(Xi1*Xi2) # Einstein radius in SI [m]
    params.y = (params.ETA*params.D_L)/(Xi_SI*params.D_S) # Dimensionless source position
    if not (1 > params.y > 0.05):
        return None

    # Calculating magnification factors and time delay
    params.mu_plus  = 1+1/params.y
    params.mu_minus = -1+1/params.y
    params.t_delay = 8*M_lz*params.y*G_SI/c_SI**3

    hf = hp.to_frequencyseries()
    f_vals = hf.sample_frequencies.numpy()
    Ff1 = np.sqrt(np.abs(params.mu_plus)) # Term1 for F(f)
    Ff2 = 1j*np.sqrt(np.abs(params.mu_minus)) # Term2 for F(f)
    Ff3 = np.exp(2*np.pi*1j*f_vals*params.t_delay) # Term3 for F(f)
    Ff = Ff1-Ff2*Ff3 # F(f)
    hlf = Ff*hf
    hl = hlf.to_timeseries()
    
    return hl

def lensing_effect(h, params, model='PML'):
    return globals()[model](h, params)

# The color of the noise matches a PSD which you provide
# aLIGOZeroDetHighPower
# aLIGOaLIGODesignSensitivityT1800044
delta_f = 1/256.
flen = int(fs//2/delta_f)+1
psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, f_lower)
def get_noise(nlen, del_t, stime):
    ts = pycbc.noise.noise_from_psd(nlen, del_t, psd)
    ts.start_time = stime
    return ts

def cal_SNR(hl, n):
    hl_f = abs(hl.to_frequencyseries())
    hl_freq = hl_f.sample_frequencies

    n_f = n.to_frequencyseries()
    
    # Manual cutoff frequency
    N = int(hl.duration/hl.delta_t)
    kmin, kmax = pycbc.filter.get_cutoff_indices(f_lower, f_upper, hl.delta_f, (N - 1) * 2)
    hl_f = hl_f[kmin+1:kmax]
    hl_freq = hl_freq[kmin+1:kmax]
    n_f = n_f[kmin+1:kmax]

    # Calculate SNR
    hl_f = hl_f.numpy()
    n_f = n_f.numpy() 
    y_integrand = hl_f*hl_f/(n_f*n_f) # Note n(f) has to be squared to give the PSD as it is calculated manually
    delta_f = hl_freq[1] - hl_freq[0]
    SNR = np.abs(np.sqrt(4*np.sum(y_integrand)*delta_f))

    return SNR

def qtransform(_h):
    if type(_h) != pycbc.types.timeseries.TimeSeries:
        raise TypeError("Type of strain for Q-transform should be pycbc timeseries")
    tt, ff, pp = _h.qtransform(1/32.,
                            logfsteps=256,
                            qrange=(8, 8),
                            frange=(f_lower, f_upper))
    return tt, ff, pp

def get_picklable(params):
    if lensing:
        picklable_params = [params.MC,
                            params.M_L/SM_SI,
                            params.y,
                            params.D_L/p_SI,
                            params.D_S/p_SI, 
                            params.z_L, 
                            params.z_s,
                            params.mu_plus,
                            params.mu_minus,
                            params.ETA,
                            params.D_LS,
                            params.M1/SM_SI,
                            params.M2/SM_SI,
                            params.spin1[0],
                            params.spin1[1],
                            params.spin1[2],
                            params.spin2[0],
                            params.spin2[1],
                            params.spin2[2],
                            params.SNR]
    else:
        picklable_params = [params.MC,
                            params.D_S/p_SI,
                            params.z_s/p_SI,
                            params.M1/SM_SI, 
                            params.M2/SM_SI,
                            params.spin1[0],
                            params.spin1[1],
                            params.spin1[2],
                            params.spin2[0],
                            params.spin2[1],
                            params.spin2[2],
                            params.SNR]
    return picklable_params

def plotstft(tt, ff, pp):
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.pcolormesh(tt, ff, pp)
    ax.axis('tight')
    ax.axis('off')
    return fig

# def feature_wise_normalization(pps, means=None, stds=None):
#     iter_cond = True if not means else False
#     for i in range(pps.shape[1]):
#         if iter_cond:
#             if i == 0:
#                 means, stds = [], []
#             mean = np.mean(pps[:, i], 0)
#             std = np.std(pps[:, i], 0)
#             means.append(mean)
#             stds.append(std)
#         else:
#             mean = means[i]
#             std = stds[i]
#         pps[:, i] = (pps[:, i] - mean)/std
#     return pps, means, stds

# def sample_wise_normalization(pps):
#     ppmin = np.min(pps)
#     ppmax = np.max(pps)
#     pps = (pps-ppmin)/(ppmax-ppmin)
#     # for i in range(pps.shape[0]):
#         # ppmin = pps[i].min()
#         # ppmax = pps[i].max()
#         # pps[i] = (pps[i]-ppmin)/(ppmax-ppmin)
#     return pps

rtrain, rval, rtest = .8, .1, .1
def save_data(pps, parameters):
    pps = np.array(pps)

    ndata = len(pps)
    ntrain = int(ndata*rtrain)
    nval = int(ndata*rval)
    
    dlens = [ntrain, nval, ndata-(ntrain+nval)]

    # if norm_method == 'feature_wise':
    #     logging.info("Perform feature wise normalization")
    #     train_norm_pps, means, stds = feature_wise_normalization(pps[:ntrain])
    #     val_norm_pps, _, _ = feature_wise_normalization(pps[ntrain:ntrain+nval], means=means, stds=stds)
    #     test_norm_pps, _, _ = feature_wise_normalization(pps[ntrain+nval:], means=means, stds=stds)
    # else:
    #     logging.info("Perform sample wise normalization")
    #     train_norm_pps = sample_wise_normalization(pps[:ntrain])
    #     val_norm_pps = sample_wise_normalization(pps[ntrain:ntrain+nval])
    #     test_norm_pps = sample_wise_normalization(pps[ntrain+nval:])

    train_norm_pps = pps[:ntrain]
    val_norm_pps = pps[ntrain:ntrain+nval]
    test_norm_pps = pps[ntrain+nval:]

    norm_pps = np.vstack((train_norm_pps, val_norm_pps, test_norm_pps))

    par_dim = np.shape(parameters[0])
    pps_dim = np.shape(norm_pps[0])
    IDs = ['training', 'validation', 'test']

    print(np.shape(parameters))
    print(np.shape(norm_pps))

    iidx = 0
    print_every = 100
    for i in range(3):
        fpname = os.path.join(output_dir, "%s_params_%s.h5" % (IDs[i], ftoken))
        fppname = fpname.replace("params", "pp")

        fp = tables.open_file(fpname, mode='w')
        fpp = tables.open_file(fppname, mode='w')

        atom1 = tables.Float32Atom()
        atom2 = tables.Float32Atom()

        array_c1 = fp.create_earray(fp.root, 'parameters', atom1, (0, par_dim[0]))
        array_c2 = fpp.create_earray(fpp.root, 'pp', atom2, (0, pps_dim[0], pps_dim[1]))

        for j in range(iidx, iidx+dlens[i]):
            if j % print_every == 0:
                logging.info("%sth pp and parameter are appended to the files" % j)
            px = np.expand_dims(parameters[j], 0)
            ppx = np.expand_dims(norm_pps[j], 0)
            array_c1.append(px)
            array_c2.append(ppx)
            iidx += 1

        fp.close()
        fpp.close()

def workflow(i):
    i = int(i)
    np.random.seed() # to guarantee different results of workers

    #Loop until we have a set of suitable parameters.
    while 1:
        #Binary and lensing parameters.
        params = sample_param(lensing, spinning)
        hp = get_waveform(params, fs)
        if lensing:
            hl = lensing_effect(hp, params, lensing_model)
        else:
            hl = hp
        if hl is None:
            #Get another set of parameter.
            continue

        hl_len = len(hl)
        dur_hl = hl_len/float(fs)
        if dur_hl >= sig_dur: # duration of generated signal is longer than user-specified duration
            # Cut the signal down to target signal duration ('sig_dur') if the signal is longer than sig_dur.
            nhl = get_noise(len(hl), hl.delta_t, hl.start_time)
            params.SNR = cal_SNR(hl, nhl)
            hl = hl[hl_len-sig_dur*fs:] # Negative start index is not supported for pycbc.TimeSeries.
            nhl = nhl[hl_len-sig_dur*fs:]
        else:
            # Pad the signal so it has sig_dur length if the signal is shorter than sig_dur.
            pad_len = sig_dur*fs - hl_len
            hl.prepend_zeros(pad_len)
            nhl = get_noise(len(hl), hl.delta_t, hl.start_time)
            params.SNR = cal_SNR(hl[pad_len:], nhl[pad_len:])

        if params.SNR >= 10 and params.SNR <= 50:
            # logging.info("%sth waveform's SNR is: %.3f" % (i, params.SNR))
            thl, fhl, phl = qtransform(hl+nhl)
            phl = phl**0.5
            # norm_phl = 2.*(phl-phl.min())/(phl.max()-phl.min())-1.
            pp = get_picklable(params)
            # fig = plotstft(thl, fhl, phl)
            # fig.canvas.draw()
            # plt.savefig(os.path.join(output_dir, './Lensed_'+str(i).zfill(5)), pad_inches=0)
            # plt.close('all')
            return [pp, phl]

def main():
    p = Pool(n_thread)
    # chucksize = number of tasks to be completed by each thread at each call.
    SampleData = [x for x in range(start_id, samples)]

    if debug:
        results = workflow(1)
        # real_results = results.get()
        print(results)
    else:
        results = p.map_async(workflow, SampleData, chunksize=4)
        count = 0
        while not results.ready():
            done = (samples - start_id) - results._number_left*results._chunksize
            time.sleep(1)
            while count<done:
                # update(None)
                count += 1
                logging.info("%s/%s done" % (count, samples-start_id))

        p.close()
        p.join()

        real_results = results.get()
        parameters = []
        pps = []
        for param, pp in real_results:
            parameters.append(param)
            pps.append(pp)

        assert len(pps) == len(parameters)

        save_data(pps, parameters)

if __name__ == "__main__":
    main()
