import os, sys
import h5py
import numpy as np
from utils import str2bool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rc('text', usetex=False)
matplotlib.rc('font', serif='Times New Roman')
matplotlib.rc('font', size=15)
matplotlib.rc('axes', labelsize=18)
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
matplotlib.rc('legend', fontsize=12)

sig_dur = 8
f_lower = 20
f_upper = 512
target_snrs = [10, 30, 50]
tolerance = 10

lmodel = sys.argv[1]
spinning = str2bool(sys.argv[2])
norm_method = sys.argv[3]

psave = 'spec_plots_%s_%s'%(lmodel, norm_method)

if spinning:
    psave += '_spinning'

if not os.path.exists(psave):
    os.makedirs(psave)

def spectrogram(tt, ff, pps, indices, key):
    fn = './%s/%s_spectrogram' % (psave, key)
    plt.figure(figsize=(10, 5))
    plt.cla()
    for i, idx in enumerate(indices):
        plt.subplot(1, len(target_snrs), i+1)
        plt.title("SNR ~ %d"%target_snrs[i])
        plt.pcolormesh(tt, ff, pps[idx])
        if i == 1:
            plt.xlabel("Time [s]")
        if i == 0:
            plt.ylabel("Frequency [Hz]")
        if i != 0:
            plt.xticks([2, 4, 6, 8], [2, 4, 6, 8])
            plt.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                left=False,      # ticks along the left edge are off
                right=False,         # ticks along the right edge are off
                labelleft=False) # labels along the bottom edge are off
    plt.subplots_adjust(wspace=0.01)
    # plt.savefig(fn+'.png', bbox_inches='tight', format='png')
    plt.savefig(fn+'.pdf', bbox_inches='tight', format='pdf')
    print("spectrogram is saved at %s"%fn)

def read_snr_pps(fpath, fpath_pp, corr, incorr):
    print("Read parameters from %s"% fpath)
    params = []
    with h5py.File(fpath, 'r') as f:
        params.append(np.array(f['parameters']))

    snr = np.vstack(params).T[-1]
    corr_snr = snr[corr]
    incorr_snr = snr[incorr]

    print("Read pp from %s"% fpath_pp)
    pps = []
    with h5py.File(fpath_pp, 'r') as f:
        pps.append(np.array(f['pp']))
    pps = np.vstack(pps)

    return corr_snr, incorr_snr, pps

def get_indices(corr, incorr, corr_snr, incorr_snr):
    corr_indices = []
    incorr_indices = []
    for tsnr in target_snrs:
        corr_lidx = tsnr - tolerance < corr_snr
        corr_uidx = tsnr + tolerance > corr_snr

        incorr_lidx = tsnr - tolerance < incorr_snr
        incorr_uidx = tsnr + tolerance > incorr_snr

        corr_idx = np.random.choice((corr_lidx*corr_uidx).nonzero()[0], 1)
        incorr_idx = np.random.choice((incorr_lidx*incorr_uidx).nonzero()[0], 1)

        corr_indices.append(corr[corr_idx.item()])
        incorr_indices.append(incorr[incorr_idx.item()])

    return corr_indices, incorr_indices

def get_results(rfn):
    print("Read results from %s"%rfn)
    results = np.load(rfn, allow_pickle=True)
    predicted = results[0]
    labels = results[1]

    ulmask = labels == 0
    mask = labels == 1

    ulpredicted = predicted[ulmask]
    ullabels = labels[ulmask]

    predicted = predicted[mask]
    labels = labels[mask]

    corr = (predicted == labels).nonzero()[0]
    incorr = (predicted != labels).nonzero()[0]

    ulcorr = (ulpredicted == ullabels).nonzero()[0]
    ulincorr = (ulpredicted != ullabels).nonzero()[0]

    return corr, incorr, ulcorr, ulincorr

def main():

    rfd1 = 'Experiments_%s/Results/'%norm_method
    rfd2 = 'results_%s_None_classification_VGG11'%lmodel
    if spinning:
        rfd2.replace('None', 'None-spinning')
    rfn = os.path.join(rfd1, rfd2, 'predicted.npy')

    corr, incorr, ulcorr, ulincorr = get_results(rfn)

    fpath = './data/imrppv2_'+lmodel
    ufpath = './data/imrppv2_None'
    if spinning:
        ufpath += '_spinning'

    fpaths = os.path.join(fpath, 'test_params_lensed.h5')
    fpaths_pp = os.path.join(fpath, 'test_pp_lensed.h5')
    ulfpaths = os.path.join(ufpath, 'test_params_unlensed.h5')
    ulfpaths_pp = os.path.join(ufpath, 'test_pp_unlensed.h5')

    corr_snr, incorr_snr, pps = read_snr_pps(fpaths, fpaths_pp, corr, incorr)
    ulcorr_snr, ulincorr_snr, ulpps = read_snr_pps(ulfpaths, ulfpaths_pp, ulcorr, ulincorr)

    corr_indices, incorr_indices = get_indices(corr, incorr, corr_snr, incorr_snr)
    ulcorr_indices, ulincorr_indices = get_indices(ulcorr, ulincorr, ulcorr_snr, ulincorr_snr)

    ff = np.linspace(f_lower, f_upper, np.shape(pps[0])[0])
    tt = np.linspace(0, sig_dur, np.shape(pps[0])[1])

    spectrogram(tt, ff, pps, corr_indices, 'lensed_corr')
    spectrogram(tt, ff, pps, incorr_indices, 'lensed_incorr')
    spectrogram(tt, ff, ulpps, ulcorr_indices, 'unlensed_corr')
    spectrogram(tt, ff, ulpps, ulincorr_indices, 'unlensed_incorr')

if __name__ == '__main__':
    main()