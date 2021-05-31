import numpy as np

import pycbc.noise
import pycbc.psd
import pycbc.waveform as pw
from lensinggw.solver.images import microimages
from lensinggw.utils.utils import param_processing
from lensinggw.waveform.waveform import gw_signal


def plotstft(tt, ff, pp):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rc('text', usetex=False)
    matplotlib.rc('font', serif='Times New Roman')
    matplotlib.rc('font', size=15)
    matplotlib.rc('axes', labelsize=18)
    matplotlib.rc('xtick', labelsize=15)
    matplotlib.rc('ytick', labelsize=15)
    matplotlib.rc('legend', fontsize=12)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    # fig.subplots_adjust(left=0.9, right=0.9, bottom=0.9, top=1)
    ax.pcolormesh(tt, ff, pp)
    ax.set(xlabel="Time [s]", ylabel= "Frequency [Hz]")
    # ax.axis('tight')
    # ax.axis('off')
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    # ax.set_yticks([32, 64, 128, 256, 512])
    fig.canvas.draw()
    plt.tight_layout()
    plt.savefig('Micro_lensed_spec', pad_inches=0, dpi=300)
    plt.savefig('Micro_lensed_spec.pdf', pad_inches=0, format='pdf', dpi=300)
    plt.close('all')

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


fs = 4096.
f_lower = 20.
f_upper = 512.
delta_f = 1/8.
flen = int(fs//2/delta_f)+1
psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, f_lower)
def get_noise(nlen, del_t, stime):
    ts = pycbc.noise.noise_from_psd(nlen, del_t, psd)
    ts.start_time = stime
    return ts

array = np.genfromtxt('microlensed_waveform_freq_domain.txt').T
freqs_lensed = array[0]
lensed_sH1_real = array[1]
lensed_sH1_img = array[2]
lensed_sH1 = lensed_sH1_real+1j*lensed_sH1_img

delta_f = freqs_lensed[1] - freqs_lensed[0]
fseries_sH1 = pw.FrequencySeries(lensed_sH1, delta_f=delta_f)
tseries_sH1 = fseries_sH1.to_timeseries(delta_t=1/4096.)

nhl = get_noise(len(tseries_sH1), tseries_sH1.delta_t, tseries_sH1.start_time)

snr = cal_SNR(tseries_sH1, nhl)

print("[GOOLEE]: %s" % snr)

tseries_snH1 = tseries_sH1 + nhl
tseries_snH1 = tseries_snH1[len(tseries_snH1)-8*4096:]

tt, ff, pp = tseries_sH1.qtransform(1/32.,
                                    logfsteps=256,
                                    qrange=(8, 8),
                                    frange=(f_lower, f_upper))

plotstft(tt, ff, pp)

pps = []
pps.append(pp)

def save_data(pps):
    import os
    import tables

    pps = np.array(pps)

    norm_pps = pps
    pps_dim = np.shape(norm_pps[0])
    dlen = len(norm_pps)

    print(np.shape(norm_pps))

    iidx = 0
    print_every = 100

    sdn = './data/imrppv2_MICRO'
    if not os.path.exists(sdn):
        os.makedirs(sdn)

    fn = os.path.join(sdn, 'test_pp_lensed.h5')
    fpp = tables.open_file(fn, mode='w')
    atom = tables.Float32Atom()
    array_c = fpp.create_earray(fpp.root, 'pp', atom, (0, pps_dim[0], pps_dim[1]))

    iidx = 0
    print_every = 100
    for j in range(iidx, iidx+dlen):
        if j % print_every == 0:
            print("%sth pp are appended to the files" % j)
        ppx = np.expand_dims(norm_pps[j], 0)
        array_c.append(ppx)
        iidx += 1

    fpp.close()

save_data(pps)