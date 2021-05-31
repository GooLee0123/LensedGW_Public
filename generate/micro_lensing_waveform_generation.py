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
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.pcolormesh(tt, ff, pp)
    ax.axis('tight')
    ax.axis('off')
    fig.canvas.draw()
    plt.savefig('Micro_lensed_spec', pad_inches=0)
    plt.close('all')

fs = 4096.
f_lower = 20.
f_upper = 512.
delta_f = 1/256.
flen = int(fs//2/delta_f)+1
psd = pycbc.psd.AdVDesignSensitivityP1200087(flen, delta_f, f_lower)
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

# Coordinates, first define them in scaled units [x (radians) /thetaE_tot]
y0,y1 = 0.1,0.5*np.sqrt(3)
l0,l1 = 0.5,0

# Redshift
zS = 0.04
zL = 0.02

# Masses
mL1  = 100
mL2  = 100
mtot = mL1+mL2

# binary point mass images, in radians
# ra  = np.array([2.06184855e-11,  6.74286421e-11, -8.55036309e-11])
# dec = np.array([2.04174704e-10, -6.17971410e-11, -5.67605886e-11])

thetaE1 = param_processing(zL, zS, mL1)
thetaE2 = param_processing(zL, zS, mL2)
thetaE  = param_processing(zL, zS, mtot)

solver_kwargs = {'SearchWindowMacro': 4*thetaE1, 'SearchWindow': 4*thetaE2}

beta0, beta1 = y0*thetaE,y1*thetaE
eta10, eta11 = l0*thetaE,l1*thetaE
eta20, eta21 = -l0*thetaE,l1*thetaE

# lens model
lens_model_list = ['POINT_MASS', 'POINT_MASS']
kwargs_point_mass_1 = {'center_x': eta10,'center_y': eta11, 'theta_E': thetaE1}
kwargs_point_mass_2 = {'center_x': eta20,'center_y': eta21, 'theta_E': thetaE2}
kwargs_lens_list = [kwargs_point_mass_1, kwargs_point_mass_2]

Img_ra, Img_dec, _, _, _  = microimages(source_pos_x=beta0,
                                        source_pos_y=beta1,
                                        lens_model_list=lens_model_list,
                                        kwargs_lens=kwargs_lens_list,
                                        **solver_kwargs)

####################
# lensed waveforms #
####################

# read the waveform parameters
config_file = 'ini_files/waveform_config.ini'

# instantiate the waveform model
waveform_model = gw_signal(config_file)

# compute the lensed waveform polarizations, strains in the requested detectors and their frequencies
freqs_lensed, hp_tilde_lensed, hc_tilde_lensed, lensed_strain_dict = \
    waveform_model.lensed_gw(Img_ra, Img_dec, beta0, beta1, zL, zS, lens_model_list, kwargs_lens_list)

# and their signal-to-noise-ratios
lensed_SNR_dict = waveform_model.lensed_snr(Img_ra, Img_dec,
                                            beta0, beta1,
                                            zL, zS,
                                            lens_model_list,
                                            kwargs_lens_list)

# access a lensed strain
lensed_sH1 = lensed_strain_dict['H1']

delta_f = freqs_lensed[1] - freqs_lensed[0]
fseries_sH1 = pw.FrequencySeries(lensed_sH1, delta_f=delta_f)
tseries_sH1 = fseries_sH1.to_timeseries(delta_t=1/4096.)

nhl = get_noise(len(tseries_sH1), tseries_sH1.delta_t, tseries_sH1.start_time)

snr = cal_SNR(tseries_sH1, nhl)

print("[GOOLEE]: %s" % snr)

tseries_snH1 = tseries_sH1 + nhl
tseries_snH1 = tseries_snH1[len(tseries_snH1)-8*4096:]

tt, ff, pp = tseries_snH1.qtransform(1/32.,
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