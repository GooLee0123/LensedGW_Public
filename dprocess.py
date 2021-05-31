import os
import h5py
import numpy as np

pp_plot = False
lensed = False
spinning = True
# norm_method = 'feature_wise'
lmodel = 'PML' if lensed else 'None'
fpath = './data/imrppv2_'+lmodel
if spinning:
    fpath += '_spinning'
# fpath += '_'+norm_method
fnames = os.listdir(fpath)
fpaths = []
fpaths_pp = []
for fname in fnames:
    if 'params' in fname:
        fpaths.append(os.path.join(fpath, fname))
    if pp_plot and 'pp' in fname:
        fpaths_pp.append(os.path.join(fpath, fname))

params = []
for path in fpaths:
    with h5py.File(path, 'r') as f:
        params.append(np.array(f['parameters']))

if pp_plot:
    if lensed:
        psave = 'lplots_'+lmodel
    else:
        psave = 'uplots_'+lmodel
    if spinning:
        psave += '_spinning'
    if not os.path.exists(psave):
        os.makedirs(psave)
    sig_dur = 8
    f_lower = 20
    f_upper = 512
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pps = []
    for path in fpaths_pp:
        with h5py.File(path, 'r') as f:
            pps.append(np.array(f['pp']))
    pps = np.vstack(pps)
    print(np.shape(pps))
    ff = np.linspace(f_lower, f_upper, np.shape(pps[0])[0])
    tt = np.linspace(0, sig_dur, np.shape(pps[0])[1])
    indices = np.random.choice(len(pps), 100, replace=False)
    for i in indices:
        plt.figure()
        plt.cla()
        plt.pcolormesh(tt, ff, pps[i])
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.savefig('./%s/%s' % (psave, i))
        plt.close('all')

params = np.vstack(params).T
mins = []
maxs = []
for p in params:
    mins.append(np.min(p))
    maxs.append(np.max(p))

min_max = np.vstack((mins, maxs))
print("min_max")
print(min_max)

spath = os.path.join(fpath, 'min_max.npy')
np.save(spath, min_max)