import itertools
import logging
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde, truncnorm

import utils

logger = logging.getLogger(__name__)

def ROC_Curves(y_test, preds, opt, title='ROC Curves'):
    fname = 'ROC_Curves'
    fsave = os.path.join(opt.plt_fsave, fname)
    import sklearn.metrics as metrics
    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    fpr_nonzero_args = fpr.nonzero()
    fpr = fpr[fpr_nonzero_args]
    tpr = tpr[fpr_nonzero_args]

    with plt.style.context(['science', 'ieee', 'high-vis']):
        plt.figure()
        plt.cla()
        # plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
        plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1],'r--')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        plt.xlim(1e-4, 1.5)
        plt.xscale('log')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.tight_layout()
        plt.savefig(fsave+'.pdf', bbox_inches='tight', format='pdf')
        plt.savefig(fsave+'.png', bbox_inches='tight', format='png')
        logger.info("ROC curves are saved at %s" % fsave)

def ParamHist(y, prdy, opt):
    fname = 'ParamHist'
    fsave = os.path.join(opt.plt_fsave, fname)

    snr_std = 30
    snr_tol = 2

    def _get_params(fname):
        G_SI    = 6.674E-11 # Gravitational constant [m^3/kg/s^2].
        c_SI    = 2.998E8   # Speed of light [m/s].
        SM_SI   = 1.989E30  # Solar mass [kg].
        logging.info("Read SNR from {}".format(fname))
        with h5py.File(fname, 'r') as f:
            params = np.array(f['parameters'], dtype=np.float32).T

        M_ch = params[0]
        M_L = params[1]
        y = params[2]
        D_s = params[4]
        z_L = params[5]
        mu_plus = params[7]
        mu_minus = params[8]
        snr = params[-1]

        M_lz = M_L*SM_SI*(1+z_L)
        if opt.lense_model == 'PML':
            beta = np.sqrt(y*y + 4)
            t_delay = 4*G_SI/(c_SI**3)*M_lz*((0.5*y*beta) + np.log((beta + y)/(beta - y))) # Time delay
        else:
            t_delay = 8*M_lz*y*G_SI/c_SI**3
        return [mu_plus, mu_minus, t_delay, M_ch, D_s], snr

    fparam = 'test_params_%s.h5'
    lfname = os.path.join(opt.dpath, opt.dhead+'_'+opt.lense_model, fparam%'lensed')

    params, snr = _get_params(lfname)

    mask = y == 1 # for lensed samples

    snr_mask1 = snr >= snr_std - snr_tol
    snr_mask2 = snr <= snr_std + snr_tol
    snr_mask = snr_mask1*snr_mask2

    y = y[mask][snr_mask]
    prdy = prdy[mask][snr_mask]

    correct = y == prdy
    incorrect = y != prdy

    prefixs = ["_mu_plus", "_mu_minus", "_t_delay",
               "_chirp_mass", "_source_dist", "_source_redshift"]
    labels = [r"$\mu_{+}$", r"$\mu_{-}$", r"$\Delta t \, [s]$",
              r"$M^{c}_{s}$", r"$D_{s}$"]
    for i, param in enumerate(params):
        temp_fsave = fsave+prefixs[i]
        pc = param[snr_mask][correct]
        pic = param[snr_mask][incorrect]

        min_bin = min(pc.min(), pic.min())
        max_bin = max(pc.max(), pic.max())
        bins = np.linspace(min_bin, max_bin, 51)

        with plt.style.context(['science', 'ieee', 'high-vis']):
            plt.figure()
            plt.cla()
            if i == 2:
                plt.hist(pc, bins=bins, color='b', alpha=0.5, label='Correct', density=True)
                plt.hist(pic, bins=bins, color='r', alpha=0.5, label='Incorrect', density=True)
                plt.yscale('log')
            else:
                plt.hist(pc, bins=bins, color='b', alpha=0.5, label='Correct', density=True)
                plt.hist(pic, bins=bins, color='r', alpha=0.5, label='Incorrect', density=True)
            plt.legend()
            plt.tight_layout()
            plt.ylabel('Number density')
            plt.xlabel(labels[i])
            plt.savefig(temp_fsave+'.pdf', bbox_inches='tight', format='pdf')
            plt.savefig(temp_fsave+'.png', bbox_inches='tight', format='png')

            logging.info("SNR histogram is saved at %s" % temp_fsave)


def SNRHist(y, prd_y, opt):
    fname = 'SNRHist'
    fsave = os.path.join(opt.plt_fsave, fname)

    def _read_snr(fname):
        logging.info("Read SNR from {}".format(fname))
        with h5py.File(fname, 'r') as f:
            params = np.array(f['parameters'], dtype=np.float32).T
            snr = params[-1]
        return snr

    fparam = 'test_params_%s.h5'
    ulfname = os.path.join(opt.dpath, opt.dhead+'_None', fparam%'unlensed')
    lfname = os.path.join(opt.dpath, opt.dhead+'_'+opt.lense_model, fparam%'lensed')
    sfname = ulfname.replace('_None', '_None_spinning')

    ulsnr = _read_snr(ulfname)
    lsnr = _read_snr(lfname)
    ssnr = _read_snr(sfname)

    if opt.triplet_class:
        mask_ul = y == 0
        mask_l = y == 1
        mask_sp = y == 2

        snrs = [ulsnr, lsnr, ssnr]
        masks = [mask_ul, mask_l, mask_sp]
        prefixs = ["_Unlensed", "_Lensed", "_Spinning"]
    else:
        if opt.spinning:
            mask_sp = y == 0
            mask_l = y == 1

            snrs = [ssnr, lsnr]
            masks = [mask_sp, mask_l]
            prefixs = ["_Spinning", "_Lensed"]
        else:
            mask_ul = y == 0
            mask_l = y == 1

            snrs = [ulsnr, lsnr]
            masks = [mask_ul, mask_l]
            prefixs = ["_Unlensed", "_Lensed"]

    for i, mask in enumerate(masks):
        temp_fsave = fsave + prefixs[i]
        temp_y = y[mask]
        temp_py = prd_y[mask]
        temp_snr = snrs[i]

        correct = temp_y == temp_py
        incorrect = temp_y != temp_py

        csnr = temp_snr[correct]
        icsnr = temp_snr[incorrect]

        csnr_mean = np.mean(csnr)
        icsnr_mean = np.mean(icsnr)

        bins = np.linspace(10, 50, 51)

        tot_hist_max = np.max(np.histogram(temp_snr, bins, density=True)[0])
        c_hist_max = np.max(np.histogram(csnr, bins, density=True)[0])
        ic_hist_max = np.max(np.histogram(icsnr, bins, density=True)[0])
        hist_ylim = max(tot_hist_max, c_hist_max, ic_hist_max)+0.1

        annot_x = (icsnr_mean+csnr_mean)/2.
        annot_y = hist_ylim/2.
        snr_dist = str(round(csnr_mean-icsnr_mean, 3))

        if np.abs(icsnr_mean - csnr_mean) < 5:
            annot_x_text = max(csnr_mean, icsnr_mean)+5
        else:
            annot_x_text = annot_x

        with plt.style.context(['science', 'ieee', 'high-vis']):
            plt.figure()
            plt.cla()
            plt.hist(csnr, bins=bins, color='b', alpha=0.5, label='Correct', density=True)
            plt.hist(icsnr, bins=bins, color='r', alpha=0.5, label='Incorrect', density=True)
            plt.vlines([csnr_mean, icsnr_mean], 0, hist_ylim, color=['b', 'r'], linestyles='--')
            plt.hlines(annot_y, csnr_mean, icsnr_mean, color='k', linestyles='--')
            plt.annotate(str(snr_dist).ljust(5, '0'), xy=(annot_x, annot_y+hist_ylim/40.), 
                        xytext=(annot_x_text, annot_y+hist_ylim/5.),
                        ha='center', arrowprops=dict(facecolor='black', shrink=0.05))
            plt.legend()
            plt.tight_layout()
            plt.ylabel('Number density')
            plt.xlabel('SNR')
            plt.savefig(temp_fsave+'.pdf', bbox_inches='tight', format='pdf')
            plt.savefig(temp_fsave+'.png', bbox_inches='tight', format='png')

            logging.info("SNR histogram is saved at %s" % temp_fsave)

def ConfusionMatrix(y, prd_y, opt, title='Confusion Matrix', cmap=plt.cm.Blues, precessing=False, tail=None):
    fname = 'ConfusionMatrix'
    fsave = os.path.join(opt.plt_fsave, fname)

    y = np.array(y, dtype=np.long)
    prd_y = np.array(prd_y, dtype=np.long)

    if opt.triplet_class:
        tick_marks = [r"$U_{N}$", r"$U_{P}$", r"$L$"]

        y = np.where(y == 1, 0.5, y)
        y = np.where(y == 2, 1, y) # unlensed_precessing label from 2 to 1
        y = np.where(y == 0.5, 2, y).astype(np.long) # lensed label from 1 to 2

        prd_y = np.where(prd_y == 1, 0.5, prd_y)
        prd_y = np.where(prd_y == 2, 1, prd_y)
        prd_y = np.where(prd_y == 0.5, 2, prd_y).astype(np.long)
    else:
        if precessing:
            tick_marks = [r"$U_{P}$", r"$L$"]
        else:
            tick_marks = [r"$U_{N}$", r"$L$"]

    nclss = 3 if opt.triplet_class else 2
    cm = np.zeros((nclss, nclss), dtype=int)
    for p, l in zip(y, prd_y):
        cm[p, l] += 1

    cmr = np.zeros(np.shape(cm))
    for i, row in enumerate(cm):
        row_sum = float(np.sum(row))
        cmr[i] = row/row_sum*100

    tick_numbs = np.arange(cm.shape[1])
    with plt.style.context(['science', 'ieee', 'high-vis']):
        plt.figure()
        plt.cla()
        plt.imshow(cmr, interpolation='nearest', cmap=cmap)
        # plt.title(title)
        plt.colorbar()
        ax = plt.gca()
        ax.set_xticklabels((ax.get_xticks()+1).astype(str))
        plt.yticks(tick_numbs, tick_marks)
        plt.xticks(tick_numbs, tick_marks)

        thresh = cmr.max() / 2.

        for i, j in itertools.product(range(cmr.shape[0]), range(cmr.shape[1])):
            plt.text(j, i, "%s\n({%.2f}\%%)" % (cm[i, j], cmr[i, j]),
                    horizontalalignment='center', verticalalignment='center',
                    color='white' if cmr[i, j] > thresh else 'black')
        plt.clim(0, 100)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        if not tail is None:
            fsave += '_'+tail
        plt.savefig(fsave+'.pdf', bbox_inches='tight', format='pdf')
        plt.savefig(fsave+'.png', bbox_inches='tight', format='png')
        logger.info("Confusion matrix is saved at %s" % fsave)

def PPPlot(ys, prds, opt, gkde=False):
    fname = 'PPPlot'
    fsave = os.path.join(opt.plt_fsave, fname)

    fparam = 'test_params_%s.h5'
    lfname = os.path.join(opt.dpath, opt.dhead+'_'+opt.lense_model, fparam%'lensed')
    params = get_params(lfname, opt.lense_model)
    snr = params[1]

    parameters = ["chirp_mass", "lense_mass", "y", "lense_redshift", 
                  "source_redshift", "mu_plus", "mu_minus"]
    labels = [r"$M^{c}_{s}$", r"$M_{L}$", r"$y$", r"$z_{L}$", 
              r"$z_{s}$", r"$\mu_{+}$", r"$\mu_{-}$"]
    scale = ['lin', 'lin', 'lin', 'lin', 'lin', 'lin', 'lin']
    if opt.lense_model == 'PML':
        lim = [[0, 50], [-5000, 100000], [0, 1],
               [0, 0.25], [0, 0.5], [0, 11], [-10, 0]]
    else:
        lim = [[0, 50], [-5000, 100000], [0, 1],
               [0, 0.25], [0, 0.5], [1, 22], [-1, 20]]
    # scale = ['lin']*9

    ys = ys[:-1]
    guide_line = lambda x: x
    for i, (y, prd) in enumerate(zip(ys, prds)):
        corr = np.corrcoef(y, prd)[0, 1]
        with plt.style.context(['science', 'ieee', 'high-vis']):
            plt.figure()
            plt.cla()
            # plt.title(parameters[i].replace("_", " ").upper())
            cmap = plt.cm.get_cmap('jet')

            # if gkde:
            ypy = np.vstack((y, prd))
            yc = gaussian_kde(ypy)(ypy)
            plt.scatter(y, prd,
                        marker='.', c=yc, s=0.5,
                        label=r"$\rho$ = %.3f" % corr,
                        cmap=cmap, linewidth=0.3,
                        edgecolor=None)
            # else:
            #     plt.scatter(y, prd, marker='.', c='b', s=1, label=r"$\rho$ = %.3f" % corr, cmap=cmap)
            # plt.text(1, 1, "CorrCoef: %.3f" % corr, ha='center', va='center',
                    # transform=plt.gca().transAxes)
            gx = np.linspace(min(min(prd), min(y))-10000, max(max(prd), max(y), 1)+10000, 100)
            gy = guide_line(gx)
            plt.plot(gx, gy, color='k', ls='dashed', lw=1)
            plt.xlabel('True '+labels[i])
            plt.ylabel('Predicted '+labels[i])
            if i == 1:
                plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
            # plt.xlim(min(min(prd), min(y)), max(max(prd), max(y)))
            # plt.ylim(min(min(prd), min(y)), max(max(prd), max(y)))
            if scale[i] == 'log':
                plt.xscale('log')
                plt.yscale('log')

            plt.xlim(lim[i])
            plt.ylim(lim[i])
            plt.tight_layout()
            plt.savefig(fsave+"_"+parameters[i]+'.pdf', bbox_inches='tight', format='pdf')
            plt.savefig(fsave+"_"+parameters[i]+'.png', bbox_inches='tight', format='png')
            plt.close('all')
            logger.info("PP Plot for %s problem is saved at %s" % (opt.problem, fsave))

def CumulativeScore(zs, pzs, opt):
    fname = 'CumulativeScore'
    fsave = os.path.join(opt.plt_fsave, fname)

    tol = 0.005
    with plt.style.context(['science', 'ieee', 'high-vis']):
        plt.figure()
        plt.cla()
        for i in range(len(ctypes)):
            y = zs[i].ravel()
            pz = pzs[i].ravel()
            zerr = np.abs(y-pz)
            C = len(y)

            MAE = np.sum(zerr)/C
            itol = 0.0
            CS = []
            while True:
                Cl = float(np.sum(zerr <= itol))
                CS.append(Cl/C*100.)
                itol += tol
                if Cl >= C:
                    break

            x = np.arange(len(CS))*tol
            plt.plot(x, CS, c=c[i], label=(ctypes[i]+' MAE: %.3f' % MAE))

        plt.ylabel("CumulativeScore(%)")
        plt.xlabel("Tolerance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fsave+'.pdf', bbox_inches='tight', format='pdf')
        plt.savefig(fsave+'.png', bbox_inches='tight', format='png')
        logger.info("Cumulative score is saved at %s" % fsave)

def CornerPlot(data, opt):
    fname = 'CornerPlot'
    fsave = os.path.join(opt.dan_fsave, fname)

    parameters = [r"$M^{c}_{s}$", r"$M_{L}$", r"$y$", r"$z_{L}$", 
                  r"$z_{s}$", r"$\mu_{+}$", r"$\mu_{-}$"]

    binsize = 20
    leng = np.shape(data)[0]
    with plt.style.context(['science', 'ieee', 'high-vis']):
        plt.figure(figsize=(10,10))
        plt.cla()
        for i in range(1, leng+1):
            for j in range(i):
                plt.subplot(leng, leng, (i-1)*leng+j+1)
                if i-1 == j:
                    plt.yscale('linear')
                    plt.hist(data[i-1], binsize, histtype='step')
                    if i == 1:
                        plt.ylabel("N")
                    if i == leng:
                        plt.xlabel(parameters[i-1])
                else:
                    plt.yscale('linear') # make sure
                    plt.scatter(data[j], data[i-1],
                                c = 'b',
                                s = 1,
                                marker='.')
                    if i == leng:
                        plt.xlabel(parameters[j])
                    if j == 0:
                        plt.ylabel(parameters[i-1])

                if i != leng:
                    plt.xticks([])
                else:
                    med = (min(data[j])+max(data[j]))/2.
                    plt.xticks([med], (["%.1f"%med]))

                if j != 0:
                    plt.yticks([])
                else:
                    if i != 1:
                        med = (min(data[i-1])+max(data[i-1]))/2.
                        plt.yticks([med], (["%.1f"%med]))

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        plt.savefig(fsave+'.pdf', bbox_inches='tight', format='pdf')
        plt.savefig(fsave+'.png', bbox_inches='tight', format='png')
        logging.info("Corner plot is saved at %s" % fsave)

def RandomScatter(data, opt):
    fname = 'RandomScatter'
    fsave = os.path.join(opt.dan_fsave, fname)

    parameters = [r"$M_{ch}$", r"$M_{L}$", r"$y$", r"$z_{L}$", 
                  r"$z_{s}$", r"$\mu_{+}$", r"$\mu_{-}$"]
    with plt.style.context(['science', 'ieee', 'high-vis']):
        plt.figure()
        plt.cla()
        for i, d in enumerate(data):
            plt.subplot(3, 3, i+1)
            y = np.random.randn(len(d))
            plt.scatter(d, y, c='b', s=1, marker='.')
            plt.xlabel(parameters[i])
        plt.tight_layout()
        plt.savefig(fsave+'.pdf', bbox_inches='tight', format='pdf')
        plt.savefig(fsave+'.png', bbox_inches='tight', format='png')
        logging.info("Random scatter is saved at %s" % fsave)

def PHist(ys, prds, opt, pname=None, title='Parameter distribution'):
    fname = 'PHist'
    fsave = os.path.join(opt.plt_fsave, fname)

    parameters = ["chirp_mass", "lense_mass", "y", "lense_redshift", 
                  "source_redshift", "mu_plus", "mu_minus"]
    xlabels = [r"$M^{c}_{s}$", r"$M_{L}$", "y", r"$z_{L}$", 
              r"$z_{s}$", r"$\mu_{+}$", r"$\mu_{-}$"]
    xscale = ['lin', 'log', 'log', 'lin', 'lin', 'lin', 'lin']

    if opt.lense_model == 'PML':
        xlim = [[0, 50], [0, 100000], [0.05, 1],
                [0, 0.235], [0, 0.5], [1, 11], [-10, 0]]
    else:
        xlim = [[0, 50], [0, 100000], [0.05, 1],
                [0, 0.235], [0, 0.5], [1, 22], [-1, 20]]

    ys = ys[:-1]
    for i, (y, prd) in enumerate(zip(ys, prds)):
        minx = min(min(y), min(prd))
        maxx = max(max(y), max(prd))
        bins = np.linspace(minx, maxx, 1001)
        y_pdf = gaussian_kde(y)(bins)
        prd_pdf = gaussian_kde(prd)(bins)
        pdf_overlap = np.where(y_pdf < prd_pdf, y_pdf, prd_pdf)
        interv = bins[1]-bins[0]
        auc_overlap = np.trapz(pdf_overlap, dx=interv)
        with plt.style.context(['science', 'ieee', 'high-vis']):
            plt.figure()
            plt.cla()
            # plt.hist(y, bins, color='r', label='True', alpha=0.5)
            # plt.hist(prd, bins, color='b', label='Predicted', alpha=0.5)
            # plt.title(parameters[i].replace("_", " ").upper())
            plt.plot(bins, y_pdf, color='r', label="True")
            plt.plot(bins, prd_pdf, color='b', label="Predicted")
            plt.fill_between(bins, y_pdf, color='r', alpha=0.3)
            plt.fill_between(bins, prd_pdf, color='b', alpha=0.3)

            if parameters[i] == "lense_mass":
                plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

            plt.xlabel(xlabels[i])
            plt.ylabel('Number density')
            # if xscale[i] == 'log':
                # plt.xscale(xscale[i])
            # plt.yscale('log')
            leg = plt.legend(loc='upper right')

            plt.xlim(xlim[i])

            plt.text(0.775, 0.75,
                     "Overlap area: %.3f" % auc_overlap,
                     ha='center', va='center',
                     transform=plt.axes().transAxes)

            plt.tight_layout()
            plt.savefig(fsave+"_"+parameters[i]+'.pdf', bbox_inches='tight', format='pdf')
            plt.savefig(fsave+"_"+parameters[i]+'.png', bbox_inches='tight', format='png')
            plt.close('all')
            logger.info("Parameter histogram for %s is saved at %s" % (parameters[i], fsave))


def get_params(fname, lmodel):
        G_SI    = 6.674E-11 # Gravitational constant [m^3/kg/s^2].
        c_SI    = 2.998E8   # Speed of light [m/s].
        SM_SI   = 1.989E30  # Solar mass [kg].
        logging.info("Read SNR from {}".format(fname))
        with h5py.File(fname, 'r') as f:
            params = np.array(f['parameters'], dtype=np.float32).T

        # M_ch = params[0]
        M_L = params[1]
        y = params[2]
        D_L = params[3]
        # D_s = params[4]
        z_L = params[5]
        # mu_plus = params[7]
        # mu_minus = params[8]
        D_LS = params[10]
        snr = params[-1]

        M_lz = M_L*SM_SI*(1+z_L)
        if lmodel == 'PML':
            beta = np.sqrt(y*y + 4)
            t_delay = 4*G_SI/(c_SI**3)*M_lz*((0.5*y*beta) + np.log((beta + y)/(beta - y))) # Time delay
        else:
            t_delay = 8*M_lz*y*G_SI/c_SI**3
        return [t_delay, snr, D_L, D_LS, M_L]


def BiPlot(y, prd, opt, rel=False):
    fname = "BiPlot"
    fsave = os.path.join(opt.plt_fsave, fname)

    parameters = ["chirp_mass", "source_redshift", "mu_plus", "mu_minus"]
    labels = [r"$M^{c}_{s}$", r"$z_{s}$", r"$\mu_{+}$", r"$\mu_{-}$"]

    fparam = 'test_params_%s.h5'
    lfname = os.path.join(opt.dpath, opt.dhead+'_'+opt.lense_model, fparam%'lensed')
    colors = get_params(lfname, opt.lense_model)
    clabels = [r"$\Delta t$", "SNR", r"$D_{L}$", r"$D_{LS}$", r"$M_{L}$"]
    prefixs = ["Delta_t", "SNR", "D_L", "D_LS", "M_L"]
    snr = y[-1]
    y = y[:-1]

    y_resid = np.abs(y[2]-prd[2])

    m_ch_resid = np.abs(y[0]-prd[0])
    z_s_resid = np.abs(y[4]-prd[4])
    mu_p_resid = np.abs(y[5]-prd[5])
    mu_m_resid = np.abs(y[6]-prd[6])

    if rel:
        parameters = [p+"_rel" for p in parameters]
        y_resid /= (np.abs(y[2])+1)
        m_ch_resid /= (np.abs(y[0])+1)
        z_s_resid /= (np.abs(y[4])+1)
        mu_p_resid /= (np.abs(y[5])+1)
        mu_m_resid /= (np.abs(y[6])+1)

    x_resids = [m_ch_resid, z_s_resid, mu_p_resid, mu_m_resid]

    with plt.style.context(['science', 'ieee', 'high-vis']):
        for i in range(5):
            bounds = [0,
                      np.percentile(np.sort(colors[i]), 33.333),
                      np.percentile(np.sort(colors[i]), 66.666),
                      colors[i].max()]
            cmap = matplotlib.colors.ListedColormap(['blue', 'green', 'red'])
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            for j, x_resid in enumerate(x_resids):
                plt.figure()
                plt.cla()
                # cmap = plt.cm.get_cmap('jet', 3)
                sp = plt.scatter(x_resid, y_resid,
                                 c=colors[i], cmap=cmap,
                                 marker='.', alpha=0.5,
                                 s=0.5, linewidth=0.3,
                                 norm=norm, edgecolor=None)
                if rel:
                    plt.ylabel("Relative residual of y")
                    plt.xlabel("Relative residual of %s" % labels[j])
                else:
                    plt.ylabel("Absolute residual of y")
                    plt.xlabel("Absolute residual of %s" % labels[j])
                cbar = plt.colorbar(sp)
                cbar.set_label(clabels[i])

                temp_sn = fsave+'_'+parameters[j]+'_C'+prefixs[i]
                if rel:
                    temp_sn += '_Rel'
                plt.savefig(temp_sn)
                plt.savefig(temp_sn+'.pdf', format='pdf', dpi=300)
                plt.close('all')
                logger.info("Biplot for %s is saved at %s" % (parameters[j], temp_sn))

def PRPlot(ys, prds, opt, gkde=False):
    fname = 'PRPlot'
    fsave = os.path.join(opt.plt_fsave, fname)

    fparam = 'test_params_%s.h5'
    lfname = os.path.join(opt.dpath, opt.dhead+'_'+opt.lense_model, fparam%'lensed')
    params = get_params(lfname, opt.lense_model)
    snr = params[1]
    params = ["chirp_mass", "lense_mass", "y", "lense_redshift", 
              "source_redshift", "mu_plus", "mu_minus"]
    labels = [r"$M^{c}_{s}$", r"$M_{L}$", r"$y$", r"$z_{L}$", 
              r"$z_{s}$", r"$\mu_{+}$", r"$\mu_{-}$"]

    if opt.lense_model == 'PML':
        xlim = [[0, 50], [0, 100000], [0.05, 1],
                [0, 0.235], [0, 0.5], [1, 11], [-10, 0]]
        ylim = [[-10, 10], [-50000, 50000], [-0.5, 0.5],
                [-0.125, 0.125], [-0.25, 0.25], [-6, 6], [-5, 5]]
    else:
        xlim = [[0, 50], [0, 100000], [0.05, 1],
                [0, 0.235], [0, 0.5], [1, 22], [-1, 20]]
        ylim = [[-25, 25], [-50000, 50000], [-0.5, 0.5],
               [-0.125, 0.125], [-0.25, 0.25], [-11, 11], [-11, 11]]

    ys = ys[:-1]
    for i, (y, prd) in enumerate(zip(ys, prds)):
        with plt.style.context(['science', 'ieee', 'high-vis']):
            plt.figure()
            plt.cla()
            cmap = plt.cm.get_cmap('jet')

            resid = prd-y
            std = np.std(resid)
            ypy = np.vstack((y, resid))
            yc = gaussian_kde(ypy)(ypy)
            sp = plt.scatter(y, resid,
                             marker='.', c=yc, s=0.5,
                             cmap=cmap, linewidth=0.3,
                             edgecolor=None)

            cbar = plt.colorbar(sp)
            cbar.set_label('density')

            plt.hlines([-std, std], xlim[i][0], xlim[i][1],
                       linestyle='dashed', linewidth=0.7, color='k')
            plt.hlines(0, xlim[i][0], xlim[i][1],
                       linestyle='solid', linewidth=0.7, color='k')
            plt.xlabel('True '+labels[i])
            plt.ylabel('Residual of '+labels[i])
            if i == 1:
                plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

            plt.xlim(xlim[i])
            plt.ylim(ylim[i])

            plt.tight_layout()
            plt.savefig(fsave+"_"+params[i]+'.pdf', format='pdf')
            plt.savefig(fsave+"_"+params[i]+'.png', format='png')
            plt.close('all')
            logger.info("PR Plot for %s problem is saved at %s" % (opt.problem, fsave))
