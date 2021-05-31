import argparse
import logging
import os
import shutil
import time
import itertools

import numpy as np
# import torch
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.utils import data

# import models
import plotter
import utils
# from checkpoint import Checkpoint
# from optim import Optimizer

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(
    format=LOG_FORMAT,
    level=getattr(logging, LOG_LEVEL)
)

column_indices = [0, 1, 2, 5, 6, 7, 8]

################ For Train ################
def prepare_optim(model, opt):
    # setting optimizer
    optimizer = Optimizer(
                        torch.optim.Adam(model.parameters(),
                        lr=0.00008,
                        betas=(0.5, 0.999),
                        weight_decay=5e-5),
                        max_grad_norm=5
                        )
    # setting scheduler of optimizer for learning rate decay.
    scheduler = ReduceLROnPlateau(optimizer.optimizer,
                                patience=5,
                                factor=0.5,
                                min_lr=0.000001)
    optimizer.set_scheduler(scheduler)

    return optimizer

def prepare_loss(opt):
    # Typo check
    problem = opt.problem
    if problem != 'regression' and problem != 'MNIST' and problem != 'classification':
        raise ValueError("Problem should be either of regression or classification")

    if opt.classification:
        opt.criterion = getattr(torch.nn, opt.cls_loss)(reduction='mean')
        opt.loss = opt.cls_loss
    else:
        if opt.reg_loss == 'Adaptive':
            opt.criterion = AdaptiveLossFunction(num_dims=1,
                                                float_dtype=np.float32,
                                                device='cuda:%s'%opt.gpuid).lossfun
        else:
            opt.criterion = getattr(torch.nn, opt.reg_loss)(reduction='mean')
        opt.loss = opt.reg_loss
    return opt

def prepare_model(opt):
    # model_names = sorted(name for name in models.__dict__ if name.islower() and
    #     not name.startswith("__") and name.startswith("vgg")
    #     and callable(models.__dict__[name]))

    n_classes = 2 if opt.classification else 8
    if opt.classification and opt.triplet_class:
        n_classes += 1

    if 'vgg' in opt.model.lower():
        logging.info("Import VGG model")
        model = models.__dict__[opt.model.lower()](n_classes=n_classes)
    elif 'inception' in opt.model.lower():
        logging.info("Import Inception model")
        import torchvision
        from models import BasicConv2d, BasicLinear

        model = torchvision.models.inception_v3(pretrained=False)
        model.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=(1,3), stride=(2,2))
        model.fc = BasicLinear(2048, n_classes, opt.classification)
        model.aux_logits = False

    return model

def loss_dictionaries(opt):
    ldic = {}
    # *_loss_every: [avg_*_loss2]
    ldic['train_loss_every'] = []
    ldic['val_loss_every'] = []
    # tot_*_group_loss: tot_*_print_loss
    ldic['tot_train_group_loss'] = np.array([0.])
    ldic['tot_val_group_loss'] = np.array([0.])
    # tot_*_epoch_loss: [tot_*_epoch_loss, count_for_average]
    ldic['tot_train_epoch_loss'] = [0., 0.]
    ldic['tot_val_epoch_loss'] = [0., 0.]

    return ldic

def save_losses(ldic, opt):
    flog = os.path.join(opt.test_epath, 'Losses', 'loss'+opt.token)
    ftloss = 'train'
    fvloss = 'val'
    if not os.path.exists(flog):
        os.makedirs(flog)
    with open(os.path.join(flog, ftloss), 'a') as ftl:
        np.savetxt(ftl, np.array(ldic['train_loss_every']).T, fmt='%.8f')
    with open(os.path.join(flog, fvloss), 'a') as fvl:
        np.savetxt(fvl, np.array(ldic['val_loss_every']).T, fmt='%.8f')

def get_normalization_args(pps, opt):
    if opt.norm_method == 'feature_wise':
        for i in range(pps.shape[1]):
            if i == 0:
                means, stds = [], []
            mean = torch.mean(pps[:, i], 0)
            std = torch.std(pps[:, i], 0)
            means.append(mean)
            stds.append(std)
        args = (means, stds)
    elif opt.norm_method == 'sample_wise':
        args = None
    else: # global_wise
        ppmin = torch.min(pps)
        ppmax = torch.max(pps)
        args = (ppmin, ppmax)

    return args

def train(model, optim, opt):
    logging.info("Train")

    model.to(opt.device)
    model.train(True)

    dparams = {'batch_size': opt.bsize,
               'shuffle': True,
               'num_workers': 6}
    vdparams = {'batch_size': opt.bsize,
               'shuffle': False,
               'num_workers': 6}
    max_training_epoch = opt.max_training_epoch

    # Load train data and launch batch generator
    training_set = utils.Dataset('training', opt, lensed=opt.lensed)
    # Load val data and launch batch generator
    validation_set = utils.Dataset('validation', opt, lensed=opt.lensed)

    pps = training_set.pp

    # Unlensed for the classification problem case
    if opt.classification:
        UL_training_set = utils.Dataset('training', opt, lensed=False)
        UL_validation_set = utils.Dataset('validation', opt, lensed=False)
        pps = torch.cat((pps, UL_training_set.pp), 0)

        if opt.triplet_class:
            opt.spinning = True
            UL_spinning_training_set = utils.Dataset('training', opt, lensed=False)
            UL_spinning_validation_set = utils.Dataset('validation', opt, lensed=False)
            pps = torch.cat((pps, UL_spinning_training_set.pp), 0)
            opt.spinning = False

    args = get_normalization_args(pps, opt)

    training_set.input_norm(args)
    validation_set.input_norm(args)
    if opt.classification:
        UL_training_set.input_norm(args)
        UL_validation_set.input_norm(args)
        if opt.triplet_class:
            UL_spinning_training_set.input_norm(args)
            UL_spinning_validation_set.input_norm(args)

    training_generator = data.DataLoader(training_set, **dparams)
    validation_generator = data.DataLoader(validation_set, **vdparams)

    batch_max_epoch = int(len(training_set)/dparams['batch_size'])+1 # batch iteraition number
    checkpoint_criterion = np.inf # minimum validation loss placeholder
    step = 0 # The number of backpropagation
    ldic = loss_dictionaries(opt) # Loss dictionary
    stime = time.time() # start time
    for epoch in range(max_training_epoch):
        #Training
        if opt.classification:
            UL_training_generator = iter(data.DataLoader(UL_training_set, **dparams))

            if opt.triplet_class:
                UL_spinning_training_generator = iter(data.DataLoader(UL_spinning_training_set, **dparams))

        for Be, (local_batch, local_y) in enumerate(training_generator):
            if opt.classification:
                UL_batch, UL_y = next(UL_training_generator)
                local_batch = torch.cat((local_batch, UL_batch), dim=0) # [bsize, 256, 256] -> [2*bsize, 256, 256]
                local_y = torch.cat((local_y, UL_y), dim=0) # [bsize, 1] -> [2*bsize, 1]

                if opt.triplet_class:
                    UL_spinning_batch, UL_spinning_y = next(UL_spinning_training_generator)
                    local_batch = torch.cat((local_batch, UL_spinning_batch), dim=0) # [2*bsize, 256, 256] -> [3*bsize, 256, 256]
                    local_y = torch.cat((local_y, UL_spinning_y), dim=0) # [2*bsize, 1] -> [3*bsize, 1]

            local_batch = local_batch.unsqueeze(1) # [n*bsize, 256, 256] -> [n*bsize, 1, 256, 256]

            local_batch = local_batch.to(opt.device)
            local_y = local_y.to(opt.device)

            optim.zero_grad()
            # input into model
            outputs = model(local_batch)
            if opt.classification:
                loss = opt.criterion(outputs, local_y)
            else:
                if opt.reg_loss == 'Adaptive':
                    loss = torch.mean(opt.criterion(outputs.view(-1, 1)-local_y.view(-1, 1)))
                else:
                    loss = opt.criterion(outputs, local_y)

            # update total train loss and its denominator for average
            ldic['tot_train_epoch_loss'][0] += loss
            ldic['tot_train_epoch_loss'][1] += 1
            # total group loss for log
            ldic['tot_train_group_loss'][0] += loss.item()

            loss.backward()
            optim.step()

            if step != 0 and step % opt.print_every == 0:
                avg_train_group_loss = ldic['tot_train_group_loss'][0]/opt.print_every
                ldic['train_loss_every'].append(avg_train_group_loss)
                ldic['tot_train_group_loss'] = np.array([0.])

                for param in optim.param_groups():
                    lr = param['lr'] # learning rate.

                # log messages
                log_msg = "Step: %d/%d, Progress %d%%, " % (
                    Be,
                    batch_max_epoch,
                    float(epoch)/(max_training_epoch)*100)
                log_msg += "loss (%s): %.5f, " % (
                        opt.loss, avg_train_group_loss)
                log_msg += "learning rate: %.6f" % lr

                # print logs
                logging.info(log_msg)

            if step != 0 and step % opt.validate_every == 0:
                model.eval()
                if opt.classification:
                    UL_validation_generator = iter(data.DataLoader(UL_validation_set, **vdparams))
                    if opt.triplet_class:
                        UL_spinning_validation_generator = iter(data.DataLoader(UL_spinning_validation_set, **dparams))

                with torch.set_grad_enabled(False):
                    val_step = 0
                    val_ntot = 0
                    val_correct = 0
                    for local_batch, local_y in validation_generator:
                        if opt.classification:
                            UL_batch, UL_y = next(UL_validation_generator)
                            local_batch = torch.cat((local_batch, UL_batch), dim=0) # [bsize, 256, 256] -> [2*bsize, 256, 256]
                            local_y = torch.cat((local_y, UL_y), dim=0) # [bsize] -> [bsize*2]
                            if opt.triplet_class:
                                UL_spinning_batch, UL_spinning_y = next(UL_spinning_validation_generator)
                                local_batch = torch.cat((local_batch, UL_spinning_batch), dim=0) # [2*bsize, 256, 256] -> [3*bsize, 256, 256]
                                local_y = torch.cat((local_y, UL_spinning_y), dim=0) # [2*bsize, 1] -> [3*bsize, 1]

                        local_batch = local_batch.unsqueeze(1) # [bsize, 256, 256] -> [bsize, 1, 256, 256]
                        local_batch = local_batch.to(opt.device)
                        local_y = local_y.to(opt.device)

                        outputs = model(local_batch)

                        if opt.classification:
                            val_loss = opt.criterion(outputs, local_y)
                        else:
                            if opt.reg_loss == 'Adaptive':
                                val_loss = torch.mean(opt.criterion(outputs.view(-1, 1)-local_y.view(-1, 1)))
                            else:
                                val_loss = opt.criterion(outputs, local_y)

                        if opt.problem == 'classification':
                            _, prd = torch.max(outputs.exp(), 1)
                            # prd = (outputs >= 0.5).float()
                            val_correct += (prd == local_y).sum().item()
                            val_ntot += prd.size(0)

                        # update total validation loss and its denominator for average
                        ldic['tot_val_epoch_loss'][0] += val_loss
                        ldic['tot_val_epoch_loss'][1] += 1
                        ldic['tot_val_group_loss'][0] += val_loss.item()

                        val_step += 1

                    avg_val_group_loss = ldic['tot_val_group_loss'][0]/val_step
                    ldic['val_loss_every'].append(avg_val_group_loss)
                    ldic['tot_val_group_loss'] = np.array([0.])

                    if opt.classification:
                        checkpoint_tester = -100.*val_correct/val_ntot
                    else:
                        checkpoint_tester = avg_val_group_loss

                    if checkpoint_tester < checkpoint_criterion:
                        checkpoint_criterion = checkpoint_tester
                        checkpoint = Checkpoint(step, epoch, model, optim, opt=opt)
                        checkpoint.save()

                    logging.info("current validation loss: %.5f" % \
                                avg_val_group_loss)

                    if opt.classification:
                        logging.info("current validation acc: %.3f%%, best acc: %.3f%%" % \
                                (-checkpoint_tester, -checkpoint_criterion))
                    else:
                        logging.info("current minimum loss: %.5f" % checkpoint_criterion)

                    if -checkpoint_criterion >= 99.999:
                        logging.info("Accuracy reached desired level. Early quitting learning.")
                        break

            model.train(True)
            step += 1

        # averaged epoch losses for training and validation
        train_epoch_avg_loss = ldic['tot_train_epoch_loss'][0]/ldic['tot_train_epoch_loss'][1]
        val_epoch_avg_loss = ldic['tot_val_epoch_loss'][0]/ldic['tot_val_epoch_loss'][1]

        # initialize total epoch loss dictionries
        ldic['tot_train_epoch_loss'] = [0., 0.]
        ldic['tot_val_epoch_loss'] = [0., 0.]

        if epoch >= opt.lr_decay_epoch:
            optim.update(avg_val_group_loss, epoch)

        log_msg = "Finished epoch %d, train loss: %.5f, " % \
                    (epoch, train_epoch_avg_loss)
        log_msg += "validation loss: %.5f, " % \
                    (val_epoch_avg_loss)
        
        logging.info(log_msg)

        save_losses(ldic, opt) # save losses every epoch

        if -checkpoint_criterion >= 99.999:
            break

    etime = time.time() # end time
    dur = etime - stime # training time
    logging.info("Training ended. Took about %.3fh" % (dur/3600.))

################ For Test ################
def set_loaded_model(model, optim=None, opt=None):
    resume_checkpoint = Checkpoint.load(model,
                                        optim=optim,
                                        opt=opt)
    model = resume_checkpoint.model
    model.to(opt.device)

    if optim:
        optim = resume_checkpoint.optim
        return model, optim
    else:
        return model

def kl_divergence(org, sim):
    from scipy.stats import gaussian_kde

    bin_min = min(org.min(), sim.min())
    bin_max = max(org.max(), sim.max())
    nbin = 50
    bins = np.linspace(bin_min, bin_max, nbin)
    bw = bins[1]-bins[0]

    p = gaussian_kde(org)(bins)*bw
    q = gaussian_kde(sim)(bins)*bw

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def Bhattacharyya_distance(org, sim):
    from scipy.stats import gaussian_kde

    bin_min = min(org.min(), sim.min())
    bin_max = max(org.max(), sim.max())
    nbin = 50
    bins = np.linspace(bin_min, bin_max, nbin)
    bw = bins[1]-bins[0]

    p = gaussian_kde(org)(bins)*bw
    q = gaussian_kde(sim)(bins)*bw

    return -np.log(np.sum(np.sqrt(p*q)))

def metric_results(y, prd, opt):
    if not os.path.exists(opt.plt_fsave):
        os.makedirs(opt.plt_fsave)
    spath = os.path.join(opt.plt_fsave, 'metrics.txt')

    nfeature = y.shape[0]-1 # [7, ndata], exclude snr
    residuals = ['Residuals']
    kl_divergences = ['KL-divergence']
    bdists = ['Bhattacharyya distance']

    for nf in range(nfeature):
        temp_y = y[nf]
        temp_prd = prd[nf]

        # residual
        residual = np.mean(np.abs(temp_prd-temp_y)/(temp_y+1))
        residuals.append(round(residual, 5))

        # kl-divergence
        kld = kl_divergence(temp_y, temp_prd)
        kl_divergences.append(round(kld, 5))

        # Bhattacharyya distance
        bdist = Bhattacharyya_distance(temp_y, temp_prd)
        bdists.append(round(bdist, 5))

    heads = ["", "Chirp mass,", "Lense mass,", "y,", "Lense redshift,", 
             "Source redshift,", "mu(+),", "mu(-)"]
    outputs = np.vstack((heads, residuals, kl_divergences, bdists))
    np.savetxt(spath, outputs, fmt='%s')
    logging.info("metric is saved at %s" % spath)

def save_results(prd, opt):
    rdir = 'results'+opt.token
    spath = os.path.join(opt.test_epath, opt.test_spath, rdir)
    if not os.path.exists(spath):
        os.makedirs(spath)
    spath = os.path.join(spath, 'predicted')

    n = 2 if opt.classification else 9
    np.save(spath, prd)
    logging.info("%s output is saved at %s.npy" % (opt.problem, spath))

def test(model, opt):
    logging.info("Test")

    # model setting
    model = set_loaded_model(model, opt=opt)
    model.eval()

    max_training_epoch = opt.max_training_epoch

    # Load train data for normalization
    if opt.norm_method != 'sample_wise':
        training_set = utils.Dataset('training', opt, lensed=opt.lensed)
        train_pps = training_set.pp
        if opt.classification:
            UL_training_set = utils.Dataset('training', opt, lensed=False)
            train_pps = torch.cat((train_pps, UL_training_set.pp), 0)
            if opt.triplet_class:
                opt.spinning = True
                UL_spinning_training_set = utils.Dataset('training', opt, lensed=False)
                train_pps = torch.cat((train_pps, UL_spinning_training_set.pp), 0)
                opt.spinning = False

        args = get_normalization_args(train_pps, opt)
    else:
        args = None

    tdparams = {'batch_size': 128,
               'shuffle': False,
               'num_workers': 6}

    test_set = utils.Dataset('test', opt, lensed=opt.lensed)
    test_set.input_norm(args)
    test_generator = data.DataLoader(test_set, **tdparams)
    L_y = test_set.y.numpy().ravel() # lensed

    if opt.classification:
        UL_test_set = utils.Dataset('test', opt, lensed=False)
        UL_test_set.input_norm(args)
        UL_test_generator = iter(data.DataLoader(UL_test_set, **tdparams))
        UL_y = UL_test_set.y.numpy().ravel() # unlensed

        if opt.triplet_class:
            opt.spinning = True
            UL_spinning_test_set = utils.Dataset('test', opt, lensed=False)
            UL_spinning_test_set.input_norm(args)
            UL_spinning_test_generator = iter(data.DataLoader(UL_spinning_test_set, **tdparams))
            UL_spinning_y = UL_spinning_test_set.y.numpy().ravel()
            opt.spinning = False

            test_generator = itertools.chain(UL_test_generator, test_generator, UL_spinning_test_generator)
            y = np.hstack((UL_y, L_y, UL_spinning_y))
        else:
            test_generator = itertools.chain(UL_test_generator, test_generator)
            y = np.hstack((UL_y, L_y))

    else:
        y = test_set.y.numpy()

    prd_arr = []
    with torch.set_grad_enabled(False):
        temp_y = []
        for local_batch, local_y in test_generator:
            local_batch = local_batch.to(opt.device)
            local_batch = local_batch.unsqueeze(1) # [bsize, 256, 256] -> [bsize, 1, 256, 256]

            prd = model(local_batch)
            if opt.classification:
                prd = prd.exp()
            prd_arr.append(prd.cpu().numpy())

    if opt.classification:
        y = np.ravel(y)
        prd_arr = np.vstack(prd_arr)

        print(prd_arr.shape)
        predicted = np.argmax(prd_arr, 1)
        print(np.sum(prd_arr, 1))
        print("predicted: ", predicted)
        print("y: ", y)
        correct = (predicted == y).sum()
        tot = float(len(y))
        acc = correct/tot*100.
        print("correct: %.3f, tot: %.3f"%(correct, tot))
        print("test accuracy: %.3f" % acc)

        prd_arr = np.vstack((predicted, y))

    else:
        prd_arr = np.vstack(prd_arr)
        prd_arr = test_set.target_unnorm(prd_arr)
        # prd_arr[prd_arr == 0] = 1e-5

    save_results(prd_arr, opt)

################ For Plot ################
def load_results(opt):
    fpath = 'results'+opt.token
    fname = 'predicted.npy'
    fdir = os.path.join(opt.test_epath, opt.test_spath, fpath, fname)
    logging.info("Load data from %s" % fdir)
    prd = np.load(fdir)
    return prd

def plot_results(opt):
    plt_subpath = 'plots'+opt.token
    opt.plt_fsave = os.path.join(opt.test_epath, opt.test_ppath, plt_subpath)
    # make directories for test outputs
    if not os.path.exists(opt.plt_fsave):
        os.makedirs(opt.plt_fsave)

    prd = load_results(opt)

    if opt.classification:
        predicted = prd[0]
        y = prd[1]
        if not opt.triplet_class:
            plotter.ROC_Curves(y, predicted, opt)
            plotter.SNRHist(y, predicted, opt)

        precessing = True if opt.spinning else False
        plotter.ConfusionMatrix(y, predicted, opt, precessing=precessing)
        plotter.ParamHist(y, predicted, opt)

        # opt.problem = 'regression'
        # opt.lensed = True
        # test_set = utils.Dataset('test', opt, lensed=opt.lensed)
        # params = test_set.y.numpy() # [1500, 7]
        # params = test_set.target_unnorm(params)
        # Ly = params.T[2]

        # mask1 = Ly < 1
        # mask2 = Ly >= 1

        # mask1 = np.hstack((np.array([True]*(len(y)-len(Ly))), mask1))
        # mask2 = np.hstack((np.array([True]*(len(y)-len(Ly))), mask2))

        # plotter.ConfusionMatrix(y[mask1], predicted[mask1], opt, precessing=precessing, tail='Y_lt_1')
        # plotter.ConfusionMatrix(y[mask2], predicted[mask2], opt, precessing=precessing, tail='Y_ge_1')
    else:
        test_set = utils.Dataset('test', opt, lensed=opt.lensed)
        y = test_set.y.numpy() # [1500, 9]
        y = test_set.target_unnorm(y)
        y, prd = y.T, prd.T

        metric_results(y, prd, opt)

        # plotter.BiPlot(y, prd, opt)
        # plotter.BiPlot(y, prd, opt, rel=True)
        plotter.PHist(y, prd, opt)
        plotter.PRPlot(y, prd, opt, gkde=opt.gkde)

def data_plot(opt):
    plt_subpath = 'danalysis'+opt.token
    opt.dan_fsave = os.path.join(opt.test_epath, opt.test_ppath, plt_subpath)
    # make directories for danalysis outputs
    if not os.path.exists(opt.dan_fsave):
        os.makedirs(opt.dan_fsave)
    
    # train_set = utils.Dataset('training', opt, lensed=opt.lensed)
    # val_set = utils.Dataset('validation', opt, lensed=opt.lensed)
    test_set = utils.Dataset('test', opt, lensed=opt.lensed)

    # train_y = train_set.y.numpy() # [120000, 7]
    # val_y = val_set.y.numpy() # [1500, 7]
    test_y = test_set.y.numpy() # [1500, 7]

    # y = np.vstack((train_y, val_y, test_y))
    y = test_y

    plotter.CornerPlot(y.T, opt) # [7, 15000]
    plotter.RandomScatter(y.T, opt)

################ MAIN ################
def main():
    opt = utils.Parser()

    if opt.danalysis:
        opt.problem = 'regression'
        data_plot(opt)
    else:
        if not opt.results_plot:
            opt.cuda = torch.cuda.is_available()
            torch.cuda.set_device(opt.gpuid)
            opt.device = torch.device('cuda:%s'%opt.gpuid if opt.cuda else 'cpu')
            if opt.device.type == 'cpu':
                raise NotImplementedError("Can't run without GPU")
            opt = prepare_loss(opt)
            model = prepare_model(opt)
            optim = prepare_optim(model, opt)
            if opt.train:
                if opt.resume:
                    # Resume training from the latest or user-specific checkpoint
                    model, optim = set_loaded_model(model, optim, opt)
                train(model, optim, opt)
            else:
                test(model, opt)
        else:
            plot_results(opt)
        

if __name__ == '__main__':
    main()
