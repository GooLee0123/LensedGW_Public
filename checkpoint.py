import os
import time
import logging
import shutil

import torch

class Checkpoint():

    def __init__(self, step, epoch, model, optim, path=None, opt=None):
        self.step = step
        self.epoch = epoch
        self.model = model
        self.optim = optim

        self._path = path
        self.opt = opt

        self.logger = logging.getLogger(__name__)

    @property
    def path(self):
        if self._path is None:
            raise LookupError("The checkpoint has not been saved.")
        return self._path

    @classmethod
    def load(cls, model, optim=None, opt=None):
        logger = logging.getLogger(__name__)
        cfname = 'checkpoint'+opt.token
        cpath = os.path.join(opt.test_epath, opt.test_cpath, cfname)
        if 'MICRO' in cpath:
            cpath = cpath.replace('MICRO', opt.tl_model)

        if opt.cdate == 'latest': # get latest checkpoint
            all_times = sorted(os.listdir(cpath), reverse=True)
            fchckpt = os.path.join(cpath, all_times[0])
        else: # get user-specific checkpoint
            fchckpt = os.path.join(cpath, opt.cdate)
        logger.info("load checkpoint from %s" % fchckpt)

        resume_model = torch.load(os.path.join(fchckpt, opt.model_state),
                                map_location=opt.device)
        resume_checkpoint = torch.load(os.path.join(fchckpt, opt.trainer_state), 
                                map_location=opt.device)

        model.load_state_dict(resume_model)
        if optim != None:
            optim.load_state_dict(resume_checkpoint['optimizer'])

        return Checkpoint(step=resume_checkpoint['step'],
                          epoch=resume_checkpoint['epoch'],
                          model=model,
                          optim=optim,
                          path=cpath)

    def save(self):
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        cfname = 'checkpoint'+self.opt.token
        directory = os.path.join(self.opt.test_epath, self.opt.test_cpath, cfname)
        if os.path.exists(directory):
            ockpts = [os.path.join(directory, ockpt) for ockpt in os.listdir(directory)]
            for ockpt in ockpts:
                self.logger.info("Remove old checkpoint %s" % ockpt)
                os.system('rm -r %s'%ockpt)
        else:
            os.makedirs(directory)
        self._path = os.path.join(directory, date_time)
        path = self._path

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        torch.save({'epoch': self.epoch,
                    'step': self.step,
                    'optimizer': self.optim.state_dict()
                    },
                    os.path.join(path, self.opt.trainer_state))
        torch.save(self.model.state_dict(), os.path.join(path, self.opt.model_state))

        self.logger.info("Validation loss being smaller than previous minimum, checkpoint is saved at %s" % path)
        return path