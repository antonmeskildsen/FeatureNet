import sys

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from progressbar import ProgressBar, Percentage, Bar, ETA, SimpleProgress, RotatingMarker

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data


class Trainer:

    def __init__(self, model, criterion, optimizer):
        self._bar = None
        self._ep_str = ''
        self._trainer = create_supervised_trainer(model, optimizer, criterion, device='cuda')
        self._evaluator = create_supervised_evaluator(model,
                                                      metrics={
                                                        'accuracy': Accuracy(),
                                                        'nll': Loss(F.nll_loss)
                                                        },
                                                      device='cuda')

    def train(self, train_data, val_data, num_epochs, batch_size=32,
              early_stopping=True, early_stopping_strikes=1, use_cuda=True):

        train_loader = data.DataLoader(train_data,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=8,
                                       drop_last=True)

        self._ep_str = 'Epoch:  0/{}'.format(num_epochs)

        @self._trainer.on(Events.EPOCH_STARTED)
        def update_epoch(trainer):
            self._ep_str = 'Epoch: {:2d}/{}'.format(trainer.state.epoch, num_epochs)
            print()
            self._bar = ProgressBar(widgets=[self._ep_str, Bar(), ETA()],
                                    max_value=len(train_loader),
                                    fd=sys.stdout).start()

        @self._trainer.on(Events.ITERATION_STARTED)
        def update_progress(trainer):
            self._bar.update(trainer.state.iteration % (len(train_loader)+1))

        self._trainer.run(train_loader, num_epochs)
