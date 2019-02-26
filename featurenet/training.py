from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.handlers import ModelCheckpoint, EarlyStopping

from tqdm import tqdm

from featurenet import visualisation
from featurenet import helpers

import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler

import matplotlib.pyplot as plt

from datetime import datetime
import visdom


class Visualizations:
    def __init__(self, env_name=None):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)
        self.loss_win = None

    def plot_loss(self, loss, step):
        self.loss_win = self.vis.line(
            [loss],
            [step],
            win=self.loss_win,
            update='append' if self.loss_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='Loss',
                title='Loss (mean per 10 steps)',
            )
        )


def flatten(output):
    y_pred, y = output
    y = torch.round(y)
    y_pred = torch.round(torch.sigmoid(y_pred))
    return y_pred.view(-1), y.view(-1)


def train(model,
          criterion,
          optimizer,
          train_loader,
          val_loader,
          num_epochs,
          log_interval,
          output_dir,
          prefix,
          early_stopping_strikes=1,
          device='cuda',
          metrics=None):
    if metrics is None:
        metrics = {
            'accuracy': Accuracy(output_transform=flatten),
            'precision': Precision(output_transform=flatten, average=True),
            'recall': Recall(output_transform=flatten, average=True),
            'nll': Loss(F.binary_cross_entropy_with_logits)
        }

    trainer = create_supervised_trainer(model, optimizer, criterion, device)
    evaluator = create_supervised_evaluator(model,
                                            metrics=metrics,
                                            device=device)

    checkpoint_handler = ModelCheckpoint(output_dir,
                                         prefix,
                                         save_interval=2,
                                         n_saved=2,
                                         create_dir=True,
                                         require_empty=False)

    early_stopping_handler = EarlyStopping(early_stopping_strikes,
                                           lambda engine: -engine.state.metrics['nll'],
                                           trainer=trainer)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'model': model})
    evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    loss_per_iteration = visualisation.Plot(title='Loss (per iteration)',
                                            y_label='Loss',
                                            x_label='Iteration',
                                            env_name='test')

    accuary_plot = visualisation.Plot(title='Accuracy',
                                      y_label='',
                                      x_label='Epoch',
                                      env_name='test')

    precision_plot = visualisation.Plot(title='Precision',
                                        y_label='',
                                        x_label='Epoch',
                                        env_name='test')

    recall_plot = visualisation.Plot(title='Recall',
                                     y_label='',
                                     x_label='Epoch',
                                     env_name='test')

    imgwin = visualisation.Images(num_cols=4, env_name='test')

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)

            loss_per_iteration.update(engine.state.output, engine.state.iteration)


    @trainer.on(Events.EPOCH_COMPLETED)
    def show_sample_outputs(engine):
        sampler = RandomSampler(val_loader.dataset, replacement=True, num_samples=10)

        img_list = []

        for i in sampler:
            inp, target = val_loader.dataset.__getitem__(i)
            out = model(inp.unsqueeze(0).cuda())
            out = torch.sigmoid(out)
            img_list.append(inp * 255)
            pred = helpers.gray_to_rgb(out.cpu()[0])
            img_list.append(pred*255)
            img_list.append(torch.round(pred)*255)
            img_list.append(helpers.gray_to_rgb(target) * 255)

        img_tensor = torch.stack(img_list)
        imgwin.update(img_tensor)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                .format(engine.state.epoch, avg_accuracy, avg_nll)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                .format(engine.state.epoch, avg_accuracy, avg_nll))

        pbar.n = pbar.last_print_n = 0

        accuary_plot.update(metrics['accuracy'], engine.state.epoch)
        precision_plot.update(metrics['precision'], engine.state.epoch)
        recall_plot.update(metrics['recall'], engine.state.epoch)

    trainer.run(train_loader, num_epochs)
