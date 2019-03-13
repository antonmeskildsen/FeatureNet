from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.handlers import ModelCheckpoint, EarlyStopping

from tqdm import tqdm_notebook
import os

from featurenet.visualisation import MetricPlot, Images, Html, properties
from featurenet.logging import Logger, ConfusionMatrix
from featurenet.helpers import time_code_id, gray_to_rgb

import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler


def flatten(output):
    y_pred, y = output
    return torch.sigmoid(y_pred), y


def train(model,
          criterion,
          optimizer,
          train_loader,
          val_loader,
          num_epochs,
          log_interval,
          output_dir,
          early_stopping_strikes=1,
          device='cuda',
          tqdm=tqdm_notebook):
    metrics = {
        'confusion_matrix': ConfusionMatrix(output_transform=flatten),
        'accuracy': Accuracy(output_transform=flatten),
        'precision': Precision(output_transform=flatten, average=True),
        'recall': Recall(output_transform=flatten, average=True),
        'nll': Loss(F.binary_cross_entropy_with_logits)
    }

    trainer = create_supervised_trainer(model, optimizer, criterion, device)
    evaluator = create_supervised_evaluator(model,
                                            metrics=metrics,
                                            device=device)

    time_code = time_code_id()

    checkpoint_handler = ModelCheckpoint(output_dir,
                                         time_code,
                                         save_interval=2,
                                         n_saved=2,
                                         create_dir=True,
                                         require_empty=False)

    #early_stopping_handler = EarlyStopping(early_stopping_strikes,
    #                                       lambda engine: -engine.state.metrics['nll'],
    #                                       trainer=trainer)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'model': model})
    #evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)

    iter_logger = Logger(['nll'], 'iter_logger')
    train_logger = Logger(['confusion_matrix', 'accuracy', 'precision', 'recall', 'nll'], 'training')
    val_logger = Logger(['confusion_matrix', 'accuracy', 'precision', 'recall', 'nll'], 'validation')
    loggers = [train_logger, val_logger]

    plot_env = 'plots'
    viz_env = 'viz'

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    loss_per_iteration = MetricPlot([iter_logger], 'nll', env_name=plot_env, units='Iterations')

    accuracy_plot = MetricPlot(loggers, 'accuracy', env_name=plot_env)

    precision_plot = MetricPlot(loggers, 'precision', env_name=plot_env)

    recall_plot = MetricPlot(loggers, 'recall', env_name=plot_env)

    imgwin = Images(num_cols=1+3*model.out_channels, env_name=viz_env)
    metrics_window = Html(env_name=plot_env)

    props = [
        {'type': 'button', 'name': 'Stop training', 'value': 'Stop training'}
    ]

    # TODO: Hacky
    @properties('plots', props)
    def handle_events(event):
        trainer.terminate()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)

            iter_logger.log_step({'nll': engine.state.output})
            loss_per_iteration.update()

    @trainer.on(Events.EPOCH_COMPLETED)
    def show_sample_outputs(engine):
        sampler = RandomSampler(val_loader.dataset, replacement=True, num_samples=10)

        img_list = []

        for i in sampler:
            inp, target = val_loader.dataset.__getitem__(i)
            img_list.append(inp * 255)
            out = model(inp.unsqueeze(0).cuda())
            #out = torch.nn.Softmax2d(out).cpu()[0]
            out = torch.sigmoid(out).cpu()[0]
            for i in range(model.out_channels):
                pred = gray_to_rgb(out[i].unsqueeze(0))
                img_list.append(pred * 255)
                img_list.append(torch.round(pred) * 255)
                img_list.append(gray_to_rgb(target[i].unsqueeze(0)) * 255)

        img_tensor = torch.stack(img_list)
        imgwin.update(img_tensor)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        pbar.set_postfix_str(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll))

        train_logger.log_step(metrics)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        pbar.set_postfix_str(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll))

        pbar.n = pbar.last_print_n = 0

        val_logger.log_step(metrics)
        metrics_window.update(val_logger.to_html())

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_plots(engine):
        accuracy_plot.update()
        precision_plot.update()
        recall_plot.update()

    @trainer.on(Events.COMPLETED)
    def trainer_finalize(engine):
        val_logger.save(os.path.join(output_dir, time_code + '_metrics_val.json'))
        train_logger.save(os.path.join(output_dir, time_code + '_metrics_train.json'))

    trainer.run(train_loader, num_epochs)
