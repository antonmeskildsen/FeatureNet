import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchnet.meter import ConfusionMeter

from featurenet import helpers

class Trainer:

    NUM_WORKERS = 8

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 train_data: Dataset,
                 val_data: Dataset,
                 num_epochs: int,
                 batch_size: int = 32,
                 early_stopping: bool = True,
                 early_stopping_tries: int = 1,
                 shuffle: bool = True,
                 ):

        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._early_stopping = early_stopping
        self._early_stopping_tries = early_stopping_tries
        self._shuffle = shuffle

        self._train_loader = DataLoader(train_data,
                                        batch_size,
                                        shuffle,
                                        num_workers=self.NUM_WORKERS,
                                        drop_last=True,
                                        )

        self._val_loader = DataLoader(val_data,
                                      batch_size,
                                      shuffle,
                                      num_workers=self.NUM_WORKERS,
                                      drop_last=True,
                                      )


    def train(self):
        for epoch in range(self._num_epochs):
            for i, (input, target) in enumerate(self._train_loader):
                input = input.cuda()
                target = target.cuda()

                output = self._model(input)

                loss = self._criterion(output, target)
                loss.backward()

                self._optimizer.zero_grad()
                self._optimizer.step()


class DataLogger:

    def __init__(self, num_classes):
        """
        :param num_classes: An integer representing the number of
            classes. If num_classes=2, then the data is expected to
            have a binary encoding whereas for cases >2, the data is
            expected to be one-hot encoded.
        """
        self._num_classes = num_classes
        self._meter = ConfusionMeter(num_classes)
        self._summary_log = self._get_empty_summary_dict()

    @staticmethod
    def _get_empty_summary_dict():
        return {
            'TP': [],
            'TN': [],
            'FP': [],
            'FN': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
        }

    def log_statistics(self, output, target):
        flat_output_rounded = output.view((-1, self._num_classes)).round()
        flat_target = target.view((-1, self._num_classes))

        output_data = flat_output_rounded.data
        target_data = flat_target.data

        self._meter.add(output_data, target_data)

    def log_epoch_summary(self):
        conf = self._meter.value()
        summary = self._calculate_summary(conf)

        self._add_summary_to_log(summary)

    @staticmethod
    def _calculate_summary(conf):
        accuracy = helpers.accuracy(conf)
        precision = helpers.precision(conf)
        recall = helpers.recall(conf)
        f1 = helpers.f1_score(precision, recall)

        return {
            'TP': conf[0][0],
            'TN': conf[1][1],
            'FP': conf[0][1],
            'FN': conf[1][0],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def _add_summary_to_log(self, summary):
        self._summary_log = \
            {val.append(summary[key]) for key, val in self._summary_log}

    def new_epoch_reset(self):
        self._meter.reset()

    def save(self, path):
        dataframe = pd.DataFrame(self._summary_log)
        dataframe.to_csv(path)


class TrainerA:
    def __init__(self, network, cuda=False):
        self.network = network
        if cuda:
            self.network.cuda()

    def evaluate(self, val_set, batch_size, input_size, val_rounds=None, threshold=0.5):
        if val_rounds is None:
            val_rounds = len(val_set)//batch_size

        val_loader = DataLoader(val_set,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=8,
                                drop_last=True)

        # Test model
        self.model.eval()
        eval_meter = ConfusionMeter(2, True)
        for i, (images, np_labels) in enumerate(itertools.islice(val_loader, val_rounds)):
            images = Variable(images).cuda()
            labels = Variable(np_labels).cuda()

            outputs = self.model(images)
            outputs = nn.Sigmoid()(outputs)
            ones = torch.ones(outputs.size()).cuda()
            zeros = torch.zeros(outputs.size()).cuda()
            outputs = torch.where(outputs > threshold, ones, zeros).round()

            eval_meter.add(outputs.view(batch_size*input_size*input_size).data, labels.view(batch_size*input_size*input_size).data)

            #if i % 10 == 0:
            #    print('{}%'.format(i / (len(val_set)/batch_size) * 100))

        #print('Validation accuracy: %f %%' % (100 * correct / total))
        self.model.train()
        return eval_meter.value()

    def training(self, path, num_epochs, batch_size, learning_rate, model_name):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        train_dset = SegmentationDataSet(path, 'train')
        val_set = SegmentationDataSet(path, 'val')

        train_loader = DataLoader(train_dset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True)

        val_acc_cur = 0
        val_acc_prev = 0
        num_bad = 0
        # Metrics
        train_metrics = {
            'TP': [],
            'TN': [],
            'FP': [],
            'FN': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        val_metrics = {
            'TP': [],
            'TN': [],
            'FP': [],
            'FN': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        train_meter = ConfusionMeter(2, True)

        # Train model
        for epoch in range(num_epochs):
            correct_t = 0
            train_conf = np.zeros((2, 2))
            for i, (images, np_target) in enumerate(train_loader):
                images = Variable(images).cuda()
                target = Variable(np_target).cuda()

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                outputs = nn.Sigmoid()(outputs)

                SIZE = 112
                # train_conf += metrics.confusion_matrix(np_targ, np_pred_labels)
                train_meter.add(outputs.view(batch_size * SIZE * SIZE).round().data, target.view(batch_size * SIZE * SIZE).data)
                if (i + 1) % 30 == 0:

                    # Compute accuracy
                    flat_out = outputs.view(SIZE * SIZE * batch_size)
                    flat_out = flat_out.round()
                    np_pred_labels = flat_out.data.cpu().numpy()
                    np_targ = np.reshape(np_target, (batch_size*SIZE*SIZE))

                    conf = train_meter.value()

                    acc = accuracy(conf) #metrics.accuracy_score(np_targ, np_pred_labels)
                    prec = precision(conf) #metrics.precision_score(np_targ, np_pred_labels)
                    rec = recall(conf)  #metrics.recall_score(np_targ, np_pred_labels)
                    f1 = f1_score(prec, rec)  #metrics.f1_score(np_targ, np_pred_labels)

                    arr = outputs.data.cpu().numpy()[0]
                    arr = arr[0]
                    arr[arr < 0.5] = 0
                    print(arr.max())

                    if (i+1) % 120 == 0:
                        plt.imshow(arr)
                        plt.show()
                        plt.imshow(target.data.cpu().numpy()[0][0])
                        plt.show()

                    print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, '
                          % (epoch + 1, num_epochs, i + 1, len(train_dset) // batch_size, loss.data[0], acc, prec, rec, f1))

                    correct = 0

            if val_acc_cur < val_acc_prev:
                num_bad += 1
            else:
                num_bad = 0

            if num_bad > 3:
                break
            elif num_bad == 0:
                torch.save(self.model.state_dict(), 'saved_models/{}.pkl'.format(model_name))

            val_acc_prev = val_acc_cur
            eval_conf = self.evaluate(val_set, batch_size, SIZE, 1000)
            val_acc_cur = (eval_conf[0,0]+eval_conf[1,1])/(batch_size*SIZE*SIZE)
            print(val_acc_cur)

            train_conf = train_meter.value()
            save_stats(train_meter.value(), train_metrics)
            save_stats(eval_conf, val_metrics)

            train_meter.reset()

        train_ps = pd.DataFrame(train_metrics)
        val_ps = pd.DataFrame(val_metrics)
        train_ps.to_csv('saved_models/metrics/{}_train.csv'.format(model_name))
        val_ps.to_csv('saved_models/metrics/{}_val.csv'.format(model_name))

        #torch.save(self.model.state_dict(), 'saved_models/mk4.pkl')

    def step(self):
        ...

