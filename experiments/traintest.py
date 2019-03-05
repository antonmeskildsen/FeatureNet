import os

from featurenet import network, training, dataset

import torch
from torch import optim, nn
from torch.utils import data

from torchsummary import summary

path = 'E:\\Documents\\Eyes\\UnityEyes_Windows\\'


def main():
    dset = dataset.SyntheticDataSet(path,
                                    subset='train_big',
                                    input_crop=(320, 320))
    valset = dataset.SyntheticDataSet(path,
                                      subset='val_big',
                                      input_crop=(320, 320))
    model = network.StandardFeatureNet(use_conv_transpose=False)
    model.cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    summary(model, (3, 112, 112))

    train_loader = data.DataLoader(dset,
                                   batch_size=32,
                                   shuffle=True,
                                   num_workers=8,
                                   drop_last=True)

    val_loader = data.DataLoader(valset,
                                 batch_size=32,
                                 shuffle=True,
                                 num_workers=8,
                                 drop_last=True)

    training.train(model,
                   criterion,
                   optimizer,
                   train_loader,
                   val_loader,
                   num_epochs=20,
                   log_interval=1,
                   output_dir=os.path.join(path, 'models_tmp'))


if __name__ == '__main__':
    main()
