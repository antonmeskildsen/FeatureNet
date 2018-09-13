import sys
sys.path.insert(0, '../')

from featurenet import network, training, dataset

import torch
from torch import optim, nn

from torchsummary import summary

path = 'E:\\Eyes\\UnityEyes_Windows\\'


def main():
    dset = dataset.SyntheticDataSet(path, subset='imgs')
    model = network.StandardFeatureNet()
    model.cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    summary(model, (3, 112, 112))

    trainer = training.Trainer(model, criterion, optimizer)
    trainer.train(dset, None, 20, use_cuda=True)

    torch.save(model.state_dict(), 'models/test2.pkl')


if __name__ == '__main__':
    main()
