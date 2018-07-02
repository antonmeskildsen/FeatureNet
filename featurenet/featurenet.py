
import torch.optim as optim
import torch.nn as nn

from featurenet import network
from featurenet import training
from featurenet import helpers

if __name__ == '__main__':
    model = network.StandardFeatureNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-5)

    trainer = training.Trainer(model, criterion, optimizer)

    # Datasets (invalid values are there just for playing with how the code
    # looks. Actual implementation will come later).
    train_data = None
    val_data = None

    trainer.train(train_data, val_data, 30, 32)