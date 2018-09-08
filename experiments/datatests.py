import sys
sys.path.insert(0, '../')

from featurenet import network, training, dataset

import os
import torch
from torch import optim, nn

model = network.StandardFeatureNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

trainer = training.Trainer(model, criterion, optimizer)

import cv2 as cv
import matplotlib.pyplot as plt

path = 'E:\\Eyes\\UnityEyes_Windows\\'

dset = dataset.SyntheticDataSet(path, subset='train')

dset.__getitem__(0)