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
plt.interactive(False)

path = 'E:\\Eyes\\UnityEyes_Windows\\'

dset = dataset.SyntheticDataSet(path, subset='imgs', input_crop=(480, 480))


input, target = dset.__getitem__(6070)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(input)
plt.subplot(1, 2, 2)
plt.imshow(target)
plt.show()