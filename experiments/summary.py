import sys
sys.path.insert(0, '../')

from featurenet import network, training_old, dataset

import os
import torch
from torch import optim, nn

from torchsummary import summary


import cv2 as cv
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

path = 'E:\\Eyes\\UnityEyes_Windows\\'


def main():
    model = network.StandardFeatureNet()

    writer = SummaryWriter('C:\\Users\\Anton\\Desktop\\tboard\\')
    dummy = torch.rand(1, 3, 112, 112)
    dummy = dummy.requires_grad_()
    writer.add_graph(model, dummy, verbose=True)


if __name__ == '__main__':
    main()
