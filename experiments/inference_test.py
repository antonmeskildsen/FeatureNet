import sys
sys.path.insert(0, '../')

from featurenet import network, training, dataset

import os
import torch
import time
from torch import optim, nn

from torchsummary import summary


import cv2 as cv
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchsample.transforms import tensor_transforms
import torchvision.transforms.functional as TF

path = 'E:\\Eyes\\UnityEyes_Windows\\'


def main():
    model = network.StandardFeatureNet()
    model.load_state_dict(torch.load('models/test2.pkl'))
    model.cuda()

    input_img = Image.open(os.path.join(path, 'train\\1.jpg'))
    input_img = TF.center_crop(input_img, (224, 224))
    input_img = TF.resize(input_img, (112, 112))

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(input_img)

    input_img = TF.to_tensor(input_img)
    input_img = input_img.unsqueeze(0)

    s = time.time()
    out = model(input_img.cuda())
    print(time.time()-s)
    out = out.cpu().detach().numpy()
    out = out[0][0]

    plt.subplot(2, 2, 2)
    plt.imshow(out)

    out[out < 0.5] = 0

    plt.subplot(2, 2, 3)
    plt.imshow(out)


    plt.show()


if __name__ == '__main__':
    main()
