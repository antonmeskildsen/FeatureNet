import cv2 as cv
import numpy as np

from featurenet import dataset

path = 'E:\\Documents\\Eyes\\UnityEyes_Windows\\'

dset = dataset.SyntheticDataSet(path, subset='imgs')

i=0
while True:
    inp, target = dset.__getitem__(i)

    i += 1

    inp = np.uint8(inp.numpy()[0]*255)
    target = np.uint8(target.numpy()[0]*255)

    img = np.hstack((inp, target))

    cv.imshow('hej', img)

    cv.waitKey(-1)
