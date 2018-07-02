import os
import json
import random

import numpy as np

import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchsample.transforms import tensor_transforms


class SegmentationDataSet(Dataset):
    def __init__(self, base_path, subset='train'):
        self.path = os.path.join(base_path, subset)
        if not os.path.exists(self.path):
            raise ValueError('Invalid base_path and subset combination - path does not exist')

        f = open(os.path.join(self.path, 'info.json'))
        dict = json.load(f)

        self._len = dict['len']
        self.elems = dict['elements']

    def __getitem__(self, item):
        input_path, target_path = self.elems[item]
        # TODO: relative paths instead
        input_img = Image.open(input_path)
        target = np.load(target_path).astype(dtype=np.float32)

        target = np.reshape(target, (112, 112, 1))

        return self.transform(input_img, target)


    def transform(self, image, map):
        # transform map to pil image
        map = TF.to_pil_image(map, mode='F')

        # Affine transformations
        angle, trans, scale, shear = transforms.RandomAffine.get_params((-15, 15), (0.3, 0.3), (0.8, 1.2), None, (112, 112))
        image = TF.affine(image, angle, trans, scale, shear)
        map = TF.affine(map, angle, trans, scale, shear)

        # Random flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            map = TF.hflip(map)

        image = TF.to_tensor(image)
        map = TF.to_tensor(map)
        #m2 = np.array(map)

        # Normalisation
        image = tensor_transforms.StdNormalize()(image)
        image = TF.normalize(image, [0, 0, 0], [1, 1, 1])

        return image, map



    def __len__(self):
        return self._len