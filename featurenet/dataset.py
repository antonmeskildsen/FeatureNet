import os
import json
import random
import glob

import numpy as np
import cv2 as cv

import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchsample.transforms import tensor_transforms


class SyntheticDataSet(Dataset):
    def __init__(self, base_path, subset='train', input_size=(640, 480), input_crop=(224, 224), target_size=(112, 112)):
        path = os.path.join(base_path, subset)
        if not os.path.exists(path):
            raise ValueError('Invalid base_path and subset combination - path does not exist')

        inputs_path = os.path.join(path, '*.jpg')
        targets_path = os.path.join(path, '*.json')

        inputs = glob.glob(inputs_path)
        targets = glob.glob(targets_path)

        self._elems = list(zip(inputs, targets))
        self._len = len(inputs)

        self._input_size = input_size
        self._input_crop = input_crop
        self._target_size = target_size

    def _adjust_coordinates(self, coordinates):
        width, height = self._input_size
        crop_x, crop_y = self._input_crop

        sub_x = (width - crop_x) / 2
        sub_y = (height - crop_y) / 2

        target_width, target_height = self._target_size

        scale_x = target_width / crop_x
        scale_y = target_height / crop_y

        for i in range(len(coordinates)):
            x, y = coordinates[i]
            x = (x - sub_x) * scale_x
            y = (y - sub_y) * scale_y

            coordinates[i] = x, y

        return coordinates

    def _create_target_img(self, coordinates):
        coordinates = [eval(s) for s in coordinates]
        coordinates = [(float(x), float(y)) for (x, y, _) in coordinates]
        coordinates = self._adjust_coordinates(coordinates)

        coordinates = np.asarray(coordinates, dtype='int')
        coordinates = coordinates.reshape((-1, 1, 2))

        target = np.zeros((1, *self._target_size), dtype=np.float32)
        print(coordinates)
        ellipse = cv.fitEllipse(coordinates)
        print('hej')
        return cv.ellipse(target, ellipse, 255, 0)

    def __getitem__(self, item):
        input_path, target_path = self._elems[item]

        input_img = Image.open(input_path)

        print(target_path)

        target_file = open(target_path)
        target_json = json.load(target_file)
        target_img = self._create_target_img(target_json['iris_2d'])

        return self._transform_input(input_img), self._transform_target(target_img)

    def _transform_input(self, input_img):
        return input_img

    def _transform_target(self, target_img):
        return target_img

    def __len__(self):
        return self._len


# class SegmentationDataSet(Dataset):
#     def __init__(self, base_path, subset='train'):
#         self.path = os.path.join(base_path, subset)
#         if not os.path.exists(self.path):
#             raise ValueError('Invalid base_path and subset combination - path does not exist')
#
#         f = open(os.path.join(self.path, 'info.json'))
#         dict = json.load(f)
#
#         self._len = dict['len']
#         self.elems = dict['elements']
#
#     def __getitem__(self, item):
#         input_path, target_path = self.elems[item]
#         # TODO: relative paths instead
#         input_img = Image.open(input_path)
#         target = np.load(target_path).astype(dtype=np.float32)
#
#         target = np.reshape(target, (112, 112, 1))
#
#         return self.transform(input_img, target)
#
#
#     def transform(self, image, map):
#         # transform map to pil image
#         map = TF.to_pil_image(map, mode='F')
#
#         # Affine transformations
#         angle, trans, scale, shear = transforms.RandomAffine.get_params((-15, 15), (0.3, 0.3), (0.8, 1.2), None, (112, 112))
#         image = TF.affine(image, angle, trans, scale, shear)
#         map = TF.affine(map, angle, trans, scale, shear)
#
#         # Random flipping
#         if random.random() > 0.5:
#             image = TF.hflip(image)
#             map = TF.hflip(map)
#
#         image = TF.to_tensor(image)
#         map = TF.to_tensor(map)
#         #m2 = np.array(map)
#
#         # Normalisation
#         image = tensor_transforms.StdNormalize()(image)
#         image = TF.normalize(image, [0, 0, 0], [1, 1, 1])
#
#         return image, map
#
#
#
#     def __len__(self):
#         return self._len