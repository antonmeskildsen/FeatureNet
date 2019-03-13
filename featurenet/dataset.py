import os
import json
import glob

import numpy as np
import cv2 as cv
import torch

import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SyntheticDataSet(Dataset):
    def __init__(self,
                 base_path, subset='train',
                 input_size=(640, 480),
                 input_crop=(480, 480),
                 input_resize=(112, 112),
                 target_size=(112, 112)):
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
        self._input_resize = input_resize

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

    def _create_interior_margin_img(self, coordinates):
        coordinates = [eval(s) for s in coordinates]
        coordinates = [(float(x), float(y)) for (x, y, _) in coordinates]
        coordinates = self._adjust_coordinates(coordinates)

        coordinates = np.asarray(coordinates, dtype=np.int)
        coordinates = coordinates.reshape((-1, 1, 2))

        target = np.zeros(self._target_size, dtype=np.uint8)
        cv.fillPoly(target, [coordinates], 255)

        return cv.flip(target, 0)

    def _create_iris_img(self, coordinates):
        coordinates = [eval(s) for s in coordinates]
        coordinates = [(float(x), float(y)) for (x, y, _) in coordinates]
        coordinates = self._adjust_coordinates(coordinates)

        coordinates = np.asarray(coordinates, dtype=np.int)
        coordinates = coordinates.reshape((-1, 1, 2))

        target = np.zeros(self._target_size, dtype=np.uint8)
        (cx, cy), (ax, ay), a = cv.fitEllipse(coordinates)
        target = cv.ellipse(target, (int(cx), int(cy)), (int(ax)//2, int(ay)//2), a, 0, 360, 255, thickness=-1)

        return cv.flip(target, 0)

    def __getitem__(self, item):
        input_path, target_path = self._elems[item]

        input_img = Image.open(input_path)

        target_file = open(target_path)
        target_json = json.load(target_file)
        margin = self._create_interior_margin_img(target_json['interior_margin_2d'])
        iris = self._create_iris_img(target_json['iris_2d'])
        marg_sub = cv.subtract(margin, iris)
        target_arr = np.dstack((iris, margin))

        return self._transform_input(input_img), self._transform_target(target_arr)

    def _transform_input(self, input_img):
        return transforms.Compose([
            transforms.CenterCrop(self._input_crop),
            transforms.Resize(self._input_resize),
            transforms.ToTensor()
        ])(input_img)

    def _transform_target(self, target_arr):
        # NOTE: Change channel number here
        target_arr = np.reshape(target_arr, (*self._target_size, 2))
        return TF.to_tensor(target_arr)

    def __len__(self):
        return self._len
