import unittest

import torch
import torch.nn.functional as F

from featurenet import training_old


class DataLoggerTests(unittest.TestCase):

    @staticmethod
    def _random_tensor(size, rounded=False, softmax=True):
        tensor = torch.randn(size, dtype=torch.float32)
        if rounded:
            tensor = tensor.round()
        if softmax:
            tensor = F.softmax(tensor)

        return tensor

    def test_log_statistics_valid(self):
        arguments = [
            {
                'num_classes': 2,
                'input': self._random_tensor(100),
                'output': self._random_tensor(100, rounded=True)
            },
            {
                'num_classes': 2,
                'input': torch.rand((1, 1, 100, 100), dtype=torch.float32),
                'output': torch.rand((1, 1, 100, 100), dtype=torch.float32).round()
            },
            {
                'num_classes': 100,
                'input': torch.rand((1, 100, 50, 50), dtype=torch.float32),
                'output': torch.rand((1, 100, 50, 50), dtype=torch.float32).round()
            }
        ]

        for arg_instance in arguments:
            logger = training_old.DataLogger(arg_instance['num_classes'])
            logger.log_statistics(arg_instance['input'],
                                  arg_instance['output'])
