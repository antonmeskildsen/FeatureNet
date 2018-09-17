import featurenet.network as network

import unittest

import torch
import torch.nn as nn
import torch.optim as optim


class ConvBlockTests(unittest.TestCase):

    invalid_setup_args = [
        (1, 1, 3, 0, 0),
        (0, 1, 3, 1, 0),
        (1, 0, 3, 1, 0),
        (1, 1, 0, 1, 0),
        (1, 1, 3, -1, 0),
        (-1, 1, 3, 1, 0),
        (1, -1, 3, 1, 0),
        (1, 1, -1, 1, 0)
    ]

    def test_invalid_setup(self):
        for args in self.invalid_setup_args:
            self.assertRaises(ValueError, network.ConvBlock, *args)

    @staticmethod
    def _iter_length(it):
        return sum(1 for _ in it)

    def test_correct_block_number(self):
        for num in [1, 5, 10]:
            block = network.ConvBlock(1, 1, 3, num)
            self.assertEqual(num, self._iter_length(block.convs.children()))

    def test_working_forward(self):
        # Setup module
        block = network.ConvBlock(1, 3, 3, 3, 0.3)

        # Setup training
        optimizer = optim.SGD(block.parameters(), lr=1e10)
        criterion = nn.BCEWithLogitsLoss()

        # Setup fake training data
        input = torch.randn((1, 1, 100, 100), dtype=torch.float32)
        target = torch.randn((1, 3, 50, 50))

        # Clone parameters before update
        before = [param.detach().clone() for param in block.parameters()]

        # Perform optimisation step
        output, _ = block(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Clone parameters after update
        after = [param.detach().clone() for param in block.parameters()]

        # Test that all weight tensors have changed
        for b, a in zip(before, after):
            c = (a != b)
            if c.sum() == 0:
                self.fail('Weights were not updated correctly for all layers!'
                          '\n {}'.format(a))


class DeconvBlockTests(unittest.TestCase):
    invalid_setup_args = [
        (1, 1, 3, 0),
        (0, 1, 3, 1),
        (1, 0, 3, 1),
        (1, 1, 0, 1),
        (1, 1, 3, -1),
        (-1, 1, 3, 1),
        (1, -1, 3, 1),
        (1, 1, -1, 1)
    ]

    def test_invalid_setup(self):
        for args in self.invalid_setup_args:
            self.assertRaises(ValueError, network.DeconvBlock, *args)

    @staticmethod
    def _iter_length(it):
        return sum(1 for _ in it)

    def test_correct_block_number(self):
        for num in [1, 5, 10]:
            block = network.DeconvBlock(1, 1, 3, num)
            self.assertEqual(num, self._iter_length(block.convs.children()))

    def test_working_forward(self):
        # Setup module
        block = network.DeconvBlock(3, 1, 3, 3)

        # Setup training
        optimizer = optim.SGD(block.parameters(), lr=1e10)
        criterion = nn.BCEWithLogitsLoss()

        # Setup fake training data
        input = torch.randn((1, 3, 50, 50), dtype=torch.float32)
        indices = torch.randint(0, 3, (1, 3, 50, 50), dtype=torch.long)
        target = torch.randn((1, 1, 100, 100))

        # Clone parameters before update
        before = [param.detach().clone() for param in block.parameters()]

        # Perform optimisation step
        output = block(input, indices)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Clone parameters after update
        after = [param.detach().clone() for param in block.parameters()]

        # Test that all weight tensors have changed
        for b, a in zip(before, after):
            c = (a != b)
            if c.sum() == 0:
                self.fail('Weights were not updated correctly for all layers!'
                          '\n {}'.format(a))


class DeconvNetTests(unittest.TestCase):

    def test_invalid_setup_args(self):
        conv = {
            'in_channels': 3,
            'out_channels': 8,
            'kernel_size': 3,
        }
        deconv = {
            'in_channels': 3,
            'out_channels': 3,
            'kernel_size': 3
        }

        configurations = [
            ([conv], [deconv], 0, 1),
            ([conv], [deconv], 1, 0),
            ([conv], [deconv], -1, 1),
            ([conv], [deconv], 1, -1)
        ]

        for args in configurations:
            self.assertRaises(ValueError, network.DeconvNet, *args)

    def test_invalid_conv_deconv_block_configuration(self):
        conv_valid = {
            'in_channels': 3,
            'out_channels': 8,
            'kernel_size': 3,
        }
        deconv_valid = {
            'in_channels': 3,
            'out_channels': 3,
            'kernel_size': 3
        }

        conv_invalid = {
            'in_channels': 0,
            'out_channels': 0,
            'kernel_size': 0,
        }
        deconv_invalid = {
            'in_channels': 0,
            'out_channels': 0,
            'kernel_size': 0
        }

        configurations = [
            ([conv_valid], [deconv_invalid], 1, 1),
            ([conv_invalid], [deconv_valid], 1, 1),
        ]

        for args in configurations:
            self.assertRaises(ValueError, network.DeconvNet, *args)

    def test_channel_correspondence_valid(self):
        conv = [
            {
                'in_channels': 3,
                'out_channels': 8,
                'kernel_size': 3,
            },
            {
                'in_channels': 8,
                'out_channels': 16,
                'kernel_size': 3,
            },
        ]
        deconv = [
            {
                'in_channels': 16,
                'out_channels': 8,
                'kernel_size': 3,
            },
            {
                'in_channels': 8,
                'out_channels': 1,
                'kernel_size': 3,
            },
        ]

        try:
            network.DeconvNet(conv, deconv, 16, 3)
        except ValueError:
            self.fail("Shouldn't raise ValueError for valid configuration")

    def test_channel_correspondence_invalid(self):
        conv = [
            {
                'in_channels': 3,
                'out_channels': 8,
                'kernel_size': 3,
            },
            {
                'in_channels': 12,
                'out_channels': 16,
                'kernel_size': 3,
            },
        ]
        deconv = [
            {
                'in_channels': 16,
                'out_channels': 12,
                'kernel_size': 3,
            },
            {
                'in_channels': 8,
                'out_channels': 1,
                'kernel_size': 3,
            },
        ]

        self.assertRaises(ValueError, network.DeconvNet,
                          *(conv, deconv, 16, 3))

    def test_correct_output_size(self):
        conv = {
            'in_channels': 3,
            'out_channels': 8,
            'kernel_size': 3,
        }

        deconv = {
            'in_channels': 8,
            'out_channels': 3,
            'kernel_size': 3
        }

        net = network.DeconvNet([conv], [deconv], 8, 3)

    def test_working_forward(self):
        # Configuration
        conv_config = [
            {
                'in_channels': 3,
                'out_channels': 8,
                'kernel_size': 3
            },
            {
                'in_channels': 8,
                'out_channels': 32,
                'kernel_size': 3
            }
        ]
        deconv_config = [
            {
                'in_channels': 32,
                'out_channels': 8,
                'kernel_size': 3
            },
            {
                'in_channels': 8,
                'out_channels': 1,
                'kernel_size': 3
            }
        ]

        # Setup module
        net = network.DeconvNet(conv_config, deconv_config, 64, 32)

        # Setup training
        optimizer = optim.SGD(net.parameters(), lr=1e10)
        criterion = nn.BCEWithLogitsLoss()

        # Setup fake training data
        input = torch.randn((1, 3, 128, 128), dtype=torch.float32)
        target = torch.randn((1, 1, 128, 128))

        # Clone parameters before update
        before = [param.detach().clone() for param in net.parameters()]

        # Perform optimisation step
        output = net(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Clone parameters after update
        after = [param.detach().clone() for param in net.parameters()]

        # Test that all weight tensors have changed
        for b, a in zip(before, after):
            c = (a != b)
            if c.sum() == 0:
                self.fail('Weights were not updated correctly for all layers!'
                          '\n {}'.format(a))


class StandardFeatureNetTests(unittest.TestCase):

    def test_correct_setup(self):
        net = network.StandardFeatureNet()

        self.assertEqual(3, len(net.conv_blocks))
        self.assertEqual(3, len(net.deconv_blocks))

    def test_valid_input_and_output(self):
        net = network.StandardFeatureNet()
        input = torch.randn((1, 3, 112, 112), dtype=torch.float32)

        output = net(input)

        self.assertEqual((1, 1, 112, 112), output.size())