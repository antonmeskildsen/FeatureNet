from torch import nn


def _validate_args(in_channels, out_channels, kernel_size, num_blocks):
    if in_channels <= 0:
        raise ValueError('in_channels must be positive!')
    if out_channels <= 0:
        raise ValueError('out_channels must be positive!')
    if kernel_size <= 0:
        raise ValueError('kernel_size must be positive!')
    if num_blocks <= 0:
        raise ValueError('num_blocks must be positive!')


class ConvBlock(nn.Module):
    """
    A sequence of layers composed of a number of convolution layers
    each combined with a batch normalisation layer and a ReLu
    activation layer. At the end is a max-pooling layer.

    The forward method outputs the max-pooling indices for use with
    max-unpooling layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_layers=3,
                 dropout_rate=0.3):
        """
        Create a convolution block consisting of a number of layers.

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of all kernels
        :param num_layers: Number of convolution/normalisation/relu
                blocks to generate
        :param dropout_rate: Rate for a final dropout layer
        """
        super().__init__()

        _validate_args(in_channels, out_channels, kernel_size, num_layers)

        conv_list = [
            self._create_single_block(in_channels, out_channels, kernel_size)
        ]

        # Create the rest of the blocks
        for _ in range(num_layers - 1):
            conv_list.append(
                self._create_single_block(out_channels, out_channels,
                                          kernel_size)
            )

        self.convs = nn.Sequential(*conv_list)
        self.drop = nn.Dropout2d(dropout_rate)
        self.pool = nn.MaxPool2d((2, 2), return_indices=True)

    def forward(self, input):
        x = self.convs(input)
        x, indices = self.pool(x)
        x = self.drop(x)
        return x, indices

    @staticmethod
    def _create_single_block(in_channels, out_channels,
                             kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1,
                      stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class DeconvBlock(nn.Module):
    """
    A sequence of layers composed of a number of deconvolution
    (a.k.a transposed convolution) layers, each with a following
    batch normalisation layer (2d). A final max-unpooling layer
    reconstructs previously destroyed high-frequency information
    by using saved indices to upsample the feature maps.

    These indices are provided alongside the input feature maps
    to the forward method.
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_layers=3):
        """
        Construct a new deconvolution block.

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of all convolution kernels
                (possibly change this)
        :param num_layers: Number of convolution/normalisation
                blocks to create
        """
        super().__init__()

        _validate_args(in_channels, out_channels, kernel_size, num_layers)

        # All deconv blocks must contain at least one block mapping the
        # number of input channels to the number of output channels
        conv_list = [
            self._create_single_block(in_channels, out_channels, kernel_size)
        ]

        # Create the rest of the blocks
        for _ in range(num_layers - 1):
            conv_list.append(
                self._create_single_block(out_channels, out_channels,
                                          kernel_size)
            )

        self.convs = nn.Sequential(*conv_list)
        self.pool = nn.MaxUnpool2d((2, 2))

    def forward(self, input, indices):
        x = self.pool(input, indices)
        x = self.convs(x)
        return x

    @staticmethod
    def _create_single_block(in_channels, out_channels,
                             kernel_size):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size, padding=1),
            nn.BatchNorm2d(out_channels)
        )


class DeconvNet(nn.Module):
    """
    A fully convolutional network architecture created for the purpose
    of semantic segmentation of images. It is inspired by ...

    The basic construction is similar to a convolutional auto-encoder,
    with an encoder, and a decoder network.

    Future versions might support pre-training the encoder separately
    from the decoder.
    """
    def __init__(self, conv_block_args, deconv_block_args, flat_channels,
                 flat_kernel_size):
        """
        Create a new DeconvNet with the desired input size, depth,
        number of internal channels, as well as the number of input and
        output channels.

        :param conv_block_args: Arguments for each ConvBlock (as a
                dictionary)
        :param deconv_block_args: Arguments for each DeconvBlock (as a
            dictionary)
        :param flat_channels: Number of channels for the dense layers
        :param flat_kernel_size: Size of the kernel used to create the
            "dense" convolutional layers
        """
        super().__init__()

        # Perform a number of steps validating the input arguments
        self._validate_parameters(conv_block_args, deconv_block_args,
                                  flat_channels, flat_kernel_size)

        # Create lists of conv and deconv blocks from the configurations
        # passed as arguments to this function
        self.conv_blocks = nn.ModuleList([
            ConvBlock(**args)
            for args in conv_block_args
        ])

        self.deconv_blocks = nn.ModuleList([
            DeconvBlock(**args)
            for args in deconv_block_args
        ])

        # The input and output from the flat channels must be compatible
        # with the configurations for the conv and deconv blocks
        flat_in_channels = conv_block_args[-1]['out_channels']
        flat_out_channels = deconv_block_args[0]['in_channels']

        # Setup the flat layers
        self.flat = nn.Conv2d(flat_in_channels, flat_channels,
                              flat_kernel_size)
        self.flat2 = nn.Conv2d(flat_channels, flat_channels, 1)
        self.unflatten = nn.ConvTranspose2d(flat_channels, flat_out_channels,
                                            flat_kernel_size)

    def _validate_parameters(self, conv_block_args, deconv_block_args,
                             flat_channels, flat_kernel_size):
        # Invalid args
        if flat_channels <= 0:
            raise ValueError('flat_channels must be positive')
        if flat_kernel_size <= 0:
            raise ValueError('flat_kernel_size must be positive')

        # Invalid types
        if type(conv_block_args) != list:
            raise TypeError('conv_block_args must be a list of dict')
        if type(deconv_block_args) != list:
            raise TypeError('deconv_block_args must be a list of dict')

        # Test channel correspondence
        self._test_layer_configuration(conv_block_args)
        self._test_layer_configuration(deconv_block_args)

    @staticmethod
    def _test_layer_configuration(conf):
        if len(conf) > 1:
            for i in range(1, len(conf)):
                if conf[i-1]['out_channels'] != \
                        conf[i]['in_channels']:
                    raise ValueError('Channel in convolutional blocks are not '
                                     'in correspondence at layer {}-{}'
                                     .format(i-1, i))

    def forward(self, input):

        x, indices = self._forward_conv_blocks(input)

        x = self.flat(x)
        x = self.flat2(x)
        x = self.unflatten(x)

        x = self._forward_deconv_blocks(x, indices)

        return x

    def _forward_conv_blocks(self, input):
        # Run forward on all conv blocks and save indices in list
        x = input
        indices = []
        for block in self.conv_blocks:
            x, next_indices = block(x)
            indices.append(next_indices)

        return x, indices

    def _forward_deconv_blocks(self, input, indices):
        # Run forward on all deconv blocks using the indices saved from
        # running the conv blocks
        x = input
        for block, cur_indices in zip(self.deconv_blocks, reversed(indices)):
            x = block(x, cur_indices)

        return x


class StandardFeatureNet(DeconvNet):
    """
    The configuration used for my bachelor project. It has only 1
    output feature map with three sections of conv and deconv blocks
    respectively, totalling 21 layers including the flat section.

    The code might serve as an example of how the general DeconvNet
    can be instantiated in a boxed manner, hiding the complexity and
    cumbersome setup behind another class.
    """

    def __init__(self):
        conv_configuration = [
            {
                'in_channels': 3,
                'out_channels': 8,
                'kernel_size': 3
            },
            {
                'in_channels': 8,
                'out_channels': 64,
                'kernel_size': 3
            },
            {
                'in_channels': 64,
                'out_channels': 128,
                'kernel_size': 3
            }
        ]

        deconv_configuration = [
            {
                'in_channels': 128,
                'out_channels': 64,
                'kernel_size': 3
            },
            {
                'in_channels': 64,
                'out_channels': 8,
                'kernel_size': 3
            },
            {
                'in_channels': 8,
                'out_channels': 1,
                'kernel_size': 3
            }
        ]

        super().__init__(conv_configuration, deconv_configuration, 256, 14)

