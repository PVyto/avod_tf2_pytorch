import torch
import torch.nn as nn
from torchvision.ops import roi_align, nms

__all__ = []


class Block(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, n_conv=2, max_pool='', batch_norm=True, activation='relu',
                 batch_norm_position='before', disable_bias=False):
        super(Block, self).__init__()
        activation_dict = {'relu': nn.ReLU, 'selu': nn.SELU}
        channels = in_channels
        layer_list = []
        # self.block
        if max_pool == 'before':
            layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
        for i in range(n_conv):
            # disable bias if conv is followed by a batchnorm layer for faster training
            #  bias=not (batch_norm and batch_norm_position == 'before')
            layer_list.append(nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=3, padding=1,
                                        bias=not batch_norm))
            if batch_norm_position == 'before' and batch_norm:
                layer_list.append(nn.BatchNorm2d(out_channels, momentum=1e-3,eps=1e-3))

            layer_list.append(activation_dict[activation]())

            if batch_norm and batch_norm_position == 'after':
                layer_list.append(nn.BatchNorm2d(out_channels))

            channels = out_channels
        if max_pool == 'after':
            layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()

    @staticmethod
    @torch.jit.script
    def forward(x1, x2, factor):
        return (x1 + x2) / factor
    # torch.jit.script decorator can fuse point-wise operations into a single CUDA kernel


class ConvBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, kernel_size=3, batch_norm=True, activation='relu'):
        super(ConvBlock, self).__init__()
        activation_dict = {'relu': nn.ReLU, 'selu': nn.SELU}
        activation_f = activation_dict[activation]
        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                            bias=not batch_norm)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels, momentum=1e-3, eps=1e-3))
        layers.append(activation_f())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpConvConcatenate(nn.Module):
    def __init__(self, conv_t_filters=None, conv_filters=None, batch_norm=False, activation='relu',
                 batch_norm_position='before'):
        super(UpConvConcatenate, self).__init__()
        if conv_filters is None:
            conv_filters = [1, 1]
        if conv_t_filters is None:
            conv_t_filters = [1, 1]
        activation_dict = {'relu': nn.ReLU, 'selu': nn.SELU}
        self.layers = nn.ModuleList()
        layer1 = []
        layer2 = []
        # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        layer1.append(nn.ConvTranspose2d(*conv_t_filters, kernel_size=3, padding=1, stride=2, output_padding=1,
                                         bias=not batch_norm))
        if batch_norm_position == 'before' and batch_norm:
            layer1.append(nn.BatchNorm2d(conv_t_filters[-1], momentum=1e-3, eps=1e-3))
        layer1.append(activation_dict[activation]())
        if batch_norm_position == 'after' and batch_norm:
            layer1.append(nn.BatchNorm2d(conv_t_filters[-1], momentum=1e-3))

        self.layers.add_module('layer1', nn.Sequential(*layer1))

        layer2.append(nn.Conv2d(*conv_filters, kernel_size=3, padding=1, bias=not batch_norm))
        if batch_norm_position == 'before' and batch_norm:
            layer2.append(nn.BatchNorm2d(conv_filters[-1], momentum=1e-3, eps=1e-3))
        layer2.append(activation_dict[activation]())
        if batch_norm_position == 'after' and batch_norm:
            layer2.append(nn.BatchNorm2d(conv_filters[-1]))

        self.layers.add_module('layer2', nn.Sequential(*layer2))

    def forward(self, x1, x2):
        out = self.layers.layer1(x1)
        out = torch.cat([x2, out], dim=1)
        # out = torch.cat([out, x2], dim=1)
        out = self.layers.layer2(out)
        return out


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=8, batch_norm=False, activation='relu'):
        super(FeatureExtractor, self).__init__()
        if out_channels == 32:
            self.block1 = Block(in_channels, out_channels, n_conv=2, batch_norm=batch_norm, activation=activation)
            self.block2 = Block(32, 64, n_conv=2, max_pool='before', batch_norm=batch_norm, activation=activation)
            self.block3 = Block(64, 128, n_conv=3, max_pool='before', batch_norm=batch_norm, activation=activation)
            self.block4 = Block(128, 256, n_conv=3, max_pool='before', batch_norm=batch_norm, activation=activation)

            self.up_conv_concat1 = UpConvConcatenate([256, 128], [256, 64], batch_norm=batch_norm,
                                                     activation=activation)
            self.up_conv_concat2 = UpConvConcatenate([64, 64], [128, 32], batch_norm=batch_norm, activation=activation)
            self.up_conv_concat3 = UpConvConcatenate([32, 32], [64, 32], batch_norm=batch_norm, activation=activation)
        else:
            self.block1 = Block(in_channels, out_channels, n_conv=2, batch_norm=batch_norm, activation=activation)
            self.block2 = Block(8, 16, n_conv=2, max_pool='before', batch_norm=batch_norm, activation=activation)
            self.block3 = Block(16, 32, n_conv=3, max_pool='before', batch_norm=batch_norm, activation=activation)
            self.block4 = Block(32, 64, n_conv=3, max_pool='before', batch_norm=batch_norm, activation=activation)

            self.up_conv_concat1 = UpConvConcatenate([64, 32], [64, 16], batch_norm=batch_norm, activation=activation)
            self.up_conv_concat2 = UpConvConcatenate([16, 16], [32, 8], batch_norm=batch_norm, activation=activation)
            self.up_conv_concat3 = UpConvConcatenate([8, 8], [16, 8], batch_norm=batch_norm, activation=activation)


    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        dec1 = self.up_conv_concat1(out4, out3)
        dec2 = self.up_conv_concat2(dec1, out2)
        dec3 = self.up_conv_concat3(dec2, out1)

        return dec3


class Predictor(nn.Module):
    _net_type_key = 'type'
    _base_net_key = 'base'
    _headers_key = 'headers'
    _activations_key_suffix = '_activations'
    _norm_key_suffix = 'norm'
    _input_channels_key = 'input_channels'
    _kernel_size_key_suffix = '_kernel_size'
    _batch_norm_key_suffix = '_batch_norm'
    _dropout_key_suffix_ = '_dropout'
    _dropout_prob_key_suffix_ = '_dropout_probability'

    def __init__(self, config=None, roi_size=None):
        super(Predictor, self).__init__()
        self._roi_crop_size = roi_size

        layer_creation_function_dict = {'linear': self._create_fully_connected_linear,
                                        'conv': self._create_fully_conv}
        self.has_base_net = self._base_net_key in config

        self.layers_type = config.get(self._net_type_key)

        self.layers = layer_creation_function_dict[self.layers_type](config)

    # print()

    def _create_fully_conv(self, config):

        layer_type = nn.Conv2d
        act_func_dict = {'softmax': nn.Softmax, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'selu': nn.SELU}
        act_func_params = {'softmax': {'dim': 0}}

        input_channels = config.get(self._input_channels_key)

        base_net = config.get(self._base_net_key)
        headers = config.get(self._headers_key)

        # base_layers: list of base net's layers
        base_layers = []
        pr_channels_last = input_channels

        # layers variable contains all the network layers : base_net + headers
        layers = nn.ModuleList()
        header_modules = nn.ModuleList()
        if base_net:
            base_activation_functions = config.get(self._base_net_key + self._activations_key_suffix)
            base_kernel_sizes = config.get(self._base_net_key + self._kernel_size_key_suffix)
            base_batch_norm = config.get(self._base_net_key + self._batch_norm_key_suffix, len(base_net) * [False])
            base_dropout = config.get(self._base_net_key + self._dropout_key_suffix_, len(base_net) * [False])
            dropout_prob = config.get(self._base_net_key + self._dropout_prob_key_suffix_)
            for idx, l in enumerate(base_net):
                base_layers.append(layer_type(pr_channels_last, l, kernel_size=base_kernel_sizes[idx]))
                add_batch_norm = base_batch_norm[idx]
                add_dropout = base_dropout[idx]
                if add_batch_norm:
                    base_layers.append(nn.BatchNorm2d(l, momentum=1e-3))
                if base_activation_functions:
                    activation_f = base_activation_functions[idx]
                    activation_params = act_func_params.get(activation_f, {})
                    base_layers.append(act_func_dict[activation_f](**activation_params)
                                       if activation_f in act_func_params else act_func_dict[activation_f]())
                if add_dropout:
                    base_layers.append(nn.Dropout2d(dropout_prob[idx]))

                pr_channels_last = l
            layers.add_module('base_net', nn.Sequential(*base_layers))
        header_act_func = config.get(self._headers_key + self._activations_key_suffix, {})

        headers_dropout_prob = config.get(self._headers_key + self._dropout_prob_key_suffix_)
        for header in headers:
            header_info = config.get(header)
            number_of_layers = len(header_info)
            header_kernel_sizes = config.get(self._headers_key + self._kernel_size_key_suffix)
            headers_batch_norm = config.get(self._headers_key + self._batch_norm_key_suffix, number_of_layers * [False])
            headers_dropout = config.get(self._headers_key + self._dropout_key_suffix_, number_of_layers * [False])
            current_header_activations = header_act_func.get(header)
            pr_channels = pr_channels_last
            header_layers = []
            for idx, layer in enumerate(header_info):
                header_layers.append(layer_type(pr_channels, layer, kernel_size=header_kernel_sizes[idx]))
                add_dropout = headers_dropout[idx]
                add_batch_norm = headers_batch_norm[idx]
                if add_batch_norm:
                    header_layers.append(nn.BatchNorm2d(layer, momentum=1e-3))
                if current_header_activations:
                    # select the current header using header's value as key to header_act_func dictionary
                    # and extract the activation function of the layer that corresponds to idx's value
                    # if is not there for some reason just get None
                    activation_f = header_act_func[header][idx] if idx <= (len(header_act_func) - 1) else None
                    if activation_f:
                        activation_params = act_func_params.get(activation_f, {})
                        header_layers.append(act_func_dict[activation_f](**activation_params)
                                             if activation_f in act_func_params else act_func_dict[activation_f]())

                if add_dropout:
                    header_layers.append(nn.Dropout2d(headers_dropout_prob[idx]))
                pr_channels = layer
            header_modules.add_module(header, nn.Sequential(*header_layers))

        layers.add_module('headers', header_modules)
        # layers.add_module(header, nn.Sequential(*header_layers))

        return layers

    def _create_fully_connected_linear(self, config):
        self.has_base_net = True
        layer_type = nn.Linear
        act_func_dict = {'softmax': nn.Softmax, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'selu': nn.SELU}
        act_func_params = {'softmax': {'dim': 0}}
        headers = config.get(self._headers_key)
        base_nn = config.get(self._base_net_key, None)

        base_layers = []
        layers = nn.ModuleList()
        pr_channels_last = config.get(self._input_channels_key) * self._roi_crop_size ** 2
        headers_dropout_prob = config.get(self._headers_key + self._dropout_prob_key_suffix_)
        base_layers.append(nn.Flatten())
        if base_nn:
            base_activation_functions = config.get(self._base_net_key + self._activations_key_suffix)
            base_batch_norm = config.get(self._base_net_key + self._batch_norm_key_suffix, len(base_nn) * [False])
            base_dropout = config.get(self._base_net_key + self._dropout_key_suffix_, len(base_nn) * [False])
            dropout_prob = config.get(self._base_net_key + self._dropout_prob_key_suffix_)
            for idx, l in enumerate(base_nn):
                base_layers.append(layer_type(pr_channels_last, l))
                add_batch_norm = base_batch_norm[idx]
                add_dropout = base_dropout[idx]
                if add_batch_norm:
                    base_layers.append(nn.BatchNorm1d(l))
                if base_activation_functions:
                    activation_f = base_activation_functions[idx]
                    activation_params = act_func_params.get(activation_f, {})
                    base_layers.append(act_func_dict[activation_f](**activation_params))

                if add_dropout:
                    base_layers.append(nn.Dropout(dropout_prob[idx]))
                pr_channels_last = l

        layers.add_module('base_net', nn.Sequential(*base_layers))

        #  CREATING HEADERS
        header_act_func = config.get(self._headers_key + self._activations_key_suffix, {})
        header_modules = nn.ModuleList()
        for header in headers:
            header_info = config.get(header, [])
            number_of_layers = len(header_info)
            headers_batch_norm = config.get(self._headers_key + self._batch_norm_key_suffix, len(header_info) * [False])
            headers_dropout = config.get(self._headers_key + self._dropout_key_suffix_, len(header_info) * [False])

            current_header_activations = header_act_func.get(header, number_of_layers * [None])
            current_header_activations += (number_of_layers - len(current_header_activations)) * [None]
            pr_channels = pr_channels_last
            # header_layers = nn.ModuleList()
            header_layers = []
            for idx, layer in enumerate(header_info):
                header_layers.append(layer_type(pr_channels, layer))
                # header_layers.append(layer_type(pr_channels, layer))
                add_batch_norm = headers_batch_norm[idx]
                add_dropout = headers_dropout[idx]
                if add_batch_norm:
                    header_layers.append(nn.BatchNorm1d(layer))
                if current_header_activations:
                    activation_f = current_header_activations[idx]
                    activation_params = act_func_params.get(activation_f, {})
                    if activation_f is not None:
                        header_layers.append(act_func_dict[activation_f](**activation_params)
                                             if activation_f in act_func_params else act_func_dict[activation_f]())

                if add_dropout:
                    header_layers.append(nn.Dropout(headers_dropout_prob[idx]))
                pr_channels = layer
            header_modules.add_module(header, nn.Sequential(*header_layers))
        layers.add_module('headers', header_modules)
        # layers.add_module(header, nn.Sequential(*header_layers))
        return layers

    # @timeit_('Predictor_')
    def forward(self, x):
        # print('input shape:', x.shape)
        if self.has_base_net:
            x = self.layers.base_net(x)
        # print(x.shape)
        return [header(x).squeeze() for header in self.layers.headers]


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

    def forward(self):
        pass


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        pass

    def forward(self):
        pass

