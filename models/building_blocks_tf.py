import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import L2

nn = tf.keras.layers


# __all__ = []


class Block(Layer):
    def __init__(self, in_channels=3, out_channels=64, n_conv=2, max_pool='', batch_norm=True, activation='relu',
                 batch_norm_position='before', disable_bias=False, dim=None, padding='same',
                 weight_init_method='glorot_uniform'):
        super(Block, self).__init__()
        activation_dict = {'relu': nn.ReLU}
        layer_list = []

        if max_pool == 'before':
            layer_list.append(nn.MaxPool2D(pool_size=2, strides=2))
        for i in range(n_conv):
            # disable bias if conv is followed by a batchnorm layer for faster training
            #  bias=not (batch_norm and batch_norm_position == 'before')
            if i == 0 and max_pool != 'before':
                layer_list.append(nn.Conv2D(filters=out_channels, kernel_size=3, use_bias=not batch_norm,
                                            padding=padding, input_shape=(*dim, in_channels),
                                            kernel_regularizer=L2(.0005), kernel_initializer=weight_init_method))
            else:
                layer_list.append(nn.Conv2D(filters=out_channels, kernel_size=3, use_bias=not batch_norm,
                                            padding=padding, kernel_regularizer=L2(.0005),
                                            kernel_initializer=weight_init_method))

            if batch_norm_position == 'before' and batch_norm:
                layer_list.append(nn.BatchNormalization())

            layer_list.append(activation_dict[activation]())

        self.layers = keras.Sequential(layers=layer_list)

    def call(self, inputs, *args, **kwargs):
        return self.layers(inputs)


class Fusion(Layer):
    def __init__(self):
        super(Fusion, self).__init__()

    @staticmethod
    def call(x1, x2, factor):
        return (x1 + x2) / factor


class ConvBlock(Layer):
    def __init__(self, in_channels=3, out_channels=32, kernel_size=3, batch_norm=True, activation='relu', dim=None):
        super(ConvBlock, self).__init__()
        activation_dict = {'relu': nn.ReLU}
        activation_f = activation_dict[activation]
        layers = [nn.Conv2D(filters=out_channels, kernel_size=kernel_size, use_bias=not batch_norm,
                            input_shape=(*dim, in_channels))]
        if batch_norm:
            layers.append(nn.BatchNormalization())
        layers.append(activation_f())
        self.block = keras.Sequential(layers=layers)

    def call(self, inputs, *args, **kwargs):
        return self.block(inputs)


class UpConvConcatenate(Layer):
    def __init__(self, conv_t_filters=None, conv_filters=None, batch_norm=False, activation='relu',
                 padding='same', input_shape_conv=None, input_shape_upconv=None, weight_init_method='glorot_uniform'):
        super(UpConvConcatenate, self).__init__()
        if conv_filters is None:
            conv_filters = [1, 1]
        if conv_t_filters is None:
            conv_t_filters = [1, 1]
        in_channels, out_channels = conv_t_filters
        conv_in_channels, conv_out_channels = conv_filters
        activation_dict = {'relu': nn.ReLU}
        self.layers = []
        layer1 = keras.Sequential()
        layer2 = keras.Sequential()
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose
        layer1.add(nn.Conv2DTranspose(out_channels, kernel_size=3, strides=2, padding=padding,
                                      use_bias=not batch_norm, input_shape=(*input_shape_upconv, in_channels),
                                      kernel_regularizer=L2(.0005), kernel_initializer=weight_init_method))
        if batch_norm:
            layer1.add(nn.BatchNormalization())
        layer1.add(activation_dict[activation]())

        self.layers.append(layer1)

        layer2.add(nn.Conv2D(conv_out_channels, kernel_size=3, padding=padding, use_bias=not batch_norm,
                             input_shape=(*input_shape_conv, conv_in_channels),
                             kernel_regularizer=L2(.0005), kernel_initializer=weight_init_method))
        if batch_norm:
            layer2.add(nn.BatchNormalization())
        layer2.add(activation_dict[activation]())

        self.layers.append(layer2)

    def call(self, x1, x2):
        out = self.layers[0](x1)
        out = tf.concat([x2, out], axis=-1)
        # out = torch.cat([out, x2], dim=1)
        out = self.layers[1](out)
        return out


class VGG16(Layer):
    def __init__(self, in_channels=3, out_channels=32, batch_norm=True, activation='relu', input_dims=None,
                 padding='same'):
        super(VGG16, self).__init__()
        self.block1 = Block(in_channels, out_channels, n_conv=2, batch_norm=batch_norm, activation=activation,
                            dim=input_dims[0], padding=padding)
        self.block2 = Block(32, 64, n_conv=2, max_pool='before', batch_norm=batch_norm, activation=activation,
                            dim=input_dims[1], padding=padding)
        self.block3 = Block(64, 128, n_conv=3, max_pool='before', batch_norm=batch_norm, activation=activation,
                            dim=input_dims[2], padding=padding)
        self.block4 = Block(128, 256, n_conv=3, max_pool='before', batch_norm=batch_norm, activation=activation,
                            dim=input_dims[3], padding=padding)
        self.downsampling_factor = 8

        (input_dims[0] / self.downsampling_factor) * 4

        # if out_channels == 32:

    def call(self):
        pass


class FeatureExtractor(Layer):
    def __init__(self, in_channels=3, out_channels=32, batch_norm=True, activation='relu', input_dims=None,
                 padding='same', weight_init_method=''):
        super(FeatureExtractor, self).__init__()
        # if out_channels == 32:
        channels_seq = [out_channels * 2 ** i for i in range(4)]
        self.block1 = Block(in_channels, channels_seq[0], n_conv=2, batch_norm=batch_norm, activation=activation,
                            dim=input_dims[0], padding=padding, weight_init_method=weight_init_method)
        self.block2 = Block(channels_seq[0], channels_seq[1], n_conv=2, max_pool='before', batch_norm=batch_norm,
                            activation=activation,
                            dim=input_dims[1], padding=padding, weight_init_method=weight_init_method)
        self.block3 = Block(channels_seq[1], channels_seq[2], n_conv=3, max_pool='before', batch_norm=batch_norm,
                            activation=activation,
                            dim=input_dims[2], padding=padding, weight_init_method=weight_init_method)
        self.block4 = Block(channels_seq[2], channels_seq[3], n_conv=3, max_pool='before', batch_norm=batch_norm,
                            activation=activation,
                            dim=input_dims[3], padding=padding, weight_init_method=weight_init_method)
        self.up_conv_concat1 = UpConvConcatenate([channels_seq[3], channels_seq[2]], [channels_seq[3], channels_seq[1]],
                                                 batch_norm=batch_norm, padding=padding,
                                                 activation=activation, input_shape_upconv=input_dims[4][0],
                                                 input_shape_conv=input_dims[4][1],
                                                 weight_init_method=weight_init_method)
        self.up_conv_concat2 = UpConvConcatenate([channels_seq[1], channels_seq[1]], [channels_seq[2], channels_seq[0]],
                                                 batch_norm=batch_norm, padding=padding,
                                                 activation=activation, input_shape_upconv=input_dims[5][0],
                                                 input_shape_conv=input_dims[5][1],
                                                 weight_init_method=weight_init_method)
        self.up_conv_concat3 = UpConvConcatenate([channels_seq[0], channels_seq[0]], [channels_seq[1], channels_seq[0]],
                                                 batch_norm=batch_norm, padding=padding,
                                                 activation=activation, input_shape_upconv=input_dims[6][0],
                                                 input_shape_conv=input_dims[6][1],
                                                 weight_init_method=weight_init_method)


    def call(self, inputs, *args, **kwargs):
        out1 = self.block1(inputs)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        dec1 = self.up_conv_concat1(out4, out3)
        dec2 = self.up_conv_concat2(dec1, out2)
        dec3 = self.up_conv_concat3(dec2, out1)
        return dec3


class Predictor(Layer):
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
    _weight_init_method_ = 'weight_init_method'

    def __init__(self, config=None, roi_size=None):
        super(Predictor, self).__init__()

        self._roi_crop_size = roi_size

        layer_creation_function_dict = {'linear': self._create_fully_connected_linear,
                                        'conv': self._create_fully_conv}
        self.has_base_net = self._base_net_key in config

        self.layers_type = config.get(self._net_type_key)

        self.weight_init_method = config.get(self._weight_init_method_, 'glorot_uniform')
        self.layers = layer_creation_function_dict[self.layers_type](config)

    # print()

    def _create_fully_conv(self, config):

        layer_type = nn.Conv2D
        act_func_dict = {'softmax': nn.Softmax, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU}
        act_func_params = {'softmax': {'dim': 0}}

        input_channels = config.get(self._input_channels_key)

        base_net = config.get(self._base_net_key)
        headers = config.get(self._headers_key)

        # base_layers: list of base net's layers
        base_layers = []
        pr_channels_last = input_channels

        # layers variable contains all the network layers : base_net + headers
        layers = {}
        header_modules = {}
        if base_net:
            base_activation_functions = config.get(self._base_net_key + self._activations_key_suffix)
            base_kernel_sizes = config.get(self._base_net_key + self._kernel_size_key_suffix)
            base_batch_norm = config.get(self._base_net_key + self._batch_norm_key_suffix, len(base_net) * [False])
            base_dropout = config.get(self._base_net_key + self._dropout_key_suffix_, len(base_net) * [False])
            dropout_prob = config.get(self._base_net_key + self._dropout_prob_key_suffix_)
            for idx, l in enumerate(base_net):
                base_layers.append(layer_type(l, kernel_size=base_kernel_sizes[idx], kernel_regularizer=L2(.0005),
                                              kernel_initializer=self.weight_init_method))
                add_batch_norm = base_batch_norm[idx]
                add_dropout = base_dropout[idx]
                if add_batch_norm:
                    base_layers.append(nn.BatchNormalization())
                if base_activation_functions:
                    activation_f = base_activation_functions[idx]
                    activation_params = act_func_params.get(activation_f, {})
                    base_layers.append(act_func_dict[activation_f](**activation_params)
                                       if activation_f in act_func_params else act_func_dict[activation_f]())
                if add_dropout:
                    base_layers.append(nn.Dropout(dropout_prob[idx]))

                pr_channels_last = l
            layers['base_net'] = keras.Sequential(layers=base_layers)
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
                header_layers.append(layer_type(layer, kernel_size=header_kernel_sizes[idx],
                                                kernel_regularizer=L2(.0005),
                                                kernel_initializer=self.weight_init_method))
                add_dropout = headers_dropout[idx]
                add_batch_norm = headers_batch_norm[idx]
                if add_batch_norm:
                    header_layers.append(nn.BatchNormalization(layer))
                if current_header_activations:

                    activation_f = header_act_func[header][idx] if idx <= (len(header_act_func) - 1) else None
                    if activation_f:
                        activation_params = act_func_params.get(activation_f, {})
                        header_layers.append(act_func_dict[activation_f](**activation_params)
                                             if activation_f in act_func_params else act_func_dict[activation_f]())

                if add_dropout:
                    header_layers.append(nn.Dropout(headers_dropout_prob[idx]))
                pr_channels = layer
            header_modules[header] = keras.Sequential(layers=header_layers)

        layers['headers'] = header_modules

        return layers

    def _create_fully_connected_linear(self, config):
        self.has_base_net = True
        layer_type = nn.Dense
        act_func_dict = {'softmax': nn.Softmax, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU}
        act_func_params = {'softmax': {'dim': 0}}
        headers = config.get(self._headers_key)
        base_nn = config.get(self._base_net_key, None)

        base_layers = []
        layers = {}
        pr_channels_last = config.get(self._input_channels_key) * self._roi_crop_size ** 2
        headers_dropout_prob = config.get(self._headers_key + self._dropout_prob_key_suffix_)
        base_layers.append(nn.Flatten())
        if base_nn:
            base_activation_functions = config.get(self._base_net_key + self._activations_key_suffix)
            base_batch_norm = config.get(self._base_net_key + self._batch_norm_key_suffix, len(base_nn) * [False])
            base_dropout = config.get(self._base_net_key + self._dropout_key_suffix_, len(base_nn) * [False])
            dropout_prob = config.get(self._base_net_key + self._dropout_prob_key_suffix_)
            for idx, l in enumerate(base_nn):
                base_layers.append(layer_type(l, kernel_regularizer=L2(.005),
                                              kernel_initializer=self.weight_init_method))
                add_batch_norm = base_batch_norm[idx]
                add_dropout = base_dropout[idx]
                if add_batch_norm:
                    base_layers.append(nn.BatchNormalization())
                if base_activation_functions:
                    activation_f = base_activation_functions[idx]
                    activation_params = act_func_params.get(activation_f, {})
                    base_layers.append(act_func_dict[activation_f](**activation_params))

                if add_dropout:
                    base_layers.append(nn.Dropout(dropout_prob[idx]))
                pr_channels_last = l

        layers['base_net'] = keras.Sequential(layers=base_layers)

        #  CREATING HEADERS
        header_act_func = config.get(self._headers_key + self._activations_key_suffix, {})
        header_modules = {}
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
                header_layers.append(layer_type(layer, kernel_regularizer=L2(.005),
                                                kernel_initializer=self.weight_init_method))
                # header_layers.append(layer_type(pr_channels, layer))
                add_batch_norm = headers_batch_norm[idx]
                add_dropout = headers_dropout[idx]
                if add_batch_norm:
                    header_layers.append(nn.BatchNormalization())
                if current_header_activations:
                    activation_f = current_header_activations[idx]
                    activation_params = act_func_params.get(activation_f, {})
                    if activation_f is not None:
                        header_layers.append(act_func_dict[activation_f](**activation_params)
                                             if activation_f in act_func_params else act_func_dict[activation_f]())

                if add_dropout:
                    header_layers.append(nn.Dropout(headers_dropout_prob[idx]))
                pr_channels = layer
            header_modules[header] = keras.Sequential(layers=header_layers)
        layers['headers'] = header_modules
        return layers


    def call(self, x):
        if self.has_base_net:
            x = self.layers['base_net'](x)
        return [tf.squeeze(header(x)) for header_name, header in self.layers['headers'].items()]
