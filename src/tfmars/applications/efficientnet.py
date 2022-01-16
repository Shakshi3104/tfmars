import math
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from ..modules.attention import BaseAttention


class Block:
    def __init__(self, activation='relu', drop_rate=0., name='', filters_in=32, filters_out=16, kernel_size=3,
                 strides=1, expand_ratio=1, se_ratio=0., id_skip=True):
        self.activation = activation
        self.drop_rate = drop_rate
        self.name = name
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.kernel_size = kernel_size
        self.strides = strides
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.id_skip = id_skip

    def __call__(self, x):
        inputs = x
        # Expansion phase
        filters = self.filters_in * self.expand_ratio
        if self.expand_ratio != 1:
            x = layers.Conv1D(
                filters,
                1,
                padding='same',
                use_bias=False,
                kernel_initializer='he_normal',
                name=self.name + 'expand_conv'
            )(x)
            x = layers.BatchNormalization(name=self.name + "expand_bn")(x)
            x = layers.Activation(self.activation, name=self.name + "expand_activation")(x)

        else:
            x = inputs

        # Depthwise Convolution
        conv_pad = 'same'
        x = layers.SeparableConv1D(int(x.shape[-1]),
                                   self.kernel_size,
                                   strides=self.strides,
                                   padding=conv_pad,
                                   use_bias=False,
                                   depthwise_initializer='he_normal',
                                   name=self.name + "dwconv")(x)
        x = layers.BatchNormalization(name=self.name + 'bn')(x)
        x = layers.Activation(self.activation, name=self.name + 'activation')(x)

        # Squeeze and Excitation phase
        if 0 < self.se_ratio <= 1:
            filters_se = max(1, int(self.filters_in + self.se_ratio))
            se = layers.GlobalAveragePooling1D(name=self.name + 'se_squeeze')(x)
            se = layers.Reshape((1, filters), name=self.name + 'se_reshape')(se)
            se = layers.Conv1D(
                filters_se,
                1,
                padding='same',
                activation=self.activation,
                kernel_initializer='he_normal',
                name=self.name + 'se_reduce'
            )(se)
            se = layers.Conv1D(
                filters,
                1,
                padding='same',
                activation='sigmoid',
                kernel_initializer='he_normal',
                name=self.name + 'se_expand'
            )(se)
            x = layers.multiply([x, se], name=self.name + 'se_excite')

        # Output phase
        x = layers.Conv1D(self.filters_out,
                          1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer='he_normal',
                          name=self.name + 'project_conv')(x)
        x = layers.BatchNormalization(name=self.name + 'project_bn')(x)
        if self.id_skip and self.strides == 1 and self.filters_in == self.filters_out:
            if self.drop_rate > 0:
                x = layers.Dropout(self.drop_rate, name=self.name + 'drop')(x)
            x = layers.add([x, inputs], name=self.name + 'add')

        return x


def EfficientNetWithAttention(width_coefficient, depth_coefficient, dropout_rate=0.2, drop_connect_rate=0.2,
                              depth_divisor=8, activation='relu', blocks_args=None,
                              include_top=True, input_shape=(256, 3), pooling=None, classes=6,
                              classifier_activation='softmax', module: BaseAttention = None):
    """
    Parameters
    ----------
    width_coefficient
    depth_coefficient
    dropout_rate
    drop_connect_rate
    depth_divisor
    activation
    blocks_args
    include_top
    input_shape
    pooling
    classes
    classifier_activation
    module
    Returns
    -------
    """
    inputs = layers.Input(shape=input_shape)

    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    # Build stem
    x = inputs

    x = layers.Conv1D(
        round_filters(32),
        3,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name='stem_conv'
    )(x)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation(activation, name='stem_activation')(x)

    # Build blocks
    if blocks_args is None:
        blocks_args = [{
            'kernel_size': 3,
            'repeats': 1,
            'filters_in': 32,
            'filters_out': 16,
            'expand_ratio': 1,
            'id_skip': True,
            'strides': 1,
            'se_ratio': 0.25
        }, {
            'kernel_size': 3,
            'repeats': 2,
            'filters_in': 16,
            'filters_out': 24,
            'expand_ratio': 6,
            'id_skip': True,
            'strides': 2,
            'se_ratio': 0.25
        }, {
            'kernel_size': 5,
            'repeats': 2,
            'filters_in': 24,
            'filters_out': 40,
            'expand_ratio': 6,
            'id_skip': True,
            'strides': 2,
            'se_ratio': 0.25
        }, {
            'kernel_size': 3,
            'repeats': 3,
            'filters_in': 40,
            'filters_out': 80,
            'expand_ratio': 6,
            'id_skip': True,
            'strides': 2,
            'se_ratio': 0.25
        }, {
            'kernel_size': 5,
            'repeats': 3,
            'filters_in': 80,
            'filters_out': 112,
            'expand_ratio': 6,
            'id_skip': True,
            'strides': 1,
            'se_ratio': 0.25
        }, {
            'kernel_size': 5,
            'repeats': 4,
            'filters_in': 112,
            'filters_out': 192,
            'expand_ratio': 6,
            'id_skip': True,
            'strides': 2,
            'se_ratio': 0.25
        }, {
            'kernel_size': 3,
            'repeats': 1,
            'filters_in': 192,
            'filters_out': 320,
            'expand_ratio': 6,
            'id_skip': True,
            'strides': 1,
            'se_ratio': 0.25
        }]

    b = 0

    blocks = [round_repeats(args['repeats']) for args in blocks_args]
    blocks = float(sum(blocks))

    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0

        # Update block input and output filters based on depth multiplier.
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']

            x = Block(activation, drop_connect_rate * b / blocks,
                      name='block{}{}_'.format(i + 1, chr(j + 97)), **args)(x)
            b += 1

        if module is not None:
            x = module(x.shape[-1], block_name='block{}'.format(i + 1))(x)

    # Build top
    x = layers.Conv1D(
        round_filters(1280),
        1,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        name='top_conv'
    )(x)
    x = layers.BatchNormalization(name='top_bn')(x)
    x = layers.Activation(activation, name='top_activation')(x)

    if include_top:
        x = layers.GlobalAveragePooling1D(name='avg_pool')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(classes, activation=classifier_activation, name='predictions')(x)

        model = Model(inputs, x)
        return model
    else:
        if pooling is None:
            model_ = Model(inputs=inputs, outputs=x)
            return model_
        elif pooling == 'avg':
            x = layers.GlobalAveragePooling1D(name="avgpool")(x)
            model_ = Model(inputs=inputs, outputs=x)
            return model_
        elif pooling == 'max':
            x = layers.GlobalMaxPooling1D(name="maxpool")(x)
            model_ = Model(inputs=inputs, outputs=x)
            return model_
        else:
            print("Not exist pooling option: {}".format(pooling))
            model_ = Model(inputs=inputs, outputs=x)
            return model_


# Attention-insertable EfficientNet B0
def EfficientNetB0WithAttention(include_top=True, input_shape=(256, 3), pooling=None, classes=6,
                                classifier_activation='softmax', module: BaseAttention = None):
    """
    Parameters
    ----------
    include_top
    input_shape
    pooling
    classes
    classifier_activation
    module
    Returns
    -------
    """
    return EfficientNetWithAttention(1.0, 1.0, 0.2, include_top=include_top, input_shape=input_shape, pooling=pooling,
                                     classes=classes, classifier_activation=classifier_activation, module=module)


# EfficientNet B0
def EfficientNetB0(include_top=True, input_shape=(256, 3), pooling=None, classes=6, classifier_activation='softmax'):
    return EfficientNetB0WithAttention(include_top, input_shape, pooling, classes, classifier_activation, module=None)
