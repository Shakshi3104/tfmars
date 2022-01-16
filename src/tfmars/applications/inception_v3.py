from tensorflow.keras import layers
from tensorflow.keras.models import Model
from ..modules.attention import BaseAttention


class Conv1DBN:
    def __init__(self, filters, kernel_size, padding='same', strides=1, name=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.name = name

    def __call__(self, x):
        if self.name is not None:
            bn_name = self.name + "_bn"
            conv_name = self.name + "_conv"
        else:
            bn_name = None
            conv_name = None

        x = layers.Conv1D(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=False,
            name=conv_name
        )(x)
        x = layers.BatchNormalization(scale=False, name=bn_name)(x)
        x = layers.Activation('relu', name=self.name)(x)
        return x


def InceptionV3WithAttention(include_top=True, input_shape=(256, 3), pooling=None, classes=6,
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

    inputs = layers.Input(shape=input_shape)

    x = Conv1DBN(32, 3, strides=2, padding='valid')(inputs)
    x = Conv1DBN(32, 3, padding='valid')(x)
    x = Conv1DBN(64, 3)(x)
    x = layers.MaxPooling1D(3, strides=2)(x)
    if module is not None:
        x = module(64, block_name="conv1")(x)

    x = Conv1DBN(80, 1, padding='valid')(x)
    x = Conv1DBN(192, 3, padding='valid')(x)
    x = layers.MaxPooling1D(3, strides=2)(x)
    if module is not None:
        x = module(192, block_name="conv2")(x)

    # mixed 0
    branch1x1 = Conv1DBN(64, 1)(x)

    branch5x5 = Conv1DBN(48, 1)(x)
    branch5x5 = Conv1DBN(64, 5)(branch5x5)

    branch3x3dbl = Conv1DBN(64, 1)(x)
    branch3x3dbl = Conv1DBN(96, 3)(branch3x3dbl)
    branch3x3dbl = Conv1DBN(96, 3)(branch3x3dbl)

    branch_pool = layers.AveragePooling1D(3, strides=1, padding='same')(x)
    branch_pool = Conv1DBN(32, 1)(branch_pool)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], name='mixed0')

    if module is not None:
        x = module(256, block_name="mixed0")(x)

    # mixed 1
    branch1x1 = Conv1DBN(64, 1)(x)

    branch5x5 = Conv1DBN(48, 1)(x)
    branch5x5 = Conv1DBN(64, 5)(branch5x5)

    branch3x3dbl = Conv1DBN(64, 1)(x)
    branch3x3dbl = Conv1DBN(96, 3)(branch3x3dbl)
    branch3x3dbl = Conv1DBN(96, 3)(branch3x3dbl)

    branch_pool = layers.AveragePooling1D(3, strides=1, padding='same')(x)
    branch_pool = Conv1DBN(64, 1)(branch_pool)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], name='mixed1')

    if module is not None:
        x = module(288, block_name="mixed1")(x)

    # mixed 2
    branch1x1 = Conv1DBN(64, 1)(x)

    branch5x5 = Conv1DBN(48, 1)(x)
    branch5x5 = Conv1DBN(64, 5)(branch5x5)

    branch3x3dbl = Conv1DBN(64, 1)(x)
    branch3x3dbl = Conv1DBN(96, 3)(branch3x3dbl)
    branch3x3dbl = Conv1DBN(96, 3)(branch3x3dbl)

    branch_pool = layers.AveragePooling1D(3, strides=1, padding='same')(x)
    branch_pool = Conv1DBN(64, 1)(branch_pool)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], name='mixed2')

    if module is not None:
        x = module(288, block_name="mixed2")(x)

    # mixed 3
    branch3x3 = Conv1DBN(384, 3, strides=2, padding='valid')(x)

    branch3x3dbl = Conv1DBN(64, 1)(x)
    branch3x3dbl = Conv1DBN(96, 3)(branch3x3dbl)
    branch3x3dbl = Conv1DBN(96, 3, strides=2, padding='valid')(branch3x3dbl)

    branch_pool = layers.MaxPooling1D(3, strides=2)(x)
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], name='mixed3')

    if module is not None:
        x = module(768, block_name="mixed3")(x)

    # mixed 4
    branch1x1 = Conv1DBN(192, 1)(x)

    branch7x7 = Conv1DBN(128, 1)(x)
    branch7x7 = Conv1DBN(128, 1)(branch7x7)
    branch7x7 = Conv1DBN(128, 7)(branch7x7)

    branch7x7dbl = Conv1DBN(128, 1)(x)
    branch7x7dbl = Conv1DBN(128, 7)(branch7x7dbl)
    branch7x7dbl = Conv1DBN(128, 1)(branch7x7dbl)
    branch7x7dbl = Conv1DBN(128, 7)(branch7x7dbl)
    branch7x7dbl = Conv1DBN(128, 1)(branch7x7dbl)

    branch_pool = layers.AveragePooling1D(3, strides=1, padding='same')(x)
    branch_pool = Conv1DBN(192, 1)(branch_pool)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], name='mixed4')

    if module is not None:
        x = module(640, block_name="mixed4")(x)

    # mixed 5, 6
    for i in range(2):
        branch1x1 = Conv1DBN(192, 1)(x)

        branch7x7 = Conv1DBN(160, 1)(x)
        branch7x7 = Conv1DBN(160, 1)(branch7x7)
        branch7x7 = Conv1DBN(192, 7)(branch7x7)

        branch7x7dbl = Conv1DBN(160, 1)(x)
        branch7x7dbl = Conv1DBN(160, 7)(branch7x7dbl)
        branch7x7dbl = Conv1DBN(160, 1)(branch7x7dbl)
        branch7x7dbl = Conv1DBN(160, 7)(branch7x7dbl)
        branch7x7dbl = Conv1DBN(192, 1)(branch7x7dbl)

        branch_pool = layers.AveragePooling1D(3, strides=1, padding='same')(x)
        branch_pool = Conv1DBN(192, 1)(branch_pool)
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], name='mixed' + str(5 + i))

        if module is not None:
            x = module(768, block_name="mixed" + str(5 + i))(x)

    # mixed 7
    branch1x1 = Conv1DBN(192, 1)(x)

    branch7x7 = Conv1DBN(192, 1)(x)
    branch7x7 = Conv1DBN(192, 1)(branch7x7)
    branch7x7 = Conv1DBN(192, 7)(branch7x7)

    branch7x7dbl = Conv1DBN(192, 1)(x)
    branch7x7dbl = Conv1DBN(192, 7)(branch7x7dbl)
    branch7x7dbl = Conv1DBN(192, 1)(branch7x7dbl)
    branch7x7dbl = Conv1DBN(192, 7)(branch7x7dbl)
    branch7x7dbl = Conv1DBN(192, 1)(branch7x7dbl)

    branch_pool = layers.AveragePooling1D(3, strides=1, padding='same')(x)
    branch_pool = Conv1DBN(192, 1)(branch_pool)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], name='mixed7')

    if module is not None:
        x = module(768, block_name="mixed7")(x)

    # mixed 8
    branch3x3 = Conv1DBN(192, 1)(x)
    branch3x3 = Conv1DBN(320, 3, strides=2, padding='valid')(branch3x3)

    branch7x7x3 = Conv1DBN(192, 1)(x)
    branch7x7x3 = Conv1DBN(192, 1)(branch7x7x3)
    branch7x7x3 = Conv1DBN(192, 7)(branch7x7x3)
    branch7x7x3 = Conv1DBN(192, 3, strides=2, padding='valid')(branch7x7x3)

    branch_pool = layers.MaxPooling1D(3, strides=2)(x)
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], name='mixed8')

    if module is not None:
        x = module(1280, block_name="mixed8")(x)

    # mixed 9, 10
    for i in range(2):
        branch1x1 = Conv1DBN(320, 1)(x)

        branch3x3 = Conv1DBN(384, 1)(x)
        branch3x3_1 = Conv1DBN(384, 1)(branch3x3)
        branch3x3_2 = Conv1DBN(384, 3)(branch3x3)
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], name='mixed9_' + str(i))

        branch3x3dbl = Conv1DBN(448, 1)(x)
        branch3x3dbl = Conv1DBN(384, 3)(branch3x3dbl)
        branch3x3dbl_1 = Conv1DBN(384, 1)(branch3x3dbl)
        branch3x3dbl_2 = Conv1DBN(384, 3)(branch3x3dbl)
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2])

        branch_pool = layers.AveragePooling1D(3, strides=1, padding='same')(x)
        branch_pool = Conv1DBN(192, 1)(branch_pool)
        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], name='mixed' + str(9 + i))

        if module is not None:
            x = module(2048, block_name="mixed" + str(9 + i))(x)

    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling1D(name='avg_pool')(x)
        y = layers.Dense(classes, activation=classifier_activation, name='predictions')(x)

        model = Model(inputs, y)
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
