import tensorflow as tf
from ..modules.attention import BaseAttention


class Conv3:
    def __init__(self, out_planes, stride=1, name=""):
        self.out_planes = out_planes
        self.stride = stride
        self.name = name

    def __call__(self, x):
        x = tf.keras.layers.Conv1D(self.out_planes, 3, strides=self.stride, padding='same', use_bias=False,
                                   name=self.name)(x)
        return x


class Conv1:
    def __init__(self, out_planes, stride=1, name=""):
        self.out_planes = out_planes
        self.stride = stride
        self.name = name

    def __call__(self, x):
        x = tf.keras.layers.Conv1D(self.out_planes, 1, strides=self.stride, padding='same', use_bias=False,
                                   name=self.name)(x)
        return x


class BasicBlock:
    def __init__(self, planes, stride=1, downsample=None, block_name=""):
        self.planes = planes
        self.stride = stride
        self.downsample = downsample
        self.block_name = block_name

    def __call__(self, x):
        identity = x

        x = tf.keras.layers.BatchNormalization(name=self.block_name + "_bn_1")(x)
        x = Conv3(self.planes, stride=self.stride, name=self.block_name + "_conv3_1")(x)

        x = tf.keras.layers.BatchNormalization(name=self.block_name + "_bn_2")(x)
        x = tf.keras.layers.ReLU(name=self.block_name + "_relu")(x)

        x = Conv3(self.planes, stride=1, name=self.block_name + "_conv3_2")(x)
        x = tf.keras.layers.BatchNormalization(name=self.block_name + "_bn_3")(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        ch_x = x.shape[-1]
        ch_identity = identity.shape[-1]

        if ch_x != ch_identity:
            identity = tf.pad(identity, [[0, 0], [0, 0], [0, ch_x - ch_identity]])

        x = tf.keras.layers.Add(name=self.block_name + "_add")([x, identity])
        return x


class Bottleneck:
    def __init__(self, planes, stride=1, downsample=None, block_name=""):
        self.planes = planes
        self.stride = stride
        self.downsample = downsample
        self.block_name = block_name

    def __call__(self, x):
        expansion = 4
        width = self.planes

        identity = x

        x = tf.keras.layers.BatchNormalization(name=self.block_name + "_bn_1b")(x)
        x = Conv1(width, stride=1, name=self.block_name + "_conv1_1b")(x)

        x = tf.keras.layers.BatchNormalization(name=self.block_name + "_bn_2b")(x)
        x = tf.keras.layers.ReLU(name=self.block_name + "_relu_2b")(x)
        x = Conv3(width, stride=self.stride, name=self.block_name + "_conv3_2b")(x)

        x = tf.keras.layers.BatchNormalization(name=self.block_name + "_bn_3b")(x)
        x = tf.keras.layers.ReLU(name=self.block_name + "_relu_3b")(x)
        x = Conv1(width * expansion, stride=1, name=self.block_name + "_conv1_expand")(x)

        x = tf.keras.layers.BatchNormalization(name=self.block_name + "_bn_4")(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        ch_x = x.shape[-1]
        ch_identity = identity.shape[-1]

        if ch_x != ch_identity:
            identity = tf.pad(identity, [[0, 0], [0, 0], [0, ch_x - ch_identity]])

        x = tf.keras.layers.Add(name=self.block_name + "_add")([x, identity])
        return x


class Stack:
    def __init__(self, block_fn, n_layer, add_rate, stride, is_first=False, name=""):
        self.block_fn = block_fn
        self.n_layer = n_layer
        self.add_rate = add_rate
        self.stride = stride
        self.is_first = is_first
        self.name = name

    def __call__(self, x):
        if self.block_fn is BasicBlock:
            expansion = 1
        elif self.block_fn is Bottleneck:
            expansion = 4

        downsample = None

        if self.stride != 1:
            def _downsample(x):
                x = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)(x)
                return x

            downsample = _downsample

        if self.is_first:
            inplanes = x.shape[-1]
        else:
            inplanes = x.shape[-1] // expansion

        outplanes = int(inplanes) + self.add_rate
        x = self.block_fn(outplanes, self.stride, downsample, block_name=self.name + "_1")(x)
        outplanes += self.add_rate

        for i in range(1, self.n_layer):
            x = self.block_fn(outplanes, stride=1, block_name=self.name + "_{}".format(i + 2))(x)
            outplanes += self.add_rate

        return x


# Attention-insertable PyramidNet 18
def PyramidNet18WithAttention(include_top=True, input_shape=(256, 3), pooling=None, classes=6,
                              classifier_activation='softmax',
                              module: BaseAttention = None, alpha=48):
    """
    Parameters
    ----------
    include_top
    input_shape
    pooling
    classes
    classifier_activation
    module
    alpha


    Returns
    -------
    """
    block = BasicBlock
    layers = [2, 2, 2, 2]
    N = sum(layers)
    add_rate = int(alpha / N)

    x = inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv1D(64, kernel_size=7, strides=2, padding='same', use_bias=False, name='bottom_conv')(x)
    x = tf.keras.layers.BatchNormalization(name="bottom_conv_bn")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=4, strides=2, padding='same', name='bottom_pool')(x)

    # stack residual blocks
    x = Stack(block, layers[0], add_rate, stride=1, is_first=True, name="stack1")(x)
    if module is not None:
        x = module(x.shape[-1], block_name="stack1")(x)

    for i in range(1, len(layers)):
        x = Stack(block, layers[i], add_rate, stride=2, name="stack{}".format(i + 1))(x)
        if module is not None:
            x = module(x.shape[-1], block_name="stack{}".format(i + 1))(x)

    if include_top:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(classes, activation=classifier_activation, name='predictions')(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        return model
    else:
        if pooling is None:
            model_ = tf.keras.models.Model(inputs=inputs, outputs=x)
            return model_
        elif pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling1D(name="avgpool")(x)
            model_ = tf.keras.models.Model(inputs=inputs, outputs=x)
            return model_
        elif pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling1D(name="maxpool")(x)
            model_ = tf.keras.models.Model(inputs=inputs, outputs=x)
            return model_
        else:
            print("Not exist pooling option: {}".format(pooling))
            model_ = tf.keras.models.Model(inputs=inputs, outputs=x)
            return model_


# PyramidNet 18
def PyramidNet18(include_top=True, input_shape=(256, 3), pooling=None, classes=6,
                 classifier_activation='softmax', alpha=48):
    return PyramidNet18WithAttention(include_top, input_shape, pooling, classes, classifier_activation,
                                     module=None, alpha=alpha)
