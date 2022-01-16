from tensorflow.keras import layers
from tensorflow.keras.models import Model
from ..modules.attention import BaseAttention


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBlock:
    def __init__(self, strides, filters, kernel=3):
        """
        Adds an initial convolution layer (with batch normalization and relu6).
        """
        self.strides = strides
        self.filters = filters
        self.kernel = kernel

    def __call__(self, x):
        x = layers.Conv1D(
            self.filters,
            self.kernel,
            padding='same',
            use_bias=False,
            strides=self.strides,
            name='Conv1')(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv1_bn')(x)
        x = layers.ReLU(6., name='Conv1_relu')(x)
        return x


class SepConvBlock:
    def __init__(self, filters, alpha, pointwise_conv_filters, depth_multiplier=1, strides=1):
        """
        Adds an separable convolution block
        """
        self.filters = filters
        self.alpha = alpha
        self.pointwise_conv_filters = (pointwise_conv_filters * alpha)
        self.depth_multiplier = depth_multiplier
        self.strides = strides

    def __call__(self, x):

        x = layers.SeparableConv1D(
            self.pointwise_conv_filters,
            kernel_size=3,
            padding='same',
            depth_multiplier=self.depth_multiplier,
            strides=self.strides,
            use_bias=False,
            name="Conv_sep"
        )(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_sep_bn')(x)
        x = layers.ReLU(6., name='Conv_sep_relu')(x)
        return x


class InvertedResBlock:
    def __init__(self, kernel, expansion, alpha, filters, block_id, stride=1):
        self.kernel = kernel
        self.expansion = expansion
        self.alpha = alpha
        self.filters = filters
        self.block_id = block_id
        self.stride = stride

    def __call__(self, x):
        in_channels = x.shape[-1]
        pointwise_conv_filters = int(self.filters * self.alpha)
        pointwise_filters = _make_divisible(pointwise_conv_filters, 8)

        inputs = x
        prefix = 'block_{}_'.format(self.block_id)

        if self.block_id:
            x = layers.Conv1D(
                self.expansion * in_channels,
                kernel_size=1,
                padding='same',
                use_bias=False,
                activation=None,
                name=prefix + "expand"
            )(x)
            x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + "expand_bn")(x)
            x = layers.ReLU(6., name=prefix + "expand_relu")(x)
        else:
            prefix = 'expanded_conv_'

        x = layers.SeparableConv1D(
            int(x.shape[-1]),
            kernel_size=self.kernel,
            strides=self.stride,
            activation=None,
            use_bias=False,
            padding='same',
            name=prefix + 'depthwise'
        )(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_bn')(x)
        x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

        x = layers.Conv1D(
            pointwise_filters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None,
            name=prefix + "project"
        )(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_bn')(x)

        if in_channels == pointwise_filters and self.stride == 1:
            x = layers.Add(name=prefix + 'add')([inputs, x])
        return x


# Attention-insertable MnasNet
def MnasNetWithAttention(include_top=True, input_shape=(256, 3), pooling=None, classes=6, classifier_activation='softmax',
                         module: BaseAttention = None, alpha=1.0, depth_multiplier=2):
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
    depth_multiplier
    Returns
    -------
    """
    inputs = layers.Input(shape=input_shape)

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = ConvBlock(2, first_block_filters)(inputs)
    if module is not None:
        x = module(x.shape[-1], block_name="conv")(x)

    x = SepConvBlock(16, alpha, 16, depth_multiplier)(x)
    if module is not None:
        x = module(x.shape[-1], block_name="sepconv")(x)

    x = InvertedResBlock(kernel=3, expansion=3, stride=2, alpha=alpha, filters=24, block_id=1)(x)
    if module is not None:
        x = module(x.shape[-1], block_name="block1")(x)
    x = InvertedResBlock(kernel=3, expansion=3, stride=1, alpha=alpha, filters=24, block_id=2)(x)
    if module is not None:
        x = module(x.shape[-1], block_name="block2")(x)
    x = InvertedResBlock(kernel=3, expansion=3, stride=1, alpha=alpha, filters=24, block_id=3)(x)
    if module is not None:
        x = module(x.shape[-1], block_name="block3")(x)

    x = InvertedResBlock(kernel=5, expansion=3, stride=2, alpha=alpha, filters=40, block_id=4)(x)
    if module is not None:
        x = module(x.shape[-1], block_name="block4")(x)
    x = InvertedResBlock(kernel=5, expansion=3, stride=1, alpha=alpha, filters=40, block_id=5)(x)
    if module is not None:
        x = module(x.shape[-1], block_name="block5")(x)
    x = InvertedResBlock(kernel=5, expansion=3, stride=1, alpha=alpha, filters=40, block_id=6)(x)
    if module is not None:
        x = module(x.shape[-1], block_name="block6")(x)

    x = InvertedResBlock(kernel=5, expansion=6, stride=2, alpha=alpha, filters=80, block_id=7)(x)
    if module is not None:
        x = module(x.shape[-1], block_name="block7")(x)
    x = InvertedResBlock(kernel=5, expansion=6, stride=1, alpha=alpha, filters=80, block_id=8)(x)
    if module is not None:
        x = module(x.shape[-1], block_name="block8")(x)
    x = InvertedResBlock(kernel=5, expansion=6, stride=1, alpha=alpha, filters=80, block_id=9)(x)
    if module is not None:
        x = module(x.shape[-1], block_name="block9")(x)

    x = InvertedResBlock(kernel=3, expansion=6, stride=1, alpha=alpha, filters=96, block_id=10)(x)
    if module is not None:
        x = module(x.shape[-1], block_name="block10")(x)
    x = InvertedResBlock(kernel=3, expansion=6, stride=1, alpha=alpha, filters=96, block_id=11)(x)
    if module is not None:
        x = module(x.shape[-1], block_name="block11")(x)

    x = InvertedResBlock(kernel=5, expansion=6, stride=2, alpha=alpha, filters=192, block_id=12)(x)
    if module is not None:
        x = module(x.shape[-1], block_name="block12")(x)
    x = InvertedResBlock(kernel=5, expansion=6, stride=1, alpha=alpha, filters=192, block_id=13)(x)
    if module is not None:
        x = module(x.shape[-1], block_name="block13")(x)
    x = InvertedResBlock(kernel=5, expansion=6, stride=1, alpha=alpha, filters=192, block_id=14)(x)
    if module is not None:
        x = module(x.shape[-1], block_name="block14")(x)
    x = InvertedResBlock(kernel=5, expansion=6, stride=1, alpha=alpha, filters=192, block_id=15)(x)
    if module is not None:
        x = module(x.shape[-1], block_name="block15")(x)

    x = InvertedResBlock(kernel=3, expansion=6, stride=1, alpha=alpha, filters=320, block_id=16)(x)
    if module is not None:
        x = module(x.shape[-1], block_name="block16")(x)

    if include_top:
        x = layers.GlobalAveragePooling1D()(x)
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


# MnasNet
def MnasNet(include_top=True, input_shape=(256, 3), pooling=None, classes=6, classifier_activation='softmax',
            alpha=1.0, depth_multiplier=2):
    return MnasNetWithAttention(include_top, input_shape, pooling, classes, classifier_activation,
                                module=None, alpha=alpha, depth_multiplier=depth_multiplier)