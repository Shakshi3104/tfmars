from tensorflow.keras import layers
from tensorflow.keras.models import Model
from ..modules.attention import BaseAttention


def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.)(x + 3.) * (1. / 6.)


def hard_swish(x):
    return layers.Multiply()([hard_sigmoid(x), x])


class SEBlock:
    def __init__(self, filters, se_ratio, prefix):
        self.filters = filters
        self.se_ratio = se_ratio
        self.prefix = prefix

    def __call__(self, x):
        inputs = x
        x = layers.GlobalAveragePooling1D(name=self.prefix + "squeeze_excite/AvgPool")(inputs)
        x = layers.Reshape((1, self.filters))(x)
        x = layers.Conv1D(
            _depth(self.filters * self.se_ratio),
            kernel_size=3,
            padding='same',
            name=self.prefix + "squeeze_excite/Conv"
        )(x)
        x = layers.ReLU(name=self.prefix + "squeeze_excite/Relu")(x)
        x = layers.Conv1D(
            self.filters,
            kernel_size=1,
            padding="same",
            name=self.prefix + 'squeeze_excite/Conv_1'
        )(x)
        x = hard_sigmoid(x)
        x = layers.Multiply(name=self.prefix + "squeeze_excite/Mul")([inputs, x])
        return x


class InvertedResBlock:
    def __init__(self, expansion, filters, kernel_size, stride, se_ratio, activation, block_id):
        self.expansion = expansion
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.se_ratio = se_ratio
        self.activation = activation
        self.block_id = block_id

    def __call__(self, x):
        shortcut = x
        prefix = 'expanded_conv/'
        infilters = x.shape[-1]

        if self.block_id:
            # Expand
            prefix = 'expanded_conv_{}/'.format(self.block_id)
            x = layers.Conv1D(
                _depth(infilters * self.expansion),
                kernel_size=1,
                padding='same',
                use_bias=False,
                name=prefix + 'expand'
            )(x)
            x = layers.BatchNormalization(
                epsilon=1e-3,
                momentum=0.999,
                name=prefix + 'expand/BatchNorm'
            )(x)
            x = self.activation(x)

        if self.stride == 2:
            x = layers.ZeroPadding1D(padding=1, name=prefix + 'depthwise/pad')(x)

        x = layers.SeparableConv1D(
            int(x.shape[-1]),
            self.kernel_size,
            strides=self.stride,
            padding='same' if self.stride == 1 else 'valid',
            use_bias=False,
            name=prefix + 'depthwise'
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + "depthwise/BatchNorm"
        )(x)
        x = self.activation(x)

        if self.se_ratio:
            x = SEBlock(_depth(infilters * self.expansion), self.se_ratio, prefix)(x)

        x = layers.Conv1D(
            self.filters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            name=prefix + 'project'
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + "project/BatchNorm"
        )(x)

        if self.stride == 1 and infilters == self.filters:
            x = layers.Add(name=prefix + 'Add')([shortcut, x])

        return x


def MobileNetV3(stack_fn, last_point_ch, include_top=True, input_shape=(256, 3), pooling=None, alpha=1.0, minimalistic=False,
                classes=6, dropout_rate=0.2, classifier_activation='softmax'):
    inputs = layers.Input(shape=input_shape)

    if minimalistic:
        kernel = 3
        activation = relu
        se_ratio = None
    else:
        kernel = 5
        activation = hard_swish
        se_ratio = 0.25

    x = inputs
    x = layers.Conv1D(
        16,
        kernel_size=3,
        strides=2,
        padding='same',
        use_bias=False,
        name='Conv'
    )(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='Conv/BN'
    )(x)
    x = activation(x)

    x = stack_fn(x, kernel, activation, se_ratio)

    last_conv_ch = _depth(x.shape[-1] * 6)

    if alpha > 1.0:
        last_conv_ch = _depth(last_conv_ch * alpha)
    x = layers.Conv1D(
        last_conv_ch,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name='Conv1'
    )(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='Conv_1/BN'
    )(x)
    x = activation(x)
    x = layers.Conv1D(
        last_point_ch,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name='Conv_2'
    )(x)
    x = activation(x)

    if include_top:
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Reshape((1, last_point_ch))(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv1D(classes, kernel_size=1, padding='same', name='Logits')(x)
        x = layers.Flatten()(x)
        x = layers.Activation(activation=classifier_activation, name='Predictions')(x)

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


# Attention-insertable MobileNetV3 Small
def MobileNetV3SmallWithAttention(include_top=True, input_shape=(256, 3), pooling=None, classes=6, classifier_activation='softmax',
                       module: BaseAttention=None, alpha=1.0, minimalistic=False):
    def stack_fn(x, kernel, activation, se_ratio):
        def depth(d):
            return _depth(d * alpha)

        x = InvertedResBlock(1, depth(16), 3, 2, se_ratio, relu, 0)(x)
        if module is not None:
            x = module(x.shape[-1], block_name="block0")(x)
        x = InvertedResBlock(72. / 16, depth(24), 3, 2, None, relu, 1)(x)
        if module is not None:
            x = module(x.shape[-1], block_name="block1")(x)
        x = InvertedResBlock(88. / 16, depth(24), 3, 1, None, relu, 2)(x)
        if module is not None:
            x = module(x.shape[-1], block_name="block2")(x)
        x = InvertedResBlock(4, depth(40), kernel, 2, se_ratio, activation, 3)(x)
        if module is not None:
            x = module(x.shape[-1], block_name="block3")(x)
        x = InvertedResBlock(6, depth(40), kernel, 1, se_ratio, activation, 4)(x)
        if module is not None:
            x = module(x.shape[-1], block_name="block4")(x)
        x = InvertedResBlock(6, depth(40), kernel, 1, se_ratio, activation, 5)(x)
        if module is not None:
            x = module(x.shape[-1], block_name="block5")(x)
        x = InvertedResBlock(3, depth(48), kernel, 1, se_ratio, activation, 6)(x)
        if module is not None:
            x = module(x.shape[-1], block_name="block6")(x)
        x = InvertedResBlock(3, depth(48), kernel, 1, se_ratio, activation, 7)(x)
        if module is not None:
            x = module(x.shape[-1], block_name="block7")(x)
        x = InvertedResBlock(6, depth(96), kernel, 2, se_ratio, activation, 8)(x)
        if module is not None:
            x = module(x.shape[-1], block_name="block8")(x)
        x = InvertedResBlock(6, depth(96), kernel, 1, se_ratio, activation, 9)(x)
        if module is not None:
            x = module(x.shape[-1], block_name="block9")(x)
        x = InvertedResBlock(6, depth(96), kernel, 1, se_ratio, activation, 10)(x)
        if module is not None:
            x = module(x.shape[-1], block_name="block10")(x)
        return x

    return MobileNetV3(stack_fn, 1024, include_top, input_shape, pooling, alpha, minimalistic, classes, 0.2,
                       classifier_activation)


# MobileNetV3 Small
def MobileNetV3(include_top=True, input_shape=(256, 3), pooling=None, classes=6, classifier_activation='softmax',
                alpha=1.0, minimalistic=False):
    return MobileNetV3SmallWithAttention(include_top, input_shape, pooling, classes, classifier_activation,
                                         module=None, alpha=alpha, minimalistic=minimalistic)
