from tensorflow.keras import layers
from tensorflow.keras.models import Model
from ..modules.attention import BaseAttention


class ConvBlock:
    def __init__(self, growth_rate, name):
        self.growth_rate = growth_rate
        self.name = name

    def __call__(self, x):
        x1 = layers.BatchNormalization(
            epsilon=1.001e-5, name=self.name + "_0_bn"
        )(x)
        x1 = layers.Activation('relu', name=self.name + "_0_relu")(x1)
        x1 = layers.Conv1D(4 * self.growth_rate, 1, use_bias=False, name=self.name + "_1_conv")(x1)
        x1 = layers.BatchNormalization(epsilon=1.001e-5, name=self.name + "_1_bn")(x1)
        x1 = layers.Activation('relu', name=self.name + "_1_relu")(x1)
        x1 = layers.Conv1D(self.growth_rate, 3, padding='same', use_bias=False, name=self.name + "_2_conv")(x1)
        x = layers.Concatenate(name=self.name + "_concat")([x, x1])
        return x


class TransitionBlock:
    def __init__(self, reduction, name):
        self.reduction = reduction
        self.name = name

    def __call__(self, x):
        x = layers.BatchNormalization(epsilon=1.001e-5, name=self.name + "_bn")(x)
        x = layers.Activation('relu', name=self.name + "_relu")(x)
        x = layers.Conv1D(int(x.shape[-1] * self.reduction),
                          1,
                          use_bias=False,
                          name=self.name + "_conv")(x)
        x = layers.AveragePooling1D(2, strides=2, name=self.name + "_pool")(x)
        return x


class DenseBlock:
    def __init__(self, blocks, name):
        self.blocks = blocks
        self.name = name

    def __call__(self, x):
        for i in range(self.blocks):
            x = ConvBlock(32, name=self.name + "_block" + str(i + 1))(x)
        return x


# Attention-insertable DenseNet 121
def DenseNet121WithAttention(include_top=True, input_shape=(256, 3), pooling=None, classes=6,
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
    blocks = [6, 12, 24, 16]

    inputs = layers.Input(shape=input_shape)

    x = layers.ZeroPadding1D(padding=(3, 3))(inputs)
    x = layers.Conv1D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding1D(padding=(1, 1))(x)
    x = layers.MaxPooling1D(3, strides=2, name='pool1')(x)

    if module is not None:
        x = module(x.shape[-1], block_name="conv1")(x)

    x = DenseBlock(blocks[0], name='conv2')(x)
    x = TransitionBlock(0.5, name='pool2')(x)
    if module is not None:
        x = module(x.shape[-1], block_name="conv2")(x)

    x = DenseBlock(blocks[1], name='conv3')(x)
    x = TransitionBlock(0.5, name='pool3')(x)
    if module is not None:
        x = module(x.shape[-1], block_name="conv3")(x)
    x = DenseBlock(blocks[2], name='conv4')(x)
    x = TransitionBlock(0.5, name='pool4')(x)
    if module is not None:
        x = module(x.shape[-1], block_name="conv4")(x)
    x = DenseBlock(blocks[3], name='conv5')(x)

    x = layers.BatchNormalization(epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)
    if module is not None:
        x = module(x.shape[-1], block_name="conv5")(x)

    if include_top:
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


# DenseNet 121
def DenseNet121(include_top=True, input_shape=(256, 3), pooling=None, classes=6, classifier_activation='softmax'):
    return DenseNet121WithAttention(include_top, input_shape, pooling, classes, classifier_activation, module=None)
