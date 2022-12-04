import tensorflow as tf
from ..modules.attention import BaseAttention


class Shortcut:
    def __init__(self, kernel_initializer='he_normal', block_name=""):
        self.kernel_initializer = kernel_initializer
        self.block_name = block_name

    def __call__(self, x):
        inputs, residual = x

        stride = int(inputs.shape[1]) / int(residual.shape[1])
        equal_channels = int(residual.shape[2]) == int(inputs.shape[2])

        shortcut = inputs
        if stride > 1 or not equal_channels:
            shortcut = tf.keras.layers.Conv1D(int(residual.shape[2]), 1, strides=int(stride),
                                              kernel_initializer=self.kernel_initializer,
                                              padding='valid', name='{}_shortcut'.format(self.block_name))(inputs)

        return tf.keras.layers.add([shortcut, residual])


class BasicBlock:
    def __init__(self, nb_fil, strides, activation='relu', kernel_initializer='he_normal', block_name=""):
        self.nb_fil = nb_fil
        self.strides = strides

        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.block_name = block_name

    def __call__(self, x):
        inputs = x

        x = tf.keras.layers.BatchNormalization(name="{}_basic_bn_1".format(self.block_name))(x)
        x = tf.keras.layers.Activation(self.activation, name="{}_basic_act_1".format(self.block_name))(x)
        x = tf.keras.layers.Conv1D(self.nb_fil, 3, strides=1, kernel_initializer=self.kernel_initializer,
                                   padding='same', name="{}_basic_conv_1".format(self.block_name))(x)
        x = tf.keras.layers.BatchNormalization(name="{}_basic_bn_2".format(self.block_name))(x)
        x = tf.keras.layers.Activation(self.activation, name="{}_basic_act_2".format(self.block_name))(x)
        x = tf.keras.layers.Conv1D(self.nb_fil, 3, strides=1, kernel_initializer=self.kernel_initializer,
                                   padding='same', name="{}_basic_conv_2".format(self.block_name))(x)

        x = Shortcut(kernel_initializer=self.kernel_initializer, block_name=self.block_name)([inputs, x])
        return x


class Bottleneck:
    def __init__(self, nb_fil, strides, activation='relu', kernel_initializer='he_normal', block_name=""):
        self.nb_fil = nb_fil
        self.strides = strides

        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.block_name = block_name

    def __call__(self, x):
        inputs = x

        x = tf.keras.layers.BatchNormalization(name="{}_bottleneck_bn_1".format(self.block_name))(x)
        x = tf.keras.layers.Activation(self.activation, name="{}_bottleneck_act_1".format(self.block_name))(x)
        x = tf.keras.layers.Conv1D(int(self.nb_fil / 4), 1, strides=self.strides,
                                   kernel_initializer=self.kernel_initializer, padding='same',
                                   name="{}_bottleneck_conv_1".format(self.block_name))(x)

        x = tf.keras.layers.BatchNormalization(name="{}_bottleneck_bn_2".format(self.block_name))(x)
        x = tf.keras.layers.Activation(self.activation, name="{}_bottleneck_act_2".format(self.block_name))(x)
        x = tf.keras.layers.Conv1D(int(self.nb_fil / 4), 3, strides=self.strides,
                                   kernel_initializer=self.kernel_initializer, padding="same",
                                   name="{}_bottleneck_conv_2".format(self.block_name))(x)

        x = tf.keras.layers.BatchNormalization(name="{}_bottleneck_bn_3".format(self.block_name))(x)
        x = tf.keras.layers.Activation(self.activation, name="{}_bottleneck_act_3".format(self.block_name))(x)
        x = tf.keras.layers.Conv1D(self.nb_fil, 1, strides=self.strides, kernel_initializer=self.kernel_initializer,
                                   padding="same", name="{}_bottleneck_conv_3".format(self.block_name))(x)

        x = Shortcut(kernel_initializer=self.kernel_initializer, block_name=self.block_name)([inputs, x])
        return x


class ResidualBlock:
    def __init__(self, nb_fil, repeats, block, is_first=False, activation='relu', kernel_initializer='he_normal',
                 block_name=""):
        self.nb_fil = nb_fil
        self.repeats = repeats
        self.is_first = is_first
        self.block = block

        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.block_name = block_name

    def __call__(self, x):
        if not self.is_first:
            x = self.block(self.nb_fil, 2, self.activation, self.kernel_initializer, block_name=self.block_name + "_1")(
                x)
        else:
            x = self.block(self.nb_fil, 1, self.activation, self.kernel_initializer, block_name=self.block_name + "_1")(
                x)

        for i in range(1, self.repeats):
            x = self.block(self.nb_fil, 1, self.activation, self.kernel_initializer,
                           block_name=self.block_name + "_{}".format(i + 1))(x)

        return x


# Attention-insertable ResNet 18
def ResNet18WithAttention(include_top=True, input_shape=(256, 3), pooling=None, classes=6,
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

    layers = [64, 128, 256, 512]
    repeats = [2, 2, 2, 2]

    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv1D(32, 7, strides=2, padding='same', kernel_initializer="he_normal", name="bottom_conv")(
        inputs)
    x = tf.keras.layers.BatchNormalization(name="bottom_conv_bn")(x)
    x = tf.keras.layers.Activation('relu', name="bottom_conv_relu")(x)

    x = ResidualBlock(layers[0], repeats=repeats[0], block=BasicBlock, is_first=True, block_name="res1")(x)
    if module is not None:
        x = module(layers[0], block_name="res1")(x)

    for i, (layer, repeat) in enumerate(zip(layers[1:], repeats[1:])):
        x = ResidualBlock(layer, repeat, BasicBlock, block_name="res{}".format(i + 2))(x)
        if module is not None:
            x = module(layer, block_name="res{}".format(i + 2))(x)

    if include_top:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        y = tf.keras.layers.Dense(classes, activation=classifier_activation, name="predictions")(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=y)
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


# ResNet 18
def ResNet18(include_top=True, input_shape=(256, 3), pooling=None, classes=6, classifier_activation='softmax'):
    return ResNet18WithAttention(include_top, input_shape, pooling, classes, classifier_activation, module=None)
