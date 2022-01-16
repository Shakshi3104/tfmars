import tensorflow as tf
from ..modules.attention import BaseAttention, SqueezeAndExcite


# ConvBlock for VGG
class ConvBlock:
    def __init__(self, repeat, filters, kernel_size=3, strides=1, padding='same', activation='relu',
                 kernel_initializer='he_normal', pool_size=2, block_id=0):
        """
        ConvBlock for VGG
            repeat: the number of Conv1D
            filters: the number of filter of Conv1D
            kernel_size: the kernel_size of Conv1D, default `3`
            strides: the strides of Conv1D, default `1`
            padding: the padding of Conv1D and MaxPooling1D, default `'same'`
            activation: the activation function of Conv1D, default `'relu'`
            kernel_initializer: the kernel_initializer of Conv1D, default `'he_normal'`
            pool_size: the pool_size of MaxPooling1D, default `2`
        """
        self.repeat = repeat
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.pool_size = pool_size
        self.block_id = block_id

    def __call__(self, x):
        for i in range(0, self.repeat):
            x = tf.keras.layers.Conv1D(self.filters, kernel_size=self.kernel_size, strides=self.strides,
                                       padding=self.padding, activation=self.activation,
                                       kernel_initializer=self.kernel_initializer,
                                       name="block{}_{}".format(self.block_id, i+1))(x)

        x = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size, padding=self.padding,
                                         name="block{}_pool".format(self.block_id))(x)
        return x


# Attention-insertable VGG16
def VGG16WithAttention(include_top=True, input_shape=(256, 3), pooling=None, classes=6, classifier_activation='softmax',
                       module: BaseAttention=None):
    """VGG16 with Submodule
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
    model: tf.keras.models.Model
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = ConvBlock(2, 64, block_id=1)(inputs)
    if module is not None:
        x = module(64, block_name="block1")(x)
    x = ConvBlock(2, 128, block_id=2)(x)
    if module is not None:
        x = module(128, block_name="block2")(x)
    x = ConvBlock(3, 256, block_id=3)(x)
    if module is not None:
        x = module(256, block_name="block3")(x)
    x = ConvBlock(3, 512, block_id=4)(x)
    if module is not None:
        x = module(512, block_name="block4")(x)
    x = ConvBlock(3, 512, block_id=5)(x)
    if module is not None:
        x = module(512, block_name="block5")(x)

    if include_top:
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(4096, activation="relu", kernel_initializer="he_normal",
                                  name="fc1")(x)
        x = tf.keras.layers.Dense(4096, activation="relu", kernel_initializer="he_normal",
                                  name="fc2")(x)
        y = tf.keras.layers.Dense(classes, activation=classifier_activation, name="prediction")(x)

        model_ = tf.keras.models.Model(inputs=inputs, outputs=y)
        return model_
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


# VGG16
def VGG16(include_top=True, input_shape=(256, 3), pooling=None, classes=6, classifier_activation='softmax'):
    return VGG16WithAttention(include_top, input_shape, pooling, classes, classifier_activation, module=None)