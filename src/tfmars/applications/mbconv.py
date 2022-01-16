import tensorflow as tf


class MBConv:
    def __init__(self, activation="relu", drop_rate=0., kernel_size=3, filters_in=32, filters_out=16, strides=1,
                 expand_ratio=1, se_ratio=0., id_skip=True, name=""):
        """
        Mobile inverted Bottleneck Convolutional layer
        Parameters
        ----------
        activation:
            activation function
        drop_rate: float
            between 0 and 1, fraction of the input units to drop
        kernel_size: integer
            the dimension of the convolution window
        filters_in: integer
            the number of input filters
        filters_out: integer
            the number of output filters
        strides: integer
            the stride of the convolution
        expand_ratio: integer
            scaling coefficient for the input filters
        se_ratio: float
            between 0 and 1, fraction to squeeze the input filters
        id_skip: boolean
        name: string
            block label
        """
        self.name = name
        self.id_skip = id_skip
        self.se_ratio = se_ratio
        self.expand_ratio = expand_ratio
        self.strides = strides
        self.filters_out = filters_out
        self.filters_in = filters_in
        self.kernel_size = kernel_size
        self.drop_rate = drop_rate
        self.activation = activation

    def __call__(self, x):
        inputs = x
        # Expansion phase
        filters = self.filters_in * self.expand_ratio
        if self.expand_ratio != 1:
            x = tf.keras.layers.Conv1D(
                filters,
                1,
                padding='same',
                use_bias=False,
                kernel_initializer="he_normal",
                name=self.name + "expand_conv"
            )(x)
            x = tf.keras.layers.BatchNormalization(name=self.name + "expand_bn")(x)
            x = tf.keras.layers.Activation(self.activation, name=self.name + "expand_activation")(x)
        else:
            x = inputs

        # Depthwise Convolution
        conv_pad = 'same'
        x = tf.keras.layers.SeparableConv1D(
            int(x.shape[-1]) if self.expand_ratio != 1 else self.filters_in,
            self.kernel_size,
            self.strides,
            padding=conv_pad,
            use_bias=False,
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
            name=self.name + "dwconv"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=self.name + "bn")(x)
        x = tf.keras.layers.Activation(self.activation, name=self.name + "activation")(x)

        # Squeeze and excitation phase
        if 0 < self.se_ratio <= 1:
            x = SqueezeAndExcite(self.filters_in, se_ratio=self.se_ratio, block_name=self.name)(x)

        # Output phase
        x = tf.keras.layers.Conv1D(
            self.filters_out,
            1,
            padding='same',
            use_bias=False,
            kernel_initializer="he_normal",
            name=self.name + "project_conv"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=self.name + "project_bn")(x)

        if self.id_skip and self.strides == 1 and self.filters_in == self.filters_out:
            if self.drop_rate > 0:
                x = tf.keras.layers.Dropout(self.drop_rate, name=self.name + "drop")(x)
            x = tf.keras.layers.add([x, inputs], name=self.name + "add")

        return x


# Squeeze-and-Excitation module
class SqueezeAndExcite:
    """squeeze-and-excitation module
    """
    def __init__(self, filters, se_ratio=0.25, block_name=""):
        """
        Parameters
        ----------
        filters: int
            output filter size
        se_ratio: float
            se ratio, se_ratio must be greater than 0 and less than or equal to 1.
        block_name: str
            block name
        """
        self.filters = filters
        self.block_name = block_name
        self.se_ratio = se_ratio

    def __call__(self, x):
        assert 0 < self.se_ratio <= 1, "se_ratio must be greater than 0 and less than or equal to 1."

        filters_se = max(1, int(self.filters * self.se_ratio))
        se = tf.keras.layers.GlobalAveragePooling1D(name="{}_se_squeeze".format(self.block_name))(x)
        se = tf.keras.layers.Reshape((1, self.filters), name="{}_se_reshape".format(self.block_name))(se)
        se = tf.keras.layers.Conv1D(
            filters_se,
            1,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name="{}_se_reduce".format(self.block_name)
        )(se)
        se = tf.keras.layers.Conv1D(
            self.filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer="he_normal",
            name="{}_se_expand".format(self.block_name)
        )(se)
        x = tf.keras.layers.multiply([x, se], name="{}_se_excite".format(self.block_name))
        return x


# Stem of MarNASNet and EfficientNet
class Stem:
    def __init__(self, filters):
        self.filters = filters

    def __call__(self, x):
        outputs = tf.keras.layers.Conv1D(self.filters, 3, strides=2, padding='same', use_bias=False,
                                         kernel_initializer="he_normal", name="stem_conv")(x)
        outputs = tf.keras.layers.BatchNormalization(name='stem_bn')(outputs)
        outputs = tf.keras.layers.Activation(activation='relu', name='stem_activation')(outputs)
        return outputs


# Top of MarNASNet and EfficientNet
class Top:
    def __init__(self, filters_top_conv, classes, dropout_rate=0.2, classifier_activation="softmax", include_top=True):
        self.filters_top_conv = filters_top_conv
        self.classes = classes
        self.dropout_rate = dropout_rate
        self.classifier_activation = classifier_activation
        self.include_top = include_top

    def __call__(self, x):
        outputs = tf.keras.layers.Conv1D(self.filters_top_conv, 1, padding='same', use_bias=False,
                                         kernel_initializer='he_normal', name='top_conv')(x)
        outputs = tf.keras.layers.BatchNormalization(name='top_bn')(outputs)
        outputs = tf.keras.layers.Activation(activation='relu', name='top_activation')(outputs)

        if self.include_top:
            outputs = tf.keras.layers.GlobalAveragePooling1D(name='avg_pool')(outputs)
            outputs = tf.keras.layers.Dropout(self.dropout_rate)(outputs)
            outputs = tf.keras.layers.Dense(self.classes, activation=self.classifier_activation,
                                            name='predictions')(outputs)

        return outputs
