import tensorflow as tf
import abc


class BaseAttention(metaclass=abc.ABCMeta):
    def __init__(self, filters, block_name):
        self.filters = filters
        self.block_name = block_name

    def __call__(self, x):
        return x


# Squeeze-and-Excitation module
class SqueezeAndExcite(BaseAttention):
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
        super().__init__(filters, block_name)
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


# selective kernel module
class SelectiveKernel(BaseAttention):
    """selective kernel module
    """
    def __init__(self, filters, block_name=""):
        """selective kernel module
        Parameters
        ----------
        filters: int
            output filter size
        block_name: str
            block name
        """
        super(SelectiveKernel, self).__init__(filters, block_name)

    def __call__(self, x):
        L = 32
        reduction = 4

        conv1_1 = tf.keras.layers.Conv1D(L, kernel_size=3, strides=1, activation='relu', padding='same',
                                         name=self.block_name + "_sk_conv1")(x)
        conv1 = tf.keras.layers.BatchNormalization(axis=2, name=self.block_name + "_sk_conv1_bn")(conv1_1)

        conv2_1 = tf.keras.layers.Conv1D(L, kernel_size=5, strides=1, activation='relu', padding='same',
                                         name=self.block_name + "_sk_conv2")(x)
        conv2 = tf.keras.layers.BatchNormalization(axis=2, name=self.block_name + "_sk_conv2_bn")(conv2_1)

        conv_add = tf.keras.layers.Add(name=self.block_name + "_sk_conv_add")([conv1, conv2])
        avg_x = tf.keras.layers.GlobalAveragePooling1D(name=self.block_name + "_sk_avgpool")(conv_add)

        channels = conv_add.shape[-1]
        d = channels // reduction
        x = tf.keras.layers.Dense(d, activation='softmax', name=self.block_name + "_sk_dense1")(avg_x)
        x = tf.keras.layers.Dense(channels * 2, activation='softmax', name=self.block_name + "_sk_dense2")(x)
        x = tf.keras.layers.Activation(activation='softmax', name=self.block_name + "_sk_softmax1")(x)

        # the part of selection
        a, b = tf.keras.layers.Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 2})(x)

        a = tf.keras.layers.Activation(activation='softmax', name=self.block_name + '_sk_softmax_a')(a)
        a = tf.keras.layers.Reshape((1, channels), name=self.block_name + '_sk_reshape_a')(a)

        b = tf.keras.layers.Activation(activation='softmax', name=self.block_name + '_sk_softmax_b')(b)
        b = tf.keras.layers.Reshape((1, channels), name=self.block_name + '_sk_reshape_b')(b)

        a = tf.keras.layers.Multiply(name=self.block_name + "_sk_excite_a")([conv1, a])
        b = tf.keras.layers.Multiply(name=self.block_name + "_sk_excite_b")([conv2, b])

        x = tf.keras.layers.Add(name=self.block_name + "_add_ab")([a, b])

        # When input_shape is not equal output_shape
        if x.shape[-1] != self.filters:
            x = tf.keras.layers.Conv1D(self.filters,
                                       1,
                                       padding='same',
                                       use_bias=False,
                                       kernel_initializer='he_normal',
                                       name=self.block_name + "_expand_conv")(x)
            x = tf.keras.layers.BatchNormalization(name=self.block_name + "_expand_bn")(x)

        return x


class ChannelAttention(BaseAttention):
    def __init__(self, filters, ratio=0.125, block_name=""):
        super(ChannelAttention, self).__init__(filters, block_name)

        self.ratio = ratio

    def __call__(self, x):
        inputs = x
        shared_1 = tf.keras.layers.Dense(int(self.filters * self.ratio), activation='relu', kernel_initializer='he_normal',
                                         name=self.block_name + "_ca_shared_1")
        shared_2 = tf.keras.layers.Dense(self.filters, activation='relu', kernel_initializer='he_normal',
                                         name=self.block_name + "_ca_shared_2")

        avg_pool = tf.keras.layers.GlobalAveragePooling1D(name=self.block_name + "_ca_avgpool")(x)
        avg_pool = tf.keras.layers.Reshape((1, self.filters), name=self.block_name + "_ca_reshape_avgpool")(avg_pool)
        avg_pool = shared_1(avg_pool)
        assert avg_pool.shape[1:] == (1, int(self.filters * self.ratio))
        avg_pool = shared_2(avg_pool)
        assert avg_pool.shape[1:] == (1, self.filters)

        max_pool = tf.keras.layers.GlobalMaxPooling1D(name=self.block_name + "_ca_maxpool")(x)
        max_pool = tf.keras.layers.Reshape((1, self.filters), name=self.block_name + "_ca_reshape_maxpool")(max_pool)
        max_pool = shared_1(max_pool)
        assert max_pool.shape[1:] == (1, int(self.filters * self.ratio))
        max_pool = shared_2(max_pool)
        assert max_pool.shape[1:] == (1, self.filters)

        ca = tf.keras.layers.Add(name=self.block_name + "_ca_add")([avg_pool, max_pool])
        ca = tf.keras.layers.Activation(activation='sigmoid', name=self.block_name + "_ca_sigmoid")(ca)
        x = tf.keras.layers.Multiply(name=self.block_name + "_ca_excite")([inputs, ca])
        return x


class SpatialAttention:
    def __init__(self, block_name=""):
        self.block_name = block_name

    def __call__(self, x):
        inputs = x

        max_pool = tf.keras.layers.MaxPooling1D(strides=1, padding='same', name=self.block_name + '_sa_maxpool')(x)
        avg_pool = tf.keras.layers.AveragePooling1D(strides=1, padding='same', name=self.block_name + '_sa_avgpool')(x)

        concat = tf.keras.layers.concatenate([avg_pool, max_pool], axis=-1)
        sa = tf.keras.layers.Conv1D(1, kernel_size=7, strides=1, activation='sigmoid', padding='same',
                                    name=self.block_name + '_sa_conv')(concat)
        x = tf.keras.layers.Multiply(name=self.block_name + "_sa_excite")([inputs, sa])
        return x


# conv block attention module
class ConvBlockAttention(BaseAttention):
    """Convolution Block Attention Module
    """
    def __init__(self, filters, ratio=0.125, block_name=""):
        """convolutional block attention module
        Parameters
        ----------
        filters: int
            output filter size
        ratio: float
            ratio, ratio must be greater than 0 and less than or equal to 1.
        block_name: str
            block name
        """
        super(ConvBlockAttention, self).__init__(filters, block_name)

        self.ratio = ratio

    def __call__(self, x):
        x = ChannelAttention(self.filters, ratio=self.ratio, block_name=self.block_name)(x)
        x = SpatialAttention(self.block_name)(x)
        return x
