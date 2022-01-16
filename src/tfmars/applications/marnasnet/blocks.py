import tensorflow as tf

from enum import Enum
from abc import *

from ..mobile_inverted_bottleneck import MBConv
from ...modules.attention import SqueezeAndExcite


# ConvOps enum
class ConvBlock(Enum):
    Conv = "Conv"
    SeparableConv = "SeparableConv"
    MBConv = "MBConv"
    ExtremeInception = "ExtremeInception"


# SkipOps enum
class SkipOperation(Enum):
    none = "none"
    pool = "pool"
    identity = "identity"


# Base conv block
class BaseBlock(metaclass=ABCMeta):
    def __init__(self, repeats, kernel_size, skip_op, strides,
                 se_ratio, block_id=1):
        self.repeats = repeats
        self.kernel_size = kernel_size
        self.skip_op = skip_op
        self.strides = strides
        self.se_ratio = se_ratio
        self.block_id = block_id

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError()


# Regular conv block
class RegularConvBlock(BaseBlock):
    def __init__(self, repeats: int, kernel_size: int, filters: int, skip_op: SkipOperation, strides: int,
                 se_ratio: float, block_id=1):
        """
        Parameters
        ----------
        repeats: int
            the number of convolutional layers
        kernel_size: int
            the dimension of the convolution window
        filters: int
            the number of filters
        skip_op: Blossom.options.SkipOperation
            skip operation
        strides: int
            the stride ot the convolution
        se_ratio: float
            between 0 and 1, fraction to squeeze the input filters
        block_id: int
            larger than 1, the block id
        """
        super().__init__(repeats, kernel_size, skip_op, strides, se_ratio, block_id)
        self.filters = filters

    def __call__(self, x):
        inputs = x

        for i in range(self.repeats):
            x = tf.keras.layers.Conv1D(
                self.filters,
                self.kernel_size,
                self.strides,
                padding='same',
                activation='relu',
                kernel_initializer='he_normal',
                name="block{}{}_conv".format(self.block_id, chr(i + 97))
            )(x)

        if 0 < self.se_ratio <= 1:
            x = SqueezeAndExcite(self.filters, self.se_ratio, block_name="block{}".format(self.block_id))(x)

        if self.skip_op == SkipOperation.pool:
            x = tf.keras.layers.MaxPooling1D(name="block{}_pool".format(self.block_id), padding='same')(x)
        elif self.skip_op == SkipOperation.identity:
            if self.strides == 1:
                shortcut = inputs
                if int(inputs.shape[-1]) != int(x.shape[-1]):
                    shortcut = tf.keras.layers.Conv1D(int(x.shape[-1]),
                                                      1,
                                                      strides=self.strides,
                                                      kernel_initializer="he_normal",
                                                      padding='valid',
                                                      name="block{}_shortcut".format(self.block_id))(x)

                x = tf.keras.layers.add([x, shortcut], name="block{}_add".format(self.block_id))

        return x


# Separable conv block
class SeparableConvBlock(BaseBlock):
    def __init__(self, repeats: int, kernel_size: int, skip_op: SkipOperation, strides: int, se_ratio: float,
                 block_id=1):
        """
        Parameters
        ----------
        repeats: int
            the number of convolutional layers
        kernel_size: int
            the dimension of the convolution window
        skip_op: Blossom.options.SkipOperation
            skip operation
        strides: int
            the stride ot the convolution
        se_ratio: float
            between 0 and 1, fraction to squeeze the input filters
        block_id: int
            larger than 1, the block id
        """
        super().__init__(repeats, kernel_size, skip_op, strides, se_ratio, block_id)

    def __call__(self, x):
        inputs = x
        filters = int(x.shape[-1])

        for i in range(self.repeats):
            x = tf.keras.layers.SeparableConv1D(
                filters,
                self.kernel_size,
                self.strides,
                padding='same',
                activation='relu',
                depthwise_initializer='he_normal',
                pointwise_initializer='he_normal',
                name="block{}{}_conv".format(self.block_id, chr(i + 97))
            )(x)

        if 0 < self.se_ratio <= 1:
            x = SqueezeAndExcite(filters, self.se_ratio, block_name="block_{}".format(self.block_id))(x)

        if self.skip_op == SkipOperation.pool:
            x = tf.keras.layers.MaxPooling1D(name="block{}_pool".format(self.block_id), padding='same')(x)
        elif self.skip_op == SkipOperation.identity:
            if self.strides == 1:
                shortcut = inputs
                if int(inputs.shape[-1]) != int(x.shape[-1]):
                    shortcut = tf.keras.layers.Conv1D(int(x.shape[-1]),
                                                      1,
                                                      strides=self.strides,
                                                      kernel_initializer="he_normal",
                                                      padding='valid',
                                                      name="block{}_shortcut".format(self.block_id))(x)

                x = tf.keras.layers.add([x, shortcut], name="block{}_add".format(self.block_id))

        return x


# MBConv block
class MBConvBlock(BaseBlock):
    def __init__(self, repeats, kernel_size, filters_in, filters_out, expand_ratio, skip_op, strides, se_ratio,
                 block_id=1):
        """
        Parameters
        ----------
        repeats: int
            the number of convolutional layers
        kernel_size: int
            the dimension of the convolution window
        filters_in: int
            the number of input filters
        filters_out: int
            the number of output filters
        expand_ratio: int
            scaling coefficient for the input filters
        skip_op: Blossom.options.SkipOperation
            skip operation
        strides: int
            the stride ot the convolution
        se_ratio: float
            between 0 and 1, fraction to squeeze the input filters
        block_id: int
            larger than 1, the block id
        """
        super().__init__(repeats, kernel_size, skip_op, strides, se_ratio, block_id)
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.expand_ratio = expand_ratio

    def __call__(self, x):
        for i in range(self.repeats):
            x = MBConv(
                drop_rate=0.2,
                kernel_size=self.kernel_size,
                filters_in=self.filters_in,
                filters_out=self.filters_out,
                strides=self.strides if i == 0 else 1,
                expand_ratio=self.expand_ratio,
                id_skip=True if self.skip_op == SkipOperation.identity else False,
                se_ratio=self.se_ratio,
                name="block{}{}_".format(self.block_id, chr(i + 97))
            )(x)

        if self.skip_op == SkipOperation.pool:
            x = tf.keras.layers.MaxPooling1D(name="block{}_pool".format(self.block_id), padding='same')(x)

        return x