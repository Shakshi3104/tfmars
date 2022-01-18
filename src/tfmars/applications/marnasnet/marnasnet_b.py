import tensorflow as tf
import math


from .blocks import ConvBlock, SkipOperation, RegularConvBlock, MBConvBlock
from ..mobile_inverted_bottleneck import Top, Stem


# MarNASNet-B
def MarNASNetB(include_top=True, input_shape=(256, 3), pooling=None, classes=6, classifier_activation='softmax',
               params=None):
    """
    Parameters
    ----------
    include_top
    input_shape
    pooling
    classes
    classifier_activation
    params

    Returns
    -------
    """

    # se ratio
    se_ratio = 0.25

    # inputs
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Build stem
    filters_in = 32
    outputs = Stem(filters_in)(inputs)

    if params is None:
        params = [
            {
                'conv_op': "MBConv",
                'kernel_size': 3,
                'skip_op': "none",
                'layers': 2,
                'filters': 32
            },
            {
                'conv_op': "MBConv",
                'kernel_size': 5,
                'skip_op': "identity",
                'layers': 5,
                'filters': 32
            },
            {
                'conv_op': "MBConv",
                'kernel_size': 5,
                'skip_op': "identity",
                'layers': 2,
                'filters': 32
            },
            {
                'conv_op': "MBConv",
                'kernel_size': 5,
                'skip_op': "identity",
                'layers': 5,
                'filters': 32
            }
        ]

    for i, param in enumerate(params):
        block_id = i + 1

        conv_op = ConvBlock[param["conv_op"]]
        skip_op = SkipOperation[param["skip_op"]]

        repeats = param["layers"]
        kernel_size = param["kernel_size"]
        filters = param["filters"]

        if conv_op == ConvBlock.Conv:
            outputs = RegularConvBlock(
                repeats=repeats,
                kernel_size=kernel_size,
                filters=filters,
                skip_op=skip_op,
                strides=1,
                se_ratio=se_ratio,
                block_id=block_id
            )(outputs)

        elif conv_op == ConvBlock.MBConv:
            outputs = MBConvBlock(
                repeats=repeats,
                kernel_size=kernel_size,
                filters_in=filters_in,
                filters_out=filters,
                expand_ratio=1,
                skip_op=skip_op,
                strides=1 if i == 0 else 2,
                se_ratio=se_ratio,
                block_id=block_id
            )(outputs)

        filters_in = filters

    # Build top
    outputs = Top(1280, classes, classifier_activation=classifier_activation,
                  include_top=include_top)(outputs)

    if not include_top:
        if pooling == "avg":
            outputs = tf.keras.layers.GlobalAveragePooling1D(name="avg_pool")(outputs)
        elif pooling == "max":
            outputs = tf.keras.layers.GlobalMaxPooling1D(name="max_pool")(outputs)

    # Create model
    model = tf.keras.models.Model(inputs, outputs)
    return model
