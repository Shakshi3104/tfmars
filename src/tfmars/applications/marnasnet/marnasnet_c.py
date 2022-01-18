import tensorflow as tf
import math


from .blocks import ConvBlock, SkipOperation, RegularConvBlock, MBConvBlock
from ..mobile_inverted_bottleneck import Top, Stem


# MarNASNet-C
def MarNASNetC(width_coefficient=1.0, depth_coefficient=1.0, depth_divisor=8,
               include_top=True, input_shape=(256, 3), pooling=None, classes=6, classifier_activation='softmax',
               params=None):
    """
    Parameters
    ----------
    width_coefficient
    depth_coefficient
    depth_divisor
    include_top
    input_shape
    pooling
    classes
    classifier_activation
    params

    Returns
    -------
    """
    def round_filters(filters_, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters_ *= width_coefficient
        new_filters = max(divisor, int(filters_ + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters_:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats_):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats_))

    # se ratio
    se_ratio = 0.25

    # inputs
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Build stem
    filters_in = 32
    filters_in = round_filters(filters_in)
    outputs = Stem(filters_in)(inputs)

    if params is None:
        params = [
            {
                'conv_op': "MBConv",
                'kernel_size': 5,
                'skip_op': "identity",
                'layers': 2,
                'filters': 32
            },
            {
                'conv_op': "Conv",
                'kernel_size': 2,
                'skip_op': "identity",
                'layers': 4,
                'filters': 32
            },
            {
                'conv_op': "MBConv",
                'kernel_size': 2,
                'skip_op': "none",
                'layers': 2,
                'filters': 192
            },
            {
                'conv_op': "MBConv",
                'kernel_size': 5,
                'skip_op': "identity",
                'layers': 5,
                'filters': 192
            }
        ]

    for i, param in enumerate(params):
        block_id = i + 1

        conv_op = ConvBlock[param["conv_op"]]
        skip_op = SkipOperation[param["skip_op"]]

        repeats = round_repeats(param["layers"])
        kernel_size = param["kernel_size"]
        filters = round_filters(param["filters"])

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
    outputs = Top(round_filters(1280), classes, classifier_activation=classifier_activation,
                  include_top=include_top)(outputs)

    if not include_top:
        if pooling == "avg":
            outputs = tf.keras.layers.GlobalAveragePooling1D(name="avg_pool")(outputs)
        elif pooling == "max":
            outputs = tf.keras.layers.GlobalMaxPooling1D(name="max_pool")(outputs)

    # Create model
    model = tf.keras.models.Model(inputs, outputs)
    return model
