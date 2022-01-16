from .efficientnet import EfficientNetWithAttention
from ..modules.attention import BaseAttention


# Attention-insertable EfficientNet lite0
def EfficientNetLite0WithAttention(include_top=True, input_shape=(256, 3), pooling=None, classes=6,
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
    blocks_args = [{
        'kernel_size': 3,
        'repeats': 1,
        'filters_in': 32,
        'filters_out': 16,
        'expand_ratio': 1,
        'id_skip': True,
        'strides': 1,
        'se_ratio': 0.0
    }, {
        'kernel_size': 3,
        'repeats': 2,
        'filters_in': 16,
        'filters_out': 24,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 2,
        'se_ratio': 0.0
    }, {
        'kernel_size': 5,
        'repeats': 2,
        'filters_in': 24,
        'filters_out': 40,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 2,
        'se_ratio': 0.0
    }, {
        'kernel_size': 3,
        'repeats': 3,
        'filters_in': 40,
        'filters_out': 80,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 2,
        'se_ratio': 0.0
    }, {
        'kernel_size': 5,
        'repeats': 3,
        'filters_in': 80,
        'filters_out': 112,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 1,
        'se_ratio': 0.0
    }, {
        'kernel_size': 5,
        'repeats': 4,
        'filters_in': 112,
        'filters_out': 192,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 2,
        'se_ratio': 0.0
    }, {
        'kernel_size': 3,
        'repeats': 1,
        'filters_in': 192,
        'filters_out': 320,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 1,
        'se_ratio': 0.0
    }]

    return EfficientNetWithAttention(1.0, 1.0, 0.2, blocks_args=blocks_args, include_top=include_top,
                                     input_shape=input_shape, pooling=pooling, classes=classes,
                                     classifier_activation=classifier_activation, module=module)


# EfficientNet lite0
def EfficientNetLite0(include_top=True, input_shape=(256, 3), pooling=None, classes=6,
                      classifier_activation='softmax'):
    return EfficientNetLite0WithAttention(include_top, input_shape, pooling, classes, classifier_activation, module=None)

