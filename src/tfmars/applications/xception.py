from tensorflow.keras import layers
from tensorflow.keras.models import Model

from ..modules.attention import BaseAttention, SqueezeAndExcite


def XceptionWithAttention(include_top=True, input_shape=(256, 3), pooling=None, classes=6,
                             classifier_activation='softmax', module: BaseAttention = SqueezeAndExcite):
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

    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(32, 3, strides=2, use_bias=False, name='block1_conv1')(inputs)
    x = layers.BatchNormalization(name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)
    x = layers.Conv1D(64, 3, use_bias=False, name='block1_conv2')(x)
    x = layers.BatchNormalization(name='block1_conv2_bn')(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)

    x = module(64, block_name="block1")(x)

    residual = layers.Conv1D(
        128, 1, strides=2, padding='same', use_bias=False
    )(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv1D(128, 3, padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = layers.BatchNormalization(name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv1D(128, 3, padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = layers.BatchNormalization(name='block2_sepconv2_bn')(x)

    x = layers.MaxPooling1D(3, strides=2, padding='same', name='block2_pool')(x)
    x = layers.add([x, residual])

    x = module(128, block_name="block2")(x)

    residual = layers.Conv1D(
        256, 1, strides=2, padding='same', use_bias=False
    )(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConv1D(256, 3, padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = layers.BatchNormalization(name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv1D(256, 3, padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = layers.BatchNormalization(name='block3_sepconv2_bn')(x)

    x = layers.MaxPooling1D(3, strides=2, padding='same', name='block3_pool')(x)
    x = layers.add([x, residual])

    x = module(256, block_name="block3")(x)

    residual = layers.Conv1D(728, 1, strides=2, padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv1D(728, 3, padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = layers.BatchNormalization(name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = layers.SeparableConv1D(728, 3, padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = layers.BatchNormalization(name='block4_sepconv2_bn')(x)

    x = layers.MaxPooling1D(3, strides=2, padding='same', name='block4_pool')(x)
    x = layers.add([x, residual])

    x = module(728, block_name="block4")(x)

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = layers.Activation('relu', name=prefix + "_sepconv1_act")(x)
        x = layers.SeparableConv1D(728, 3, padding='same', use_bias=False, name=prefix + "_sepconv1")(x)
        x = layers.BatchNormalization(name=prefix + "_sepconv1_bn")(x)
        x = layers.Activation('relu', name=prefix + "_sepconv2_act")(x)
        x = layers.SeparableConv1D(728, 3, padding='same', use_bias=False, name=prefix + "_sepconv2")(x)
        x = layers.BatchNormalization(name=prefix + "_sepconv2_bn")(x)
        x = layers.Activation('relu', name=prefix + "_sepconv3_act")(x)
        x = layers.SeparableConv1D(728, 3, padding='same', use_bias=False, name=prefix + "_sepconv3")(x)

        x = layers.add([x, residual])

        x = module(728, block_name=prefix)(x)

    residual = layers.Conv1D(1024, 1, strides=2, padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block13_sepconv1_act')(x)
    x = layers.SeparableConv1D(728, 3, padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = layers.BatchNormalization(name='block13_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block13_speconv2_act')(x)
    x = layers.SeparableConv1D(1024, 3, padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = layers.BatchNormalization(name='block13_sepconv2_bn')(x)

    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
    x = layers.add([x, residual])

    x = module(1024, block_name="block13")(x)

    x = layers.SeparableConv1D(1536, 3, padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = layers.BatchNormalization(name='block14_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv1_act')(x)

    x = layers.SeparableConv1D(2048, 3, padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = layers.BatchNormalization(name='block14_sepconv2_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv2_act')(x)

    x = module(2048, block_name="block14")(x)

    if include_top:
        x = layers.GlobalAveragePooling1D(name='avg_pool')(x)
        y = layers.Dense(classes, activation=classifier_activation,
                         name='predictions')(x)

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
            return