import tensorflow as tf
from ..modules.attention import BaseAttention


class ConvBlock:
    def __init__(self, filters, alpha, kernel=3, strides=1):
        self.filters = filters
        self.alpha = alpha
        self.kernel = kernel
        self.strides = strides

    def __call__(self, x):
        filters = int(self.filters * self.alpha)
        x = tf.keras.layers.Conv1D(filters, self.kernel, self.strides, padding='same', use_bias=False, name='conv1')(x)
        x = tf.keras.layers.BatchNormalization(name='conv1_bn')(x)
        x = tf.keras.layers.ReLU(6., name='conv1_relu')(x)
        return x


class DepthwiseConvBlock:
    def __init__(self, pointwise_conv_filter, alpha, depth_multipliter=1, strides=1, block_id=1):
        self.pointwise_conv_filter = pointwise_conv_filter
        self.alpha = alpha
        self.depth_multipliter = depth_multipliter
        self.strides = strides
        self.block_id = block_id

    def __call__(self, x):
        if self.strides != 1:
            x = tf.keras.layers.ZeroPadding1D((0, 1), name='conv_pad_%d' % self.block_id)(x)

        x = tf.keras.layers.SeparableConv1D(int(x.shape[-1]),
                                            3,
                                            padding='same' if self.strides == 1 else 'valid',
                                            depth_multiplier=self.depth_multipliter,
                                            strides=self.strides, use_bias=False,
                                            name='conv_dw_%d' % self.block_id)(x)
        x = tf.keras.layers.BatchNormalization(name='conv_dw_%d_bn' % self.block_id)(x)
        x = tf.keras.layers.ReLU(6., name='conv_dw_%d_relu' % self.block_id)(x)

        x = tf.keras.layers.Conv1D(self.pointwise_conv_filter, 1, padding='same', use_bias=False,
                                   strides=1, name='conv_pw_%d' % self.block_id)(x)
        x = tf.keras.layers.BatchNormalization(name='conv_pw_%d_bn' % self.block_id)(x)
        x = tf.keras.layers.ReLU(6., name='conv_pw_%d_relu' % self.block_id)(x)
        return x


# Attention-insertable MobileNet
def MobileNetWithAttention(include_top=True, input_shape=(256, 3), pooling=None, classes=6,
                           classifier_activation='softmax', module: BaseAttention = None,
                           alpha=1.0, depth_multiplier=1, dropout=1e-3):
    """

    Parameters
    ----------
    include_top
    input_shape
    pooling
    classes
    classifier_activation
    module
    alpha
    depth_multiplier
    dropout
    Returns
    -------
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = ConvBlock(32, alpha, strides=2)(inputs)
    if module is not None:
        x = module(32, block_name="conv1")(x)
    x = DepthwiseConvBlock(64, alpha, depth_multiplier, block_id=1)(x)
    if module is not None:
        x = module(64, block_name="dwconv1")(x)

    x = DepthwiseConvBlock(128, alpha, depth_multiplier, strides=2, block_id=2)(x)
    if module is not None:
        x = module(128, block_name="dwconv2")(x)
    x = DepthwiseConvBlock(128, alpha, depth_multiplier, block_id=3)(x)
    if module is not None:
        x = module(128, block_name="dwconv3")(x)

    x = DepthwiseConvBlock(256, alpha, depth_multiplier, strides=2, block_id=4)(x)
    if module is not None:
        x = module(256, block_name="dwconv4")(x)
    x = DepthwiseConvBlock(256, alpha, depth_multiplier, block_id=5)(x)
    if module is not None:
        x = module(256, block_name="dwconv5")(x)

    x = DepthwiseConvBlock(512, alpha, depth_multiplier, strides=2, block_id=6)(x)
    if module is not None:
        x = module(512, block_name="dwconv6")(x)
    x = DepthwiseConvBlock(512, alpha, depth_multiplier, block_id=7)(x)
    if module is not None:
        x = module(512, block_name="dwconv7")(x)
    x = DepthwiseConvBlock(512, alpha, depth_multiplier, block_id=8)(x)
    if module is not None:
        x = module(512, block_name="dwconv8")(x)
    x = DepthwiseConvBlock(512, alpha, depth_multiplier, block_id=9)(x)
    if module is not None:
        x = module(512, block_name="dwconv9")(x)
    x = DepthwiseConvBlock(512, alpha, depth_multiplier, block_id=10)(x)
    if module is not None:
        x = module(512, block_name="dwconv10")(x)
    x = DepthwiseConvBlock(512, alpha, depth_multiplier, block_id=11)(x)
    if module is not None:
        x = module(512, block_name="dwconv11")(x)

    x = DepthwiseConvBlock(1024, alpha, depth_multiplier, strides=2, block_id=12)(x)
    if module is not None:
        x = module(1024, block_name="dwconv12")(x)
    x = DepthwiseConvBlock(1024, alpha, depth_multiplier, block_id=13)(x)
    if module is not None:
        x = module(1024, block_name="dwconv13")(x)

    if include_top:

        shape = (1, int(1024 * alpha))

        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Reshape(shape, name='reshape_1')(x)
        x = tf.keras.layers.Dropout(dropout, name='dropout')(x)
        x = tf.keras.layers.Conv1D(classes, 1, padding='same', name='conv_preds')(x)
        x = tf.keras.layers.Reshape((classes,), name='reshape_2')(x)
        x = tf.keras.layers.Activation(activation=classifier_activation, name='predictions')(x)

        model_ = tf.keras.models.Model(inputs=inputs, outputs=x)
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


# MobileNet
def MobileNet(include_top=True, input_shape=(256, 3), pooling=None, classes=6,
                           classifier_activation='softmax', alpha=1.0, depth_multiplier=1, dropout=1e-3):
    return MobileNetWithAttention(include_top, input_shape, pooling, classes, classifier_activation,
                                  module=None, alpha=alpha, depth_multiplier=depth_multiplier, dropout=dropout)
