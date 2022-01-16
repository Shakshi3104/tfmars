import tensorflow as tf
from ..modules.attention import BaseAttention


# Attention-insertable Simple CNN
def SimpleCNNWithAttention(include_top=True, input_shape=(256, 3), pooling=None, classes=6, classifier_activation='softmax',
                           module: BaseAttention=None):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(50, 11, 1, padding="same", activation="relu",
                               kernel_initializer="he_normal", name="conv1")(inputs)
    x = tf.keras.layers.MaxPooling1D(2, padding="same", name="pool1")(x)
    if module is not None:
        x = module(50, block_name="conv1")(x)
    x = tf.keras.layers.Conv1D(40, 10, 1, padding="same", activation="relu",
                               kernel_initializer="he_normal", name="conv2")(x)
    x = tf.keras.layers.MaxPooling1D(3, padding="same", name="pool2")(x)
    if module is not None:
        x = module(40, block_name="conv2")(x)
    x = tf.keras.layers.Conv1D(30, 6, 1, padding="same", activation="relu",
                               kernel_initializer="he_normal", name="conv3")(x)
    x = tf.keras.layers.MaxPooling1D(1, padding="same", name="pool3")(x)
    if module is not None:
        x = module(30, block_name="conv3")(x)

    if include_top:
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1000, kernel_initializer="he_normal", name="fc")(x)
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


# Simple CNN
def SimpleCNN(include_top=True, input_shape=(256, 3), pooling=None, classes=6, classifier_activation='softmax'):
    return SimpleCNNWithAttention(include_top, input_shape, pooling, classes, classifier_activation)