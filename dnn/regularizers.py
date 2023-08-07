from tensorflow import keras
import tensorflow as tf


class OrthoRegularizer(keras.regularizers.Regularizer):
    def __init__(self, dim, ortho_lambda) -> None:
        self.ortho_lambda = ortho_lambda
        self.dim = dim

    def __call__(self, w):
        w_t = tf.transpose(w)
        identity = tf.eye(self.dim)
        ortho_loss = tf.reduce_sum(tf.square(tf.matmul(w_t, w) - identity))
        return self.ortho_lambda * ortho_loss


class OnesRegularizer(keras.regularizers.Regularizer):
    def __init__(self, shape, ortho_lambda) -> None:
        self.ortho_lambda = ortho_lambda
        self.shape = shape

    def __call__(self, w):
        ones = tf.ones(self.shape)
        ortho_loss = tf.reduce_sum(tf.square(w - ones))
        return self.ortho_lambda * ortho_loss


class WeightRegularizer(keras.regularizers.Regularizer):
    def __init__(self, ortho_lambda) -> None:
        self.ortho_lambda = ortho_lambda

    def __call__(self, w):
        ortho_loss = tf.reduce_sum(tf.square(w - 0.5))
        return self.ortho_lambda * ortho_loss


class WeightLayer(keras.layers.Layer):
    def __init__(self, dim, ortho_lambda=0.01, **kwargs):
        super(WeightLayer, self).__init__(**kwargs)
        self.dim = dim
        self.ortho_lambda = ortho_lambda

    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=(self.dim),
            initializer=keras.initializers.constant(0.5),
            trainable=True,
            regularizer=WeightRegularizer(self.ortho_lambda),
        )

    def call(self, inputs, *args, **kwargs):
        return inputs * self.weight
