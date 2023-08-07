from tensorflow import keras
import tensorflow as tf

from dnn.regularizers import OnesRegularizer, OrthoRegularizer


class MatrixTransformLayer(keras.layers.Layer):
    def __init__(self, dim, ortho_lambda=0.1, **kwargs):
        super(MatrixTransformLayer, self).__init__(**kwargs)

        self.dim = dim
        self.ortho_lambda = ortho_lambda
        self.ortho_init = MatrixTransformLayer.ortho_initializer

    @staticmethod
    def ortho_initializer(shape, dtype=None):
        matrix = tf.eye(shape[0], shape[1], dtype=dtype)
        return matrix

    def build(self, input_shape):
        self.scale = self.add_weight(
            shape=(1, 2),
            initializer=keras.initializers.Ones(),
            trainable=True,
            regularizer=OnesRegularizer((1, 2), 0.5),
        )

        self.rot = self.add_weight(
            shape=(self.dim, self.dim),
            initializer=self.ortho_init,
            trainable=True,
            regularizer=OrthoRegularizer(self.dim, 0.5),
            name="r",
        )

        if self.dim == 3:
            self.ones = self.add_weight(
                shape=(input_shape[1], 1),
                initializer=keras.initializers.Ones(),
                trainable=False,
            )
        else:
            self.trans = self.add_weight(
                shape=(1, self.dim),
                initializer=keras.initializers.Zeros(),
                trainable=True,
                name="t",
            )

    def call(self, inputs):
        """
        the input shape should be BxNx2
        each vector contains the keypoint position in the image-coord
        [u, v]^T
        """
        if self.dim == 2:
            out = tf.matmul(tf.multiply(inputs, self.scale), self.rot)
            out = out + self.trans
        else:
            out = tf.map_fn(
                lambda x: tf.matmul(tf.concat((x, self.ones), axis=-1), self.rot)[
                    :, :2
                ],
                inputs,
            )

        # out_norm = out / out[:, 2]
        return out

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "ortho_lambda": self.ortho_lambda})
        return config

    @classmethod
    def from_config(cls, config):
        dim = config.pop("dim")
        ortho_lambda = config.pop("ortho_lambda")
        return cls(dim, ortho_lambda)
