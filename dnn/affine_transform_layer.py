from tensorflow import keras
import tensorflow as tf
import numpy as np

SCALE_L = 0.5
SHEAR_L = 0.2


class AffineTransformLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AffineTransformLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(
            shape=(2,),
            initializer=keras.initializers.Zeros(),
            trainable=True,
            name="scale",
        )

        self.shear = self.add_weight(
            shape=(2,),
            initializer=keras.initializers.Zeros(),
            trainable=True,
            name="shear",
        )

        self.rot = self.add_weight(
            shape=(1,),
            initializer=keras.initializers.Zeros(),
            trainable=True,
            name="rotate",
        )

        self.trans = self.add_weight(
            shape=(2,),
            initializer=keras.initializers.Zeros(),
            trainable=True,
            name="translate",
        )

    def call(self, inputs):
        """
        the input shape should be BxNx2
        each vector contains the keypoint position in the image-coord
        [u, v]^T
        """
        sin_theta = tf.sin(self.rot)
        cos_theta = tf.cos(self.rot)

        r = [
            [
                self.scale[0]
                * SCALE_L
                * (cos_theta - self.shear[1] * SHEAR_L * sin_theta),
                self.scale[0]
                * SCALE_L
                * (sin_theta + self.shear[1] * SHEAR_L * cos_theta),
            ],
            [
                self.scale[1]
                * SCALE_L
                * (cos_theta * self.shear[0] * SHEAR_L - sin_theta),
                self.scale[1]
                * SCALE_L
                * (sin_theta * self.shear[0] * SHEAR_L + cos_theta),
            ],
        ]

        out = tf.map_fn(
            lambda x: tf.matmul(x, r + tf.eye(2)) + self.trans,
            inputs,
        )

        # out_norm = out / out[:, 2]
        return out

    def get_config(self):
        config = super().get_config()
        # config.update({"dim": self.dim, "ortho_lambda": self.ortho_lambda})
        return config

    @classmethod
    def from_config(cls, config):
        # dim = config.pop("dim")
        # ortho_lambda = config.pop("ortho_lambda")
        return cls()

    @staticmethod
    def get_trans_matrix(model):
        trans_weights = model.get_layer("affine_transform").get_weights()
        [print("trans weights\n", trans_weight) for trans_weight in trans_weights]

        scale, shear, rot, trans = trans_weights

        sin_theta = np.sin(rot)
        cos_theta = np.cos(rot)

        r = [
            [
                scale[0] * SCALE_L * (cos_theta - shear[1] * SHEAR_L * sin_theta),
                scale[0] * SCALE_L * (sin_theta + shear[1] * SHEAR_L * cos_theta),
            ],
            [
                scale[1] * SCALE_L * (cos_theta * shear[0] * SHEAR_L - sin_theta),
                scale[1] * SCALE_L * (sin_theta * shear[0] * SHEAR_L + cos_theta),
            ],
        ]

        trans_r = np.array(r).reshape(2, 2) + np.eye(2)
        trans_t = np.array(trans).reshape(1, 2)
        return {
            "r": trans_r,
            "t": trans_t,
        }
