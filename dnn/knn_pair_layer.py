from tensorflow import keras
import tensorflow as tf


class KNNPairLayer(keras.layers.Layer):
    def __init__(self, k, **kwargs):
        super(KNNPairLayer, self).__init__(**kwargs)
        self.k = k

    def build(self, input_shape):
        super(KNNPairLayer, self).build(input_shape)

    def call(self, inputs):
        """
        inputs (feature0, feature1, pos1)
        inputs shape (B x N x F, B x N x F, B x N x 2)
        """
        feat0, feat1, pos1 = inputs
        B, N, F = feat0.shape

        rep_feat0 = tf.repeat(tf.expand_dims(feat0, 2), repeats=N, axis=2)
        rep_feat1 = tf.repeat(tf.expand_dims(feat1, 2), repeats=N, axis=2)

        # distance_matrix B x N x N
        distance_matrix = tf.reduce_sum(
            tf.abs(tf.subtract(rep_feat0, tf.transpose(rep_feat1, perm=[0, 2, 1, 3]))),
            axis=-1,
        )

        # dist shape B x N x K
        # indices shape B x N x K
        dist, indices = tf.nn.top_k(-distance_matrix, k=self.k)

        # neighbors shape B x N x K x 2
        matched_pos = tf.gather(pos1, indices, batch_dims=1)

        # output shape B x N x 2
        output = tf.squeeze(matched_pos, axis=2)

        dist = tf.squeeze(dist, axis=-1)
        th = tf.reduce_max(dist, axis=-1) - tf.reduce_min(dist, axis=-1) / 4
        select = tf.cast(dist < th, tf.float32) + 1e-6

        dist = tf.divide(dist, select)

        return output, dist

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "k": self.k,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        k = config.pop("k")
        return cls(k)
