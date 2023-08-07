from tensorflow import keras
import tensorflow as tf


class KNNSelfLayer(keras.layers.Layer):
    def __init__(self, k, **kwargs):
        super(KNNSelfLayer, self).__init__(**kwargs)
        self.k = k

    def build(self, input_shape):
        super(KNNSelfLayer, self).build(input_shape)

    def call(self, inputs):
        """
        input shape B x N x F
        """
        B, N, F = inputs.shape

        expand_input = tf.repeat(tf.expand_dims(inputs, 2), repeats=N, axis=2)

        # distance_matrix B x N x N
        distance_matrix = tf.reduce_sum(
            tf.abs(
                tf.subtract(expand_input, tf.transpose(expand_input, perm=[0, 2, 1, 3]))
            ),
            axis=-1,
        )

        # indices shape B x N x K
        _, indices = tf.nn.top_k(-distance_matrix, k=self.k + 1)

        # neighbors shape B x N x K x F
        neighbors = tf.gather(inputs, indices, batch_dims=1)

        # output shape B x N x F x (1+ K)
        output = tf.transpose(neighbors, perm=[0, 1, 3, 2])

        return output

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
