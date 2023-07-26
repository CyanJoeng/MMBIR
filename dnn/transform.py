from tensorflow import keras
import tensorflow as tf
import numpy as np
from .model_parameters import NUM_FEAT_INPUT


class OrthoRegularizer(keras.regularizers.Regularizer):
    def __init__(self, dim, ortho_lambda) -> None:
        self.ortho_lambda = ortho_lambda
        self.dim = dim

    def __call__(self, w):
        w_t = tf.transpose(w)
        identity = tf.eye(self.dim)
        ortho_loss = tf.reduce_sum(tf.square(tf.matmul(w_t, w) - identity))
        return self.ortho_lambda * ortho_loss


class MatrixTransformLayer(keras.layers.Layer):
    def __init__(self, dim, ortho_initializer=None, ortho_lambda=0.1, **kwargs):
        super(MatrixTransformLayer, self).__init__(**kwargs)

        self.dim = dim
        self.ortho_reg = OrthoRegularizer(self.dim, ortho_lambda)
        self.ortho_init = ortho_initializer

    def build(self, input_shape):
        if self.ortho_init is not None:
            self.rot = self.add_weight(
                shape=(self.dim, self.dim),
                initializer=self.ortho_init,
                trainable=True,
                regularizer=self.ortho_reg,
                name="r",
            )
            self.trans = self.add_weight(
                shape=(1, self.dim),
                initializer=keras.initializers.zeros(),
                trainable=True,
                name="t",
            )

        else:
            self.rot = None
            self.trans = None

    def call(self, inputs):
        """
        the input shape should be Nx3
        each vector contains the keypoint position in the image-coord
        [u, v, 1]^T
        """
        out = tf.matmul(inputs, self.rot)
        if self.dim == 2:
            out = out + self.trans
        # out_norm = out / out[:, 2]
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "initializer": keras.initializers.serialize(self.ortho_init),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        dim = config.pop("dim")
        initializer = keras.initializers.deserialize(config.pop("initializer"))
        return cls(dim, initializer)


def matrix_initializer(shape, dtype=None):
    matrix = tf.eye(shape[0], shape[1], dtype=dtype)
    return matrix


def net_transform_pos(num_feature_input: int):
    """
    the input is a set of keypoint positions in moving image,
    and the shape of the input is BxNx2
    """
    N = num_feature_input
    DIM = 2

    input = keras.Input(shape=(N, DIM))
    output = MatrixTransformLayer(
        DIM, ortho_initializer=matrix_initializer, name="matrix"
    )(input)
    model = keras.Model(inputs=input, outputs=output, name="trans_model")
    return model


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


def net_matches(num_feature_input: int):
    N = num_feature_input
    DIM = 2

    input_0 = keras.Input(shape=(N, DIM))
    input_1 = keras.Input(shape=(N, DIM))

    trans_model = net_transform_pos(num_feature_input)

    transformed_input_0 = trans_model(input_0)

    diff = tf.reduce_sum(tf.square(input_1 - transformed_input_0), axis=-1)

    # concat_input = tf.stack([input_0, input_1], axis=-1)

    # weight = keras.Sequential(
    #     [
    #         keras.layers.Reshape((N, DIM, 2)),
    #         keras.layers.Conv2D(
    #             32, (1, DIM), strides=1, padding="valid", activation="relu"
    #         ),
    #         keras.layers.Conv2D(
    #             128, (1, 1), strides=1, padding="valid", activation="relu"
    #         ),
    #         keras.layers.MaxPool2D(pool_size=(N, 1)),
    #         keras.layers.Reshape((-1,)),
    #         keras.layers.Dense(256, activation="relu"),
    #         keras.layers.Dropout(rate=0.5),
    #         keras.layers.Dense(N, activation="softmax"),
    #     ],
    #     name="weight",
    # )(concat_input)

    weighted_diff = WeightLayer(N)(diff)

    out = tf.reduce_sum(weighted_diff, axis=-1)

    model = keras.Model(inputs=(input_0, input_1), outputs=out)
    return model


def calc_trans_matrix_by_matches(data_pano, data_he):
    EPOCHS = 20
    BATCH_SIZE = 1
    DIM = 2
    SIAMESE = False

    num_features = data_pano[0].shape[0]

    input_pano_pos = np.array(data_pano[0]).reshape((1, -1, 2)).astype(np.float32)
    input_he_pos = np.array(data_he[0]).reshape((1, -1, 2)).astype(np.float32)
    if DIM == 3:
        input_pano_pos = np.concatenate(
            (
                input_pano_pos,
                np.ones((1, num_features, 1)),
            ),
            axis=-1,
        )
        input_he_pos = np.concatenate(
            (input_he_pos, np.ones((1, num_features, 1))), axis=-1
        )

    print(f"input pos shape {input_pano_pos.shape} {input_he_pos.shape}")

    num_feat = input_he_pos.shape[1]

    if not SIAMESE:
        model = net_transform_pos(num_feat)
    else:
        model = net_matches(num_features)
    model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
    model.summary()

    # diff_pos = (input_he_pos - input_pano_pos).reshape(-1, 2)

    # diff_pos = np.mean(diff_pos, axis=0)
    # print("diff pos ", diff_pos)
    # input_pano_pos = input_pano_pos + diff_pos

    if not SIAMESE:
        model.fit(x=input_pano_pos, y=input_he_pos, epochs=EPOCHS)
        trans_weights = model.get_layer("matrix").get_weights()
    else:
        target_distance = np.array([0.0])
        dataset = tf.data.Dataset.zip(
            (
                tf.data.Dataset.zip(
                    (
                        tf.data.Dataset.from_tensor_slices(input_pano_pos),
                        tf.data.Dataset.from_tensor_slices(input_he_pos),
                    )
                ),
                tf.data.Dataset.from_tensor_slices(target_distance),
            )
        )
        dataset = dataset.batch(1)

        model.fit(dataset, epochs=EPOCHS)

        trans_weights = model.get_layer("trans_model").get_layer("matrix").get_weights()
    print("trans_weights", trans_weights)

    r = trans_weights[0]
    t = trans_weights[1]

    R = np.eye(3, dtype=np.float32)
    R[:2, :2] = r
    R[2:, :2] = t
    print("trans R\n ", R)

    return R
