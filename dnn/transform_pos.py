from tensorflow import keras
import tensorflow as tf
from dnn.knn_pair_layer import KNNPairLayer
from pathlib import Path

from dnn.matrix_transform_layer import MatrixTransformLayer
from dnn.model_parameters import (
    GLOBAL_FEAT_LEN,
    OUT_FEAT_LEN,
    SPP_FEAT_LEN,
    NUM_FEAT_INPUT,
)


# def net_transform_pos(num_feature_input: int):
#     """
#     the input is a set of keypoint positions in moving image,
#     and the shape of the input is BxNx2
#     """
#     N = num_feature_input
#     DIM = 2

#     input = keras.Input(shape=(N, DIM))
#     output = MatrixTransformLayer(DIM, name="matrix")(input)
#     model = keras.Model(inputs=input, outputs=output, name="trans_model")
#     return model


def net_transform_pos(num_neighbor: int = 1):
    """
    input feat shape BxNxF
        B: batch size
        N: number of keypoints in a image
        F: length of keypoint feature
        K: number of neighbors
    input pos shape BxNx2
        two dimensions in image coordinate
    outout feat shape BxNxOUT_FEAT_LEN
    """

    N, F, K = NUM_FEAT_INPUT, SPP_FEAT_LEN, num_neighbor
    input_0_feat = keras.Input(shape=(N, F, K))
    input_0_pos = keras.Input(shape=(N, 2))
    input_1_feat = keras.Input(shape=(N, F, K))
    input_1_pos = keras.Input(shape=(N, 2))

    feat_layers = keras.Sequential(
        [
            keras.layers.Reshape((N, F, K)),
            keras.layers.Conv2D(
                64, (1, F), strides=1, padding="valid", activation="relu"
            ),
            # keras.layers.Conv2D(
            #     64, (1, 1), strides=1, padding="valid", activation="relu"
            # ),
            keras.layers.Reshape((N, -1)),
        ]
    )

    new_feat0 = feat_layers(input_0_feat)
    new_feat1 = feat_layers(input_1_feat)

    transed_pos = MatrixTransformLayer(dim=2, name="trans_matrix")(input_0_pos)
    matched_pos, dist = KNNPairLayer(k=1)((new_feat0, new_feat1, input_1_pos))

    output = tf.reduce_sum(
        tf.divide(
            tf.reduce_sum(tf.square(tf.subtract(transed_pos, matched_pos)), axis=-1),
            dist,
        ),
        axis=-1,
    )

    model = keras.Model(
        inputs=[input_0_feat, input_0_pos, input_1_feat, input_1_pos], outputs=output
    )

    return model


def load_trained_trans_pos_net(num_neighbor) -> keras.Model:
    model_path = (
        Path("outputs") / "dnn" / "models" / f"trans_pos_model_n{num_neighbor}.h5"
    )
    print("Load model from ", str(model_path))
    assert model_path.exists()
    with keras.utils.custom_object_scope(
        {
            "KNNPairLayer": KNNPairLayer,
        }
    ):
        model = keras.models.load_model(model_path)
    return model
