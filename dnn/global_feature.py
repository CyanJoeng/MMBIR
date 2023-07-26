from tensorflow import keras
import tensorflow as tf
from pathlib import Path
from dnn.model_parameters import *


def net_point_feature(num_keypoints: int, feat_len: int, trans_matrix=None):
    """
    input feat shape BxNxF
        B: batch size
        N: number of keypoints in a image
        F: length of keypoint feature
    input pos shape BxNx2
        two dimensions in image coordinate
    outout feat shape BxNxOUT_FEAT_LEN
    """

    N, F = num_keypoints, feat_len
    input_feat = keras.Input(shape=(N, F), name="input_feat")
    input_pos = keras.Input(shape=(N, 2), name="input_pos")

    out_feat = keras.Sequential(
        [
            keras.layers.Reshape((N, F, 1)),
            keras.layers.Conv2D(
                128, (1, F), strides=1, padding="valid", activation="relu"
            ),
        ]
    )(input_feat)

    out_pos = keras.Sequential(
        [
            keras.layers.Reshape((N, 2, 1)),
            keras.layers.Conv2D(
                64, (1, 2), strides=1, padding="valid", activation="relu"
            ),
        ]
    )(input_pos)

    input = keras.layers.Concatenate(axis=-1)((out_feat, out_pos))
    output_point_feat = keras.Sequential(
        [
            keras.layers.Conv2D(
                256, (1, 1), strides=1, padding="valid", activation="relu"
            ),
            keras.layers.Conv2D(
                512, (1, 1), strides=1, padding="valid", activation="relu"
            ),
            keras.layers.Conv2D(
                OUT_FEAT_LEN, (1, 1), strides=1, padding="valid", activation="relu"
            ),
        ],
        name="point_feat",
    )(input)

    point_feat_model = keras.Model(
        inputs=(input_feat, input_pos),
        outputs=output_point_feat,
        name="point_feat_model",
    )

    return point_feat_model


def net_global_feature(num_keypoints=0, feat_len=0):
    N, F = num_keypoints, feat_len
    input_feat = keras.Input(shape=(N, F), name="input_feat")
    input_pos = keras.Input(shape=(N, 2), name="input_pos")

    point_feat_model = net_point_feature(num_keypoints, feat_len)

    output_point_feat = point_feat_model((input_feat, input_pos))
    output_img_feat = keras.Sequential(
        [
            keras.layers.MaxPool2D(pool_size=(N, 1)),
            keras.layers.Reshape((-1,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(GLOBAL_FEAT_LEN, activation="relu"),
        ],
        name="global_feat",
    )(output_point_feat)

    net = keras.Model(
        inputs=(input_feat, input_pos),
        outputs=output_img_feat,
        name="global_feat_model",
    )
    return net


def net_trans_pos(num_keypoints, feat_len):
    N, F = num_keypoints, feat_len
    input_pos = keras.Input(shape=(N, 2), name="input_pos")

    out_feat = keras.Sequential(
        [
            keras.layers.Reshape((N, 2, 1)),
            keras.layers.Conv2D(
                32, (1, 2), strides=1, padding="valid", activation="relu"
            ),
            keras.layers.Conv2D(
                64, (1, 1), strides=1, padding="valid", activation="relu"
            ),
            keras.layers.Conv2D(
                128, (1, 1), strides=1, padding="valid", activation="relu"
            ),
            keras.layers.MaxPool2D(pool_size=(N, 1)),
            keras.layers.Reshape((-1,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(6, activation="relu"),
        ]
    )(input_pos)

    matrix_data = tf.reshape(out_feat, (-1, 3, 2))

    r = matrix_data[:, :2, :]
    t = matrix_data[:, 2:, :]
    print(f"r {r.shape} t {t.shape}")

    model = keras.Model(inputs=input_pos, outputs=(r, t), name="trans_model")
    return model


def net_siamese_global_feats(num_keypoints: int, feat_len: int, trans=False):
    N, F = num_keypoints, feat_len
    global_feat_model = net_global_feature(num_keypoints, feat_len)

    input_0_feat = keras.Input(shape=(N, F))
    input_0_pos = keras.Input(shape=(N, 2))
    input_1_feat = keras.Input(shape=(N, F))
    input_1_pos = keras.Input(shape=(N, 2))

    out_0 = global_feat_model((input_0_feat, input_0_pos))
    out_1 = global_feat_model((input_1_feat, input_1_pos))

    print(f"[Siamese global]out 0 {out_0.shape}  out1 {out_1.shape}")

    if trans:
        g_feat_0 = tf.reshape(out_0, (-1, GLOBAL_FEAT_LEN, 1, 1))
        g_feat_1 = tf.reshape(out_1, (-1, GLOBAL_FEAT_LEN, 1, 1))
        concat_globale_feat = tf.concat((g_feat_0, g_feat_1), axis=-1)
        trans_weight = keras.Sequential(
            [
                keras.layers.Conv2D(
                    64, (1, 1), strides=1, padding="valid", activation="relu"
                ),
                keras.layers.Conv2D(
                    128, (1, 1), strides=1, padding="valid", activation="relu"
                ),
                keras.layers.MaxPool2D(pool_size=(GLOBAL_FEAT_LEN, 1)),
                keras.layers.Reshape((-1,)),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(rate=0.5),
                keras.layers.Dense(6, activation="sigmoid"),
                keras.layers.Reshape((3, 2)),
            ]
        )(concat_globale_feat)

        print(f"[Siamese global]trans weight {trans_weight.shape}")

        trans_0_pos = (
            tf.matmul(input_0_pos, trans_weight[:, :2, :]) + trans_weight[:, 2:, :]
        )

        print(f"[Siamese global]trans 0 pos {trans_0_pos.shape}")
        out_0 = global_feat_model((input_0_feat, trans_0_pos))
        print(f"[Siamese global]out 0 {out_0.shape}  out1 {out_1.shape}")

    output = keras.layers.Lambda(lambda x: tf.square(x[0] - x[1]))([out_0, out_1])
    output = keras.layers.Dense(1, activation="sigmoid")(output)

    siamese_model = keras.Model(
        inputs=[input_0_feat, input_0_pos, input_1_feat, input_1_pos], outputs=output
    )
    return siamese_model


def get_trained_point_feat_net(num_features) -> keras.Model:
    model_path = Path("outputs") / "dnn" / "models" / "siamese_model.h5"
    assert model_path.exists()

    model = keras.models.load_model(model_path)
    point_feat_model = net_point_feature(num_features, SPP_FEAT_LEN)
    point_feat_model.set_weights(
        model.get_layer("global_feat_model").get_layer("point_feat_model").get_weights()
    )
    return point_feat_model
