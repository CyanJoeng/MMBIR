from sys import argv, path as sys_path
from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from dnn.transform import (
    net_matches,
    net_transform_pos,
    MatrixTransformLayer,
    matrix_initializer,
)
from dnn.model_parameters import NUM_FEAT_INPUT


def load_training_data(data_path):
    print(f"loading data from {str(data_path)}")

    with open(str(data_path), "rb") as f:
        data = pickle.load(f)
    assert len(data) == 2

    pano = data["pano"]
    pano_pos = pano["pt"]
    pano_feat = pano["desc"]

    he = data["he"]
    he_pos = he["pt"]
    he_feat = he["desc"]

    return ((pano_pos, pano_feat), (he_pos, he_feat))


if __name__ == "__main__":
    if len(argv) != 3:
        print(f"Usage: {argv[0]} method data_id")
        exit(-1)
    method = argv[1]
    data_id = argv[2]

    input_data_path = (
        Path("outputs") / "cache" / "dnn" / data_id / f"{method}_features.pkl"
    )
    output_path = Path("outputs") / "dnn" / "trans"
    output_path.mkdir(parents=True, exist_ok=True)

    EPOCHS = 20
    BATCH_SIZE = 1

    data_pano, data_he = load_training_data(input_data_path)

    input_pano_pos = np.array(data_pano[0]).reshape((1, -1, 2)).astype(np.float32)
    input_he_pos = np.array(data_he[0]).reshape((1, -1, 2)).astype(np.float32)

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

    model = net_matches(num_feature_input=input_he_pos.shape[1])
    model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
    model.summary(expand_nested=True)

    coord_scale = 100
    input_he_pos /= coord_scale
    input_pano_pos /= coord_scale

    model.fit(dataset, epochs=EPOCHS)

    trans_weights = model.get_layer("trans_model").get_layer("matrix").get_weights()
    print("trans weights ", trans_weights)

    trans_rot = np.transpose(trans_weights[0])
    trans = {
        "r": np.array(trans_weights[0]),
        "t": np.array(trans_weights[1]) * coord_scale,
    }
    print("transform \n", trans)

    with open(str(output_path / "trans2d_r_t.pkl"), "wb") as f:
        pickle.dump(
            trans,
            f,
        )
    print("save transformation matrix to ", str(output_path / "trans2d_r_t.pkl"))

    model.save(str(output_path / "model_trans2d.h5"))

    # with keras.utils.custom_object_scope(
    #     {
    #         "MatrixTransformLayer": MatrixTransformLayer,
    #         "matrix_initializer": matrix_initializer,
    #     }
    # ):
    #     model = keras.models.load_model(str(output_path / "model_trans2d.h5"))
    # model.summary()

    weight = model.get_layer("weight_layer").get_weights()
    print(weight > tf.reduce_mean(weight))
