from sys import argv
from pathlib import Path
import numpy as np
from tensorflow import keras
import tensorflow as tf

from dnn.model_parameters import *
from dnn.global_feature import net_siamese_global_feats
from keypoints.feature_gspp import Gspp, GsppFeature

from utils.dataset import load_image
from utils.display import show_keypoints
from keypoints.feature_spp import Spp


EPOCHS = 40
BATCH_SIZE = 2

FEATURE_METHOD = Spp
NEIGHBOR = 1
# NEIGHBOR = GsppFeature.N_EDGE

DOWN_SIZE = 4


def method_str():
    if FEATURE_METHOD is Spp:
        return "spp"
    else:
        return "gspp"


def detect_features(img_path, cache_dir, label):
    img = load_image(img_path, verbose=True, downsize=DOWN_SIZE)

    method = FEATURE_METHOD(img)
    method.compute(num_feat=NUM_FEAT_DETECT, cache_dir=str(cache_dir / label))
    features = method.features

    show_keypoints(
        img,
        features,
        str(cache_dir / label / f"keypoints_x{DOWN_SIZE}.tif"),
    )
    return features, img.shape


def data_pipeline(dataset_folder, data_id, cache_dir):
    data_dir = Path(dataset_folder) / "H&E_IMC" / "Pair" / data_id
    data_dir = Path(data_dir).absolute()
    main_id = str(data_id).split("_")[0]

    cache_dir = Path(cache_dir / method_str() / data_id)
    cache_dir.mkdir(parents=True, exist_ok=True)

    img_path = str(data_dir / f"HE{main_id}.tif")
    he_features, img_he_shape = detect_features(img_path, cache_dir, "he")

    img_path = str(data_dir / f"{main_id}_panorama.tif")
    pano_features, _ = detect_features(img_path, cache_dir, "pano")

    return he_features, pano_features, img_he_shape


def set_desc(data, feature):
    if FEATURE_METHOD is Spp:
        data[:, 0] = feature.desc
    else:
        for idx in range(NEIGHBOR):
            data[:, idx] = feature.desc[idx]["desc"]


def set_pos(data, feature, scale=1):
    if FEATURE_METHOD is Spp:
        center = np.array(feature.keypoint.pt)
        data[:, 0] = center / scale * 2 - 1
    else:
        center = np.array(feature.keypoint.pt)
        for idx in range(NEIGHBOR):
            off = np.array(feature.desc[idx]["vec"])
            data[:, idx] = (off + center) / scale * 2 - 1


def create_train_dataset(dataset):
    feat_len = SPP_FEAT_LEN
    print(f"feature len {feat_len}")

    input_he_feat = np.zeros((SAMPLES * 4, NUM_FEAT_INPUT, feat_len, NEIGHBOR))
    input_he_pose = np.zeros((SAMPLES * 4, NUM_FEAT_INPUT, 2, NEIGHBOR))
    input_pano_feat = np.zeros((SAMPLES * 4, NUM_FEAT_INPUT, feat_len, NEIGHBOR))
    input_pano_pose = np.zeros((SAMPLES * 4, NUM_FEAT_INPUT, 2, NEIGHBOR))
    input_label = np.zeros((SAMPLES * 4,), dtype=np.int32)

    # create dataset by random sampling
    for data_id, (he_id, pano_id, label) in enumerate(
        [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    ):
        he_features, pano_features, he_img_shape = (
            dataset[he_id][0],
            dataset[pano_id][1],
            dataset[he_id][2],
        )

        he_img_sz = np.array(he_img_shape)[:2][::-1]
        for sample_id in range(SAMPLES):
            input_label[SAMPLES * data_id + sample_id] = label

            sample_he = np.random.choice(
                range(NUM_FEAT_DETECT), size=NUM_FEAT_INPUT, replace=False
            )
            sample_pano = np.random.choice(
                range(NUM_FEAT_DETECT), size=NUM_FEAT_INPUT, replace=False
            )

            for idx in range(NUM_FEAT_INPUT):
                set_desc(
                    input_he_feat[SAMPLES * data_id + sample_id, idx],
                    he_features[sample_he[idx]],
                )
                set_pos(
                    input_he_pose[SAMPLES * data_id + sample_id, idx],
                    he_features[sample_he[idx]],
                    he_img_sz,
                )

                set_desc(
                    input_pano_feat[SAMPLES * data_id + sample_id, idx],
                    pano_features[sample_pano[idx]],
                )
                set_pos(
                    input_pano_pose[SAMPLES * data_id + sample_id, idx],
                    pano_features[sample_pano[idx]],
                    he_img_sz,
                )

    return (input_he_feat, input_he_pose, input_pano_feat, input_pano_pose), input_label


if __name__ == "__main__":
    if len(argv) != 4:
        print(f"Usage: {argv[0]} dataset_folder data_id_0 data_id_1")
        exit(-1)

    # get model
    model = net_siamese_global_feats(NUM_FEAT_INPUT, SPP_FEAT_LEN, NEIGHBOR)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy()],
    )
    model.summary(expand_nested=True)

    # create dataset
    dataset_folder = argv[1]
    data_id_0, data_id_1 = argv[2], argv[3]
    cache_dir = (
        Path("outputs") / "cache" / "dnn" / f"{data_id_0}-{data_id_1}-x{DOWN_SIZE}"
    )
    output_dir = Path("outputs") / "dnn" / f"{data_id_0}-{data_id_1}-x{DOWN_SIZE}"

    dataset = (
        data_pipeline(dataset_folder, data_id_0, cache_dir),
        data_pipeline(dataset_folder, data_id_1, cache_dir),
    )
    dataset = create_train_dataset(dataset)
    dataset_len = dataset[1].shape[0]
    print(f"dataset len {dataset_len}")

    dataset = tf.data.Dataset.zip(
        (
            tf.data.Dataset.zip(
                tuple([tf.data.Dataset.from_tensor_slices(data) for data in dataset[0]])
            ),
            tf.data.Dataset.from_tensor_slices(dataset[1]),
        )
    )
    dataset = dataset.shuffle(buffer_size=dataset_len)
    dataset = dataset.batch(batch_size=BATCH_SIZE)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(output_dir / f"siamese_model_n{NEIGHBOR}.h5")
    model_save_cb = keras.callbacks.ModelCheckpoint(
        model_path, "binary_accuracy", save_best_only=True, mode="max"
    )

    model.fit(dataset, epochs=EPOCHS, callbacks=[model_save_cb])

    # model.save(model_path)
    print(f"Save model to {model_path}")

    # model = keras.models.load_model(str(output_dir / "siamese_model.h5"))
    # point_feat_model = net_point_feature(NUM_FEAT_INPUT, FEAT_LEN)
    # point_feat_model.set_weights(
    #     model.get_layer("global_feat_model").get_layer("point_feat_model").get_weights()
    # )
    # point_feat_model.summary()
    # for weight in point_feat_model.get_weights():
    #     print(weight)
