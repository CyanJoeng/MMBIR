from sys import argv
from pathlib import Path
import numpy as np
from tensorflow import keras
import tensorflow as tf

from dnn.model_parameters import *
from dnn.global_feature import net_siamese_global_feats

from utils.dataset import load_image
from utils.display import show_keypoints
from keypoints.feature_spp import Spp


EPOCHS = 40
BATCH_SIZE = 2


def data_pipeline(data_id):
    data_dir = Path(argv[0]).parent.parent / "datasets" / "H&E_IMC" / "Pair" / data_id
    data_dir = Path(data_dir).absolute()
    main_id = str(data_id).split("_")[0]

    img_path = str(data_dir / f"HE{main_id}.tif")
    img_he = load_image(img_path, verbose=True, downsize=DOWN_SIZE)

    img_path = str(data_dir / f"{main_id}_panorama.tif")
    img_pano = load_image(img_path, verbose=True, downsize=DOWN_SIZE)

    print(f"Data id {data_id} image size he:{img_he.shape} pano:{img_pano.shape}")

    spp = Spp(img_he)
    spp_cache_dir = Path(cache_dir / "he" / data_id)
    spp_cache_dir.mkdir(parents=True, exist_ok=True)
    spp.compute(num_feat=NUM_FEAT_DETECT, layers=5, cache_dir=spp_cache_dir)
    he_features = spp.features

    spp = Spp(img_pano)
    spp_cache_dir = Path(cache_dir / "pano" / data_id)
    spp_cache_dir.mkdir(parents=True, exist_ok=True)
    spp.compute(num_feat=NUM_FEAT_DETECT, layers=5, cache_dir=spp_cache_dir)
    pano_features = spp.features

    show_keypoints(
        img_he, he_features, str(cache_dir / "he" / f"{data_id}_keypoints_x4.tif")
    )
    show_keypoints(
        img_pano,
        pano_features,
        str(cache_dir / "pano" / f"{data_id}_keypoints_x4.tif"),
    )
    return he_features, pano_features


if __name__ == "__main__":
    if len(argv) != 3:
        print(f"Usage: {argv[0]} data_id_0 data_id_1")
        exit(-1)

    # get model
    model = net_siamese_global_feats(NUM_FEAT_INPUT, SPP_FEAT_LEN)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy()],
    )
    model.summary(expand_nested=True)

    # create dataset
    data_id_0, data_id_1 = argv[1], argv[2]
    cache_dir = Path("outputs") / "cache" / "dnn" / f"{data_id_0}-{data_id_1}-x4"
    output_dir = Path("outputs") / "dnn" / f"{data_id_0}-{data_id_1}-x4"

    dataset = (data_pipeline(data_id_0), data_pipeline(data_id_1))
    feat_len = len(dataset[0][0][0].desc)
    print(f"feature len {feat_len}")

    input_he_feat = np.zeros((SAMPLES * 4, NUM_FEAT_INPUT, feat_len))
    input_he_pose = np.zeros((SAMPLES * 4, NUM_FEAT_INPUT, 2))
    input_pano_feat = np.zeros((SAMPLES * 4, NUM_FEAT_INPUT, feat_len))
    input_pano_pose = np.zeros((SAMPLES * 4, NUM_FEAT_INPUT, 2))
    input_label = np.zeros((SAMPLES * 4,), dtype=np.int32)

    # create dataset by random sampling
    for data_id, (he_id, pano_id, label) in enumerate(
        [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    ):
        he_features, pano_features = dataset[he_id][0], dataset[pano_id][1]
        for sample_id in range(SAMPLES):
            input_label[SAMPLES * data_id + sample_id] = label

            sample_he = np.random.choice(
                range(NUM_FEAT_DETECT), size=NUM_FEAT_INPUT, replace=False
            )
            sample_pano = np.random.choice(
                range(NUM_FEAT_DETECT), size=NUM_FEAT_INPUT, replace=False
            )

            for idx in range(NUM_FEAT_INPUT):
                input_he_feat[SAMPLES * data_id + sample_id, idx, :] = he_features[
                    sample_he[idx]
                ].desc
                input_he_pose[SAMPLES * data_id + sample_id, idx, :] = he_features[
                    sample_he[idx]
                ].keypoint.pt
                input_pano_feat[SAMPLES * data_id + sample_id, idx, :] = pano_features[
                    sample_pano[idx]
                ].desc
                input_pano_pose[SAMPLES * data_id + sample_id, idx, :] = pano_features[
                    sample_pano[idx]
                ].keypoint.pt

    dataset_len = input_label.shape[0]
    print(f"dataset len {dataset_len}")
    dataset = tf.data.Dataset.zip(
        (
            tf.data.Dataset.zip(
                tuple(
                    [
                        tf.data.Dataset.from_tensor_slices(data)
                        for data in (
                            input_he_feat,
                            input_he_pose,
                            input_pano_feat,
                            input_pano_pose,
                        )
                    ]
                )
            ),
            tf.data.Dataset.from_tensor_slices(input_label),
        )
    )
    dataset = dataset.shuffle(buffer_size=dataset_len)
    dataset = dataset.batch(batch_size=BATCH_SIZE)

    model.fit(
        dataset,
        epochs=EPOCHS,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(output_dir / "siamese_model.h5")
    model.save(model_path)
    print(f"Save model to {model_path}")

    # model = keras.models.load_model(str(output_dir / "siamese_model.h5"))
    # point_feat_model = net_point_feature(NUM_FEAT_INPUT, FEAT_LEN)
    # point_feat_model.set_weights(
    #     model.get_layer("global_feat_model").get_layer("point_feat_model").get_weights()
    # )
    # point_feat_model.summary()
    # for weight in point_feat_model.get_weights():
    #     print(weight)
