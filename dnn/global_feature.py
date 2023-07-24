from tensorflow import keras
import tensorflow as tf


def net_global_feature(num_keypoints: int, feat_len: int):
    """
    input feat shape BxNxF
        B: batch size
        N: number of keypoints in a image
        F: length of keypoint feature
    input pos shape BxNx2
        two dimensions in image coordinate
    """

    N, F = num_keypoints, feat_len
    input_feat = keras.Input(shape=(N, F))
    input_pos = keras.Input(shape=(N, 2))

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
        ]
    )(input)
    output_img_feat = keras.Sequential(
        [
            keras.layers.MaxPool2D(pool_size=(N, 1)),
            keras.layers.Reshape((-1,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(64, activation="relu"),
        ]
    )(output_point_feat)

    net = keras.Model(inputs=(input_feat, input_pos), outputs=output_img_feat)
    return net


def net_siamese_global_feats(num_keypoints: int, feat_len: int):
    N, F = num_keypoints, feat_len

    input_0_feat = keras.Input(shape=(N, F))
    input_0_pos = keras.Input(shape=(N, 2))
    input_1_feat = keras.Input(shape=(N, F))
    input_1_pos = keras.Input(shape=(N, 2))

    global_feat_model = net_global_feature(num_keypoints, feat_len)

    out_0 = global_feat_model((input_0_feat, input_0_pos))
    out_1 = global_feat_model((input_1_feat, input_1_pos))
    output = keras.layers.Lambda(lambda x: tf.square(x[0] - x[1]))([out_0, out_1])
    output = keras.layers.Dense(1, activation="sigmoid")(output)

    siamese_model = keras.Model(
        inputs=[input_0_feat, input_0_pos, input_1_feat, input_1_pos], outputs=output
    )
    return siamese_model


if __name__ == "__main__":
    from sys import argv, path as sys_path
    from pathlib import Path
    import numpy as np

    sys_path.insert(0, str(Path(sys_path[0]).parent))

    from utils.dataset import load_image
    from utils.display import show_keypoints
    from keypoints.feature_spp import Spp

    if len(argv) != 3:
        print(f"Usage: {argv[0]} data_id_0 data_id_1")
        exit(-1)

    DOWN_SIZE = 4
    NUM_FEAT_DETECT = 400
    NUM_FEAT_INPUT = 200
    EPOCHS = 20
    SAMPLES = 5
    BATCH_SIZE = 2
    FEAT_LEN = 340

    # get model
    model = net_siamese_global_feats(NUM_FEAT_INPUT, FEAT_LEN)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.AUC()],
    )

    # create dataset
    data_id_0, data_id_1 = argv[1], argv[2]
    cache_dir = Path("outputs") / "cache" / "dnn" / f"{data_id_0}-{data_id_1}-x4"

    def data_pipeline(data_id):
        data_dir = (
            Path(argv[0]).parent.parent / "datasets" / "H&E_IMC" / "Pair" / data_id
        )
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
    # dataset_x = np.array(
    #     [
    #         (
    #             input_he_feat[idx],
    #             input_he_pose[idx],
    #             input_pano_feat[idx],
    #             input_pano_pose[idx],
    #         )
    #         for idx in range(SAMPLES * 4)
    #     ]
    # )
    # dataset_y = input_label

    model.fit(dataset, epochs=EPOCHS)
