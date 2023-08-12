from pathlib import Path
import pickle
from sys import argv
import os

import numpy as np
from tensorflow import keras
import tensorflow as tf
from affine_transform_layer import AffineTransformLayer

from dnn.model_parameters import *
from dnn.transform_pos import net_transform_pos, load_trained_trans_pos_net
from keypoints.feature_gspp import Gspp, GsppFeature
from keypoints.feature_spp import Spp, SppFeature
from keypoints.transform import trans_image_by
from utils.dataset import load_image
from utils.display import show_keypoints, show_matches, show_trans_img
from utils.score import calc_score


EPOCHS = 200
LR = 0.01


DOWN_SIZE = 4

FEATURE_METHOD = Spp

if FEATURE_METHOD is Spp:
    NEIGHBOR = 1
else:
    NEIGHBOR = GsppFeature.N_EDGE


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
        img, features, str(cache_dir / label / f"keypoints_x{DOWN_SIZE}.tif")
    )
    return img, features, img.shape


def data_pipeline(dataset_folder, data_id, cache_dir):
    data_dir = Path(dataset_folder) / "H&E_IMC" / "Pair" / data_id
    data_dir = Path(data_dir).absolute()
    main_id = str(data_id).split("_")[0]

    cache_dir = Path(cache_dir / method_str() / data_id)
    cache_dir.mkdir(parents=True, exist_ok=True)

    img_path = str(data_dir / f"{main_id}_panorama.tif")
    pano_img, pano_features, pano_shape = detect_features(img_path, cache_dir, "pano")

    img_path = str(data_dir / f"HE{main_id}.tif")
    he_img, he_features, he_shape = detect_features(img_path, cache_dir, "he")

    return (pano_img, he_img), (pano_features, he_features), (pano_shape, he_shape)


def set_desc(data, feature):
    if FEATURE_METHOD is Spp:
        data[:, 0] = feature.desc
    else:
        for idx in range(NEIGHBOR):
            data[:, idx] = feature.desc[idx]["desc"]


def set_pos(data, feature, scale=1):
    center = np.array(feature.keypoint.pt)
    # data[:] = center
    data[:] = center / scale * 2 - 1


def create_train_dataset(pano_features, he_features, pano_shape, he_shape):
    feat_len = SPP_FEAT_LEN
    print(f"feature len {feat_len}")

    input_pano_feat = np.zeros((1, NUM_FEAT_INPUT, feat_len, NEIGHBOR))
    input_pano_pose = np.zeros((1, NUM_FEAT_INPUT, 2))
    input_he_feat = np.zeros((1, NUM_FEAT_INPUT, feat_len, NEIGHBOR))
    input_he_pose = np.zeros((1, NUM_FEAT_INPUT, 2))
    input_label = np.zeros((1,), dtype=np.int32)

    # create dataset by random sampling

    pano_sz = np.array(pano_shape)[:2][::-1]
    he_sz = np.array(he_shape)[:2][::-1]

    for idx in range(NUM_FEAT_INPUT):
        set_desc(input_pano_feat[0, idx], pano_features[idx])
        set_pos(input_pano_pose[0, idx], pano_features[idx], pano_sz)

        set_desc(input_he_feat[0, idx], he_features[idx])
        set_pos(input_he_pose[0, idx], he_features[idx], he_sz)

    input_label[0] = 0

    inputs = (input_pano_feat, input_pano_pose, input_he_feat, input_he_pose)
    dataset = (inputs, input_label)

    return dataset


if __name__ == "__main__":
    if len(argv) != 3:
        print(f"Usage: {argv[0]} dataset_folder data_id")
        exit(-1)

    if "DOWN_SIZE" in os.environ:
        DOWN_SIZE = int(os.environ.get("DOWN_SIZE"))
    if "LR" in os.environ:
        LR = float(os.environ.get("LR"))
    if "EPOCHS" in os.environ:
        EPOCHS = int(os.environ.get("EPOCHS"))

    # create dataset
    dataset_folder = argv[1]
    data_id = argv[2]
    cache_dir = Path("outputs") / "cache" / "dnn" / "trans" / f"{data_id}-x{DOWN_SIZE}"
    output_dir = Path("outputs") / "dnn" / "trans" / f"{data_id}-x{DOWN_SIZE}"

    # get model
    model = net_transform_pos(NEIGHBOR)
    model.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")
    model.summary(expand_nested=True)

    dataset_cache_path = cache_dir / method_str() / "dataset.pkl"
    if not dataset_cache_path.exists():
        imgs, feats, shapes = data_pipeline(dataset_folder, data_id, cache_dir)

        with open(str(dataset_cache_path), "wb") as f:
            pickle.dump({"imgs": imgs, "feats": feats, "shapes": shapes}, f)
    else:
        with open(str(dataset_cache_path), "rb") as f:
            pack = pickle.load(f)
        imgs, feats, shapes = pack["imgs"], pack["feats"], pack["shapes"]

    if "SCORE" in os.environ:
        with open(str(output_dir / "trans2d_r_t.pkl"), "rb") as f:
            print(
                "load transformation matrix from ", str(output_dir / "trans2d_r_t.pkl")
            )
            trans = pickle.load(f)

        R = np.eye(3, dtype=np.float32)
        R[:2, :2] = trans["r"]
        R[2:, :2] = trans["t"]
        # R = trans["r"]

        print("transform \n", R)

        score = calc_score(imgs[0], imgs[1], R)
        print(f"score: {score}")
        exit(0)

    dataset_raw = create_train_dataset(feats[0], feats[1], shapes[0], shapes[1])
    dataset_len = dataset_raw[1].shape[0]
    print(f"dataset len {dataset_len}")

    dataset_train = tf.data.Dataset.zip(
        (
            tf.data.Dataset.zip(
                tuple(
                    [
                        tf.data.Dataset.from_tensor_slices(data)
                        for data in dataset_raw[0]
                    ]
                )
            ),
            tf.data.Dataset.from_tensor_slices(dataset_raw[1]),
        )
    )
    dataset_train = dataset_train.batch(batch_size=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(output_dir / f"trans_pos_model_n{NEIGHBOR}.h5")
    model_save_cb = keras.callbacks.ModelCheckpoint(
        model_path, "loss", save_best_only=True, mode="min"
    )
    stop_cb = keras.callbacks.EarlyStopping(
        "loss", min_delta=0.5, mode="min", patience=3, start_from_epoch=20
    )

    model.fit(
        dataset_train,
        epochs=EPOCHS,
        callbacks=[
            model_save_cb,
            stop_cb,
        ],
    )

    # model.save(model_path)
    print(f"Save model to {model_path}")

    trans = AffineTransformLayer.get_trans_matrix(model)

    with open(str(output_dir / "trans2d_r_t.pkl"), "wb") as f:
        pickle.dump(
            trans,
            f,
        )
    print("save transformation matrix to ", str(output_dir / "trans2d_r_t.pkl"))

    R = np.eye(3, dtype=np.float32)
    R[:2, :2] = trans["r"]
    R[2:, :2] = trans["t"]
    # R = trans["r"]

    print("transform \n", R)
    transed_data = trans_image_by(R, imgs)
    save_path = str(output_dir / "overlay.tif")
    show_trans_img(imgs[1], transed_data, save_path)

    # model = load_trained_trans_pos_net(NEIGHBOR)
    # model.summary()

    # trans_weights = model.get_layer("trans_matrix").get_weights()
    # print("trans weights ", trans_weights)
