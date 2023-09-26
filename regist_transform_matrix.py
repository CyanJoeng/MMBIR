from pathlib import Path
import pickle
from sys import argv
import os

import numpy as np
from tensorflow import keras
import tensorflow as tf

from dnn.model_parameters import *
from dnn.transform_pos import net_transform_pos, load_trained_trans_pos_net
from dnn.affine_transform_layer import AffineTransformLayer
from keypoints.feature_gspp import Gspp, GsppFeature
from keypoints.feature_spp import Spp, SppFeature
from keypoints.transform import sample_pix_with
from utils.dataset import load_image
from utils.display import show_keypoints, show_matches, show_overlay_img
from utils.score import calc_score


EPOCHS = 200
LR = 0.01


NO_CACHE = False
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

    img_path = str(data_dir / f"HE{main_id}.tif")
    mov_img, mov_features, mov_shape = detect_features(img_path, cache_dir, "mov")

    img_path = str(data_dir / f"{main_id}_panorama.tif")
    fix_img, fix_features, fix_shape = detect_features(img_path, cache_dir, "fix")

    return (mov_img, fix_img), (mov_features, fix_features), (mov_shape, fix_shape)


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


def create_train_dataset(mov_feats, fix_feats, mov_shape, fix_shape):
    feat_len = SPP_FEAT_LEN
    print(f"feature len {feat_len}")

    input_mov_feat = np.zeros((1, NUM_FEAT_INPUT, feat_len, NEIGHBOR))
    input_mov_pose = np.zeros((1, NUM_FEAT_INPUT, 2))
    input_fix_feat = np.zeros((1, NUM_FEAT_INPUT, feat_len, NEIGHBOR))
    input_fix_pose = np.zeros((1, NUM_FEAT_INPUT, 2))
    input_label = np.zeros((1,), dtype=np.int32)

    # create dataset by random sampling

    pano_sz = np.array(mov_shape)[:2][::-1]
    he_sz = np.array(fix_shape)[:2][::-1]

    for idx in range(NUM_FEAT_INPUT):
        set_desc(input_mov_feat[0, idx], mov_feats[idx])
        set_pos(input_mov_pose[0, idx], mov_feats[idx], pano_sz)

        set_desc(input_fix_feat[0, idx], fix_feats[idx])
        set_pos(input_fix_pose[0, idx], fix_feats[idx], he_sz)

    input_label[0] = 0

    inputs = (input_mov_feat, input_mov_pose, input_fix_feat, input_fix_pose)
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
    if "NO_CACHE" in os.environ:
        NO_CACHE = True

    # create dataset
    dataset_folder = argv[1]
    data_id = argv[2]
    output_dir = Path("outputs") / "dnn" / "trans" / f"{data_id}-x{DOWN_SIZE}"
    cache_dir = output_dir / "cache"

    # get model
    model = net_transform_pos(NEIGHBOR)
    model.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")
    model.summary(expand_nested=True)

    dataset_cache_path = cache_dir / method_str() / "dataset.pkl"
    if NO_CACHE or not dataset_cache_path.exists():
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
    sample_data = sample_pix_with(R, imgs)
    save_path = str(output_dir / "overlay.tif")
    show_overlay_img(imgs[1], sample_data, save_path)

    # model = load_trained_trans_pos_net(NEIGHBOR)
    # model.summary()

    # trans_weights = model.get_layer("trans_matrix").get_weights()
    # print("trans weights ", trans_weights)
