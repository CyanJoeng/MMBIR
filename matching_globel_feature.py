from dnn.global_feature import get_trained_global_feat_net
from keypoints.feature import feature_match
from keypoints.feature_spp import Spp
from utils.display import show_matches, show_keypoints
from utils.dataset import load_image

from sys import argv
from pathlib import Path
import numpy as np
import cv2


def get_image_features(image, cache_dir, label, num_feat=200):
    spp = Spp(image)
    spp.compute(num_feat=num_feat, cache_dir=(cache_dir / label))
    features = spp.features
    print(f"feature count: {label} {len(features)}")
    show_keypoints(image, features, str(cache_dir / f"keypoints_{label}.tif"))
    return features


def matching_points(img_left, img_right, feat_left, feat_right):
    matches = feature_match(
        feat_left, feat_right, filter_fun=Spp.refine_fun, cache_dir=str(cache_dir)
    )

    show_matches(
        img_left, img_right, matches, save_path=str(cache_dir / "feature_matches.tif")
    )


def img_block_selection(image, pre_block, step=0.1):
    x, y, w, h = pre_block
    img_h, img_w = image.shape[:2]
    if img_h > img_w:
        off_y = int(img_h * step)
        y = 0 if w == 0 else y + off_y
        x = 0
        w = img_w
        h = img_w
    else:
        off_x = int(img_w * step)
        x = 0 if w == 0 else x + off_x
        y = 0
        w = img_h
        h = img_h

    return x, y, w, h


def matching_global_block(global_feature, block_feature, model):
    num_feat_input = len(global_feature)
    feat_len = global_feature[0].desc.shape[0]

    input_feat = np.zeros((2, num_feat_input, feat_len))
    input_pose = np.zeros((2, num_feat_input, 2))

    # create dataset by random sampling
    for data_id, data in enumerate([global_feature, block_feature]):
        for feat_id in range(num_feat_input):
            input_feat[data_id, feat_id, :] = data[feat_id].desc
            input_pose[data_id, feat_id, :] = data[feat_id].keypoint.pt

    feat_output = model.predict((input_feat, input_pose))
    feat_output = np.array(feat_output)

    return np.linalg.norm(feat_output[0] - feat_output[1])


def main(image, cache_dir, num_feat=200):
    img_h, img_w = image.shape[:2]
    global_feat = get_image_features(image, cache_dir, "global", num_feat)

    feat_net = get_trained_global_feat_net(num_feat)
    feat_net.summary()

    step = min(img_h, img_w) / max(img_h, img_w) / 2
    block = (0, 0, 0, 0)
    idx = -1
    block_scores = []
    while (block[0] + block[2]) <= img_w and (block[1] + block[3]) <= img_h:
        block = img_block_selection(image, block, step)
        idx += 1
        x, y, w, h = block

        print(f"block {idx} {block}")

        block_img = image[y : y + h, x : x + w]
        block_feat = get_image_features(block_img, cache_dir, f"block_{idx}", num_feat)
        score = matching_global_block(global_feat, block_feat, feat_net)
        block_scores.append((block, score))

        cv2.imwrite(str(Path(cache_dir) / f"img_block{idx}.tif"), block_img)

    [print(idx, bs[1]) for idx, bs in enumerate(block_scores)]
    with open(str(Path(cache_dir) / "scores.txt"), "w") as f:
        for line in [f"{block}, {score}\n" for block, score in block_scores]:
            f.write(line)
    print("write scores to ", f"{cache_dir}/scores.txt")


if __name__ == "__main__":
    if len(argv) != 3:
        print(f"Usage: {argv[0]} dataset_path data_id")
        exit(-1)

    DOWN_SIZE = 8

    dataset_path = argv[1]
    data_id = argv[2]
    main_id = data_id.split("_")[0]
    cache_dir = Path("outputs") / "g_feat" / "slide" / data_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "he").mkdir(parents=True, exist_ok=True)
    (cache_dir / "pano").mkdir(parents=True, exist_ok=True)

    img_he_path = Path(dataset_path) / "H&E_IMC" / "Pair" / data_id / f"HE{main_id}.tif"
    img_he = load_image(str(img_he_path), downsize=DOWN_SIZE)
    print(f"he shape {img_he.shape}")
    img_pano_path = (
        Path(dataset_path) / "H&E_IMC" / "Pair" / data_id / f"{main_id}_panorama.tif"
    )
    img_pano = load_image(str(img_pano_path), downsize=DOWN_SIZE)
    print(f"pano shape {img_pano.shape}")

    main(img_he, (cache_dir / "he"))
    # main(img_pano, (cache_dir / "pano"))
