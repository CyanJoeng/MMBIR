import argparse
from pathlib import Path
import pickle
from typing import List, Tuple, overload

import numpy as np

from dnn.global_feature import get_trained_point_feat_net
from dnn.model_parameters import SPP_FEAT_LEN
from dnn.transform_match import calc_trans_matrix_by_matches
from keypoints.feature import Feature, PointFeature, feature_match
from keypoints.feature_gspp import Gspp, GsppFeature
from keypoints.feature_orb import Orb
from keypoints.feature_sift import Sift
from keypoints.feature_spp import Spp, SppFeature
from keypoints.transform import (
    calc_trans_matrix_by_lstsq,
    filter_by_fundamental,
    filter_by_homography,
    find_trans_matrix,
    matches_to_pts,
    sample_pix_with,
    unify_img_pts,
)
from matplotlib import pyplot as plt
from utils.dataset import load_image, read_dataset_folder
from utils.display import *


def get_method(name: str = ""):
    methods = {"orb": Orb, "sift": Sift, "spp": Spp, "gspp": Gspp}
    return methods[name] if name != "" else methods.keys()


def get_num_neighbor(name: str = ""):
    methods = {"orb": 1, "sift": 1, "spp": 1, "gspp": GsppFeature.N_EDGE}
    return methods[name] if name != "" else methods.keys()


def feature_detect(
    img, method: Orb | Sift | Spp, num_feat, is_he=False, verbose=False, cache_dir=""
) -> List[Feature]:
    if verbose:
        print(f"feature detect {img.shape}")

    # if method is Spp and not is_he:
    #     print("feature on non-he")
    #     num_feat = None

    detector = method(img)
    detector.compute(num_feat=num_feat, cache_dir=cache_dir)

    assert detector.features is not None
    return detector.features


def match(
    feats_moving: List[Feature],
    feats_fixed: List[Feature],
    count: int,
    method: None,
    refine=False,
    verbose=False,
    cache_dir="",
) -> List[Tuple[Feature]]:
    if verbose:
        print("feature match")
        print(f"\ttop count {count}")

    filter_fun = None
    if refine:
        filter_fun = method.refine_fun

    pairs = method.match(
        feats_moving,
        feats_fixed,
        top_count=count,
        filter_fun=filter_fun,
        cache_dir=cache_dir,
    )
    distances = np.array([m[2] for m in pairs])
    print(
        f"\tmatch distance min {distances.min()} max {distances.max()} mean {distances.mean()} medium {np.median(distances)}"
    )
    plt.hist(distances)
    if cache_dir != "":
        print(f"\tmatch save distribution under {cache_dir}")
        plt.savefig(Path(cache_dir) / "match_dist_distribution.pdf", format="pdf")
    if verbose:
        plt.show()
    return pairs


def point_feat_dnn_trans(
    feats_moving: List[Feature],
    feats_fixed: List[Feature],
    method: Orb | Sift | Spp | Gspp,
    num_neighbor,
):
    len_mov, len_fix = len(feats_moving), len(feats_fixed)
    assert len_mov == len_fix
    num_feat_input = len_mov
    feat_len = SPP_FEAT_LEN

    input_feat = np.zeros((2, num_feat_input, feat_len, num_neighbor))
    input_pose = np.zeros((2, num_feat_input, 2, num_neighbor))

    def set_desc(data, feature):
        if method is Spp:
            data[:, 0] = feature.desc
        else:
            for idx in range(num_neighbor):
                data[:, idx] = feature.desc[idx]["desc"]

    def set_pos(data, feature, scale=1):
        if method is Spp:
            center = np.array(feature.keypoint.pt)
            data[:, 0] = center / scale * 2 - 1
        else:
            center = np.array(feature.keypoint.pt)
            for idx in range(num_neighbor):
                off = np.array(feature.desc[idx]["vec"])
                data[:, idx] = (off + center) / scale * 2 - 1

    # create dataset by random sampling
    for data_id, data in enumerate([feats_moving, feats_fixed]):
        for feat_id in range(num_feat_input):
            set_desc(input_feat[data_id, feat_id], data[feat_id])
            set_pos(input_pose[data_id, feat_id], data[feat_id])

    model = get_trained_point_feat_net(num_feat_input, num_neighbor)
    feat_output = model.predict((input_feat, input_pose))
    feat_output = np.array(feat_output)

    new_feat_moving = [
        SppFeature(feats_moving[feat_id].keypoint.pt, feat_output[0, feat_id])
        for feat_id in range(num_feat_input)
    ]
    new_feat_fixed = [
        SppFeature(feats_fixed[feat_id].keypoint.pt, feat_output[1, feat_id])
        for feat_id in range(num_feat_input)
    ]
    return new_feat_moving, new_feat_fixed


def get_matched_dnn_features(
    matches: List[Tuple[PointFeature, PointFeature, float]], he_img_shape
):
    matches_len = len(matches)
    print(f"\tget matched dnn features  len {np.array(matches[0][0].desc).shape}")
    desc_len = np.array(matches[0][0].desc).shape[1]

    pano_pt = np.zeros((matches_len, 2), dtype=np.float32)
    pano_desc = np.zeros((matches_len, desc_len), dtype=np.float32)

    he_pt = np.zeros((matches_len, 2), dtype=np.float32)
    he_desc = np.zeros((matches_len, desc_len), dtype=np.float32)

    img_sz = np.array(he_img_shape)[:2][::-1]

    for idx, (pano, he, _) in enumerate(matches):
        pano_pt[idx] = np.array(pano.keypoint.pt).flatten() / img_sz * 2 - 1

        pano_desc[idx] = np.array(pano.desc).flatten()

        he_pt[idx] = np.array(he.keypoint.pt).flatten() / img_sz * 2 - 1
        he_desc[idx] = np.array(he.desc).flatten()

    return ((pano_pt, pano_desc), (he_pt, he_desc))


def cache_matched_dnn_features(
    matched_features,
    data_id: str,
    method: str,
):
    cache_path = Path("outputs") / "cache" / "dnn" / data_id / f"{method}_features.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_data = {
        "pano": {
            "pt": matched_features[0][0],
            "desc": matched_features[0][1],
        },
        "he": {
            "pt": matched_features[1][0],
            "desc": matched_features[1][1],
        },
    }

    with open(str(cache_path), "wb") as f:
        pickle.dump(cache_data, f)
    print("cache features to ", str(cache_path))


def registration_pipeline(img_path, args):
    verbose = args.verbose
    method = get_method(args.method)
    data_id = Path(img_path[0]).parts[-2]
    save_dir = Path("outputs") / args.method / data_id
    dnn_dir = Path("outputs") / "dnn"

    Path(save_dir).mkdir(exist_ok=True, parents=True)

    def detect(path, label: str):
        print("Feature detect on img", path)
        img = load_image(path, verbose, downsize=args.downsize)
        print(f"img {label} shape ", img.shape)
        point_features = feature_detect(
            img,
            method,
            num_feat=args.features,
            verbose=verbose,
        )
        print(f"detected point feature on {label}: count {len(point_features)}")
        show_keypoints(img, point_features, str(save_dir / f"keypoints_{label}.tif"))
        return img, point_features

    mov_id, mov_label = 1, "he"
    fix_id, fix_label = 0, "pano"

    img_mov, feat_mov = detect(img_path[mov_id], mov_label)
    img_fix, feat_fix = detect(img_path[fix_id], fix_label)

    matches = match(
        feat_mov,
        feat_fix,
        args.count,
        method,
        # refine=True,
        verbose=verbose,
        cache_dir=str(save_dir),
    )
    print("matched (pano-he) count  ", len(matches))

    save_path = str(save_dir / "matches.tif")
    is_exit = show_matches(img_mov, img_fix, matches, save_path, verbose=verbose)

    trans_matrix, homo_matches = filter_by_homography(matches)

    save_path = str(save_dir / "matches_homo.tif")
    is_exit = show_matches(img_mov, img_fix, homo_matches, save_path, verbose=verbose)

    pts_mov, pts_fix = matches_to_pts(homo_matches)
    pts_mov = unify_img_pts(pts_mov, img_mov.shape)
    pts_fix = unify_img_pts(pts_fix, img_fix.shape)

    trans_matrix = calc_trans_matrix_by_lstsq(pts_mov, pts_fix)

    DNN = False
    if DNN:
        # dnn refine desc
        feat_mov, feat_fix = point_feat_dnn_trans(
            feat_mov, feat_fix, method, get_num_neighbor(args.method)
        )
        print(f"feat len -> {feat_mov[0].desc.shape}")

        # match new desc with Spp matching method
        save_dir = save_dir / "dnn"
        save_dir.mkdir(exist_ok=True)
        matches = match(
            feat_mov,
            feat_fix,
            args.count,
            Spp,
            verbose=verbose,
            cache_dir=str(save_dir),
        )
        print("dnn refined matched count  ", len(matches))

        save_path = str(save_dir / "matches.tif")
        is_exit = show_matches(img_mov, img_fix, matches, save_path, verbose=verbose)

        matched_pts_feat = get_matched_dnn_features(matches, img_fix.shape)
        if args.cache_feature:
            cache_matched_dnn_features(matched_pts_feat, data_id, args.method)

        # trans_matrix = calc_trans_matrix_by_matches(
        #     matched_features[0], matched_features[1]
        # )
        # trans_matrix, homo_matches = find_trans_matrix(
        #     matched_features[0], matched_features[1]
        # )
        trans_matrix = calc_trans_matrix_by_lstsq(
            matched_pts_feat[0][0], matched_pts_feat[1][0]
        )
        # trans_weight_path = str(dnn_dir / "trans" / "trans2d_r_t.pkl")
        # with open(trans_weight_path, "rb") as f:
        #     trans_matrix = pickle.load(f)

    transed_data = sample_pix_with(trans_matrix, (img_mov, img_fix))
    save_path = str(save_dir / "overlay.tif")
    show_overlay_img(img_fix, transed_data, save_path)

    return is_exit


def helper():
    parser = argparse.ArgumentParser(
        prog="Reg_CV", description="Registration with OpenCV"
    )

    parser.add_argument("dataset_folder")
    parser.add_argument(
        "-f", "--features", type=int, default=300, help="count of features"
    )
    parser.add_argument("-c", "--count", type=int, default=30, help="count of matches")
    parser.add_argument(
        "--downsize", type=int, default=1, help="downsize original images"
    )
    parser.add_argument(
        "-m",
        "--method",
        default="orb",
        choices=get_method(),
        help="count of matches",
    )
    parser.add_argument(
        "-s", "--sample", default=None, type=str, help="samples to be used"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="show interim informations",
    )
    parser.add_argument(
        "--cache_feature",
        default=False,
        action="store_true",
        help="cache dnn feature for training",
    )

    args = parser.parse_args()
    return args


def main(args):
    for data in read_dataset_folder(
        args.dataset_folder, sample_name=args.sample, verbose=args.verbose
    ):
        is_exit = registration_pipeline(data, args)
        if is_exit:
            break


if __name__ == "__main__":
    args = helper()
    main(args)
