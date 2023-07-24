import argparse
from pathlib import Path
from typing import List, Tuple, overload

import cv2
from matplotlib import pyplot as plt
import numpy as np

from utils.dataset import load_image, read_dataset_folder
from keypoints.feature import Feature, PointFeature, feature_match
from keypoints.feature_gspp import Gspp
from keypoints.feature_orb import Orb
from keypoints.feature_sift import Sift
from keypoints.feature_spp import Spp
from keypoints.transform import filter_by_fundamental, find_trans_matrix, trans_image_by

from utils.display import *


def get_method(name: str = ""):
    methods = {"orb": Orb, "sift": Sift, "spp": Spp, "gspp": Gspp}
    return methods[name] if name != "" else methods.keys()


def feature_detect(
    img, method: Orb | Sift | Spp, num_feat, is_he=False, verbose=False, cache_dir=""
) -> List[Feature]:
    if verbose:
        print(f"feature detect {img.shape}")

    if method is Spp and not is_he:
        print("feature on non-he")
        num_feat = None

    detector = method(img)
    detector.compute(num_feat=num_feat, cache_dir=cache_dir)

    assert detector.features is not None
    return detector.features


def match(
    feats_moving: List[Feature],
    feats_fixed: List[Feature],
    count: int,
    method: None,
    verbose=False,
    cache_dir="",
) -> List[Tuple[Feature]]:
    if verbose:
        print("feature match")
        print(f"\ttop count {count}")

    pairs = method.match(
        feats_moving,
        feats_fixed,
        top_count=count,
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


def registration_pipeline(img_path: Tuple[str], args):
    verbose = args.verbose
    method = get_method(args.method)
    data_id = Path(img_path[0]).parts[-2]
    save_dir = Path("outputs") / args.method / data_id

    save_dir.mkdir(exist_ok=True)

    img_moving = load_image(img_path[0], verbose, downsize=args.downsize)
    print("img_moving shape ", img_moving.shape)
    img_fixed = load_image(img_path[1], verbose, downsize=args.downsize)
    print("img_fixed shape ", img_fixed.shape)

    feat_moving = feature_detect(
        img_moving,
        method,
        num_feat=args.features,
        verbose=verbose,
    )
    print(f"feat_moving len {len(feat_moving)}")
    show_keypoints(img_moving, feat_moving, str(save_dir / "keypoints_pano.tif"))

    feat_fixed = feature_detect(
        img_fixed,
        method,
        is_he=True,
        num_feat=args.features,
        verbose=verbose,
    )
    print(f"feat_fixed len {len(feat_moving)}")
    show_keypoints(img_fixed, feat_fixed, str(save_dir / "keypoints_fixed.tif"))

    if verbose:
        print("count  ", args.count)
    matches = match(feat_moving, feat_fixed, args.count, method, verbose)

    save_path = str(save_dir / "matches.tif")
    is_exit = show_matches(img_moving, img_fixed, matches, save_path, verbose=verbose)

    trans_matrix, refined_matches = filter_by_fundamental(matches)

    save_path = str(save_dir / "matches_homo.tif")
    is_exit = show_matches(
        img_moving, img_fixed, refined_matches, save_path, verbose=verbose
    )

    transed_data = trans_image_by(trans_matrix, img_moving)
    save_path = str(save_dir / "overlay.tif")
    show_overlay(img_fixed, transed_data, save_path)

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
