from typing import List, Tuple
from feature import Feature, PointFeature
from feature_orb import Orb
from dataset import read_dataset_folder, load_image
import numpy as np
import argparse
import cv2


def feature_detect(img, is_he=False, verbose=False) -> List[Feature]:
    if verbose:
        print(f"feature detect {img}")

    orb = Orb(img)
    orb.compute()

    assert orb.features is not None
    return orb.features


def match(
    feats_moving: List[Feature], feats_fixed: List[Feature], verbose=False
) -> List[Tuple[Feature]]:
    if verbose:
        print("feature match")
    pairs = Orb.match(feats_moving, feats_fixed)
    return pairs


def show_matches(
    img_moving: np.ndarray,
    img_fixed: np.ndarray,
    matches: List[Tuple[PointFeature]],
) -> int:
    assert img_moving.shape[2] == img_fixed.shape[2]
    assert img_moving.dtype == img_fixed.dtype and img_fixed.dtype == np.uint8

    new_h = max(img_moving.shape[0], img_fixed.shape[0])
    new_w = img_moving.shape[1] + img_fixed.shape[1]
    new_c = img_moving.shape[2]

    img_show = np.zeros((new_h, new_w, new_c), dtype=np.uint8)
    img_show[: img_moving.shape[0], : img_moving.shape[1], :] = img_moving
    img_show[: img_fixed.shape[0], img_moving.shape[1] :, :] = img_fixed

    for match in matches:
        pt0, pt1 = match[0].keypoint.pt, match[1].keypoint.pt
        pt0, pt1 = np.array(pt0).astype(np.int32), np.array(pt1).astype(np.int32)
        pt1[0] += img_moving.shape[1]

        img_show = cv2.line(img_show, pt0, pt1, (255, 255, 0), new_h // 1000)

    cv2.imshow("matches", img_show)
    return cv2.waitKey()


def registration_pipeline(img_path: Tuple[str], verbose=False):
    img_moving = load_image(img_path[0], verbose)
    img_fixed = load_image(img_path[1], verbose)

    feat_moving = feature_detect(img_moving, verbose)
    feat_fixed = feature_detect(img_fixed, verbose)

    matches = match(feat_moving, feat_fixed, verbose)

    is_exit = False
    if verbose:
        print("show matches")
        code = show_matches(img_moving, img_fixed, matches)
        print(code)
        is_exit = code == 113  # 'q'
    return is_exit


def helper():
    parser = argparse.ArgumentParser(
        prog="Reg_CV", description="Registration with OpenCV"
    )

    parser.add_argument("dataset_folder")
    parser.add_argument("-c", "--count", help="count of matches")
    parser.add_argument(
        "-v",
        "--verbose",
        default=True,
        action="store_true",
        help="show interim informations",
    )

    args = parser.parse_args()
    return args


def main(args):
    for data in read_dataset_folder(args.dataset_folder, verbose=args.verbose):
        is_exit = registration_pipeline(data, args.verbose)
        if is_exit:
            break


if __name__ == "__main__":
    args = helper()
    main(args)
