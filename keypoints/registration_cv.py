from typing import List, Tuple, overload
from feature import Feature, PointFeature
from feature_orb import Orb
from feature_sift import Sift
from feature_spp import Spp
from transform import trans_image_by
from transform import find_trans_matrix
from transform import filter_by_fundamental
from dataset import read_dataset_folder, load_image
import numpy as np
from pathlib import Path
import argparse
import cv2


def get_method(name: str):
    return {"orb": Orb, "sift": Sift, "spp": Spp}[name]


def feature_detect(
    img, method: Orb | Sift | Spp, num_feat, is_he=False, verbose=False
) -> List[Feature]:
    if verbose:
        print(f"feature detect {img.shape}")

    if method is Spp and not is_he:
        print("feature on non-he")
        num_feat = None

    detector = method(img)
    detector.compute(num_feat=num_feat)

    assert detector.features is not None
    return detector.features


def match(
    feats_moving: List[Feature],
    feats_fixed: List[Feature],
    count: int,
    method: Orb | Sift,
    verbose=False,
) -> List[Tuple[Feature]]:
    if verbose:
        print("feature match")

    pairs = method.match(feats_moving, feats_fixed, top_count=count)
    return pairs


def show_keypoints(
    img: np.ndarray, keypoints: List[PointFeature], save_path: str = None
):
    scale = 1

    img_show = cv2.resize(img, np.array(img.shape[:2])[::-1] // scale)
    for kp in keypoints:
        pt = kp.keypoint.pt
        pt = np.array(pt).astype(np.int32) // scale
        img_show = cv2.circle(img_show, pt, 3, (255, 0, 0), 1)

    if save_path is not None:
        print("keypoints save path: ", save_path)
        cv2.imwrite(save_path, img_show)


def show_matches(
    img_moving: np.ndarray,
    img_fixed: np.ndarray,
    matches: List[Tuple[PointFeature]],
    save_path: str,
    verbose=False,
) -> int:
    assert img_moving.shape[2] == img_fixed.shape[2]
    assert img_moving.dtype == img_fixed.dtype and img_fixed.dtype == np.uint8

    scale = 1

    img_moving = cv2.resize(img_moving, np.array(img_moving.shape[:2])[::-1] // scale)
    img_fixed = cv2.resize(img_fixed, np.array(img_fixed.shape[:2])[::-1] // scale)

    new_h = max(img_moving.shape[0], img_fixed.shape[0])
    new_w = img_moving.shape[1] + img_fixed.shape[1]
    new_c = img_moving.shape[2]

    img_show = np.zeros((new_h, new_w, new_c), dtype=np.uint8)
    img_show[: img_moving.shape[0], : img_moving.shape[1], :] = img_moving
    img_show[: img_fixed.shape[0], img_moving.shape[1] :, :] = img_fixed

    for match in matches:
        pt0, pt1 = match[0].keypoint.pt, match[1].keypoint.pt
        pt0, pt1 = (
            np.array(pt0).astype(np.int32) // scale,
            np.array(pt1).astype(np.int32) // scale,
        )
        pt1[0] += img_moving.shape[1]

        img_show = cv2.line(img_show, pt0, pt1, (255, 255, 0), 1)

    is_exit = False
    if verbose:
        print("show matches")
        cv2.imshow("matches", img_show)
        code = cv2.waitKey()
        print(code)
        is_exit = code == 113  # 'q'

    print("save path: ", save_path)
    cv2.imwrite(save_path, img_show)


def show_overlay(img_fixed: np.ndarray, overlay_pts: np.ndarray, save_path: str):
    h, w, _ = img_fixed.shape
    print(show_overlay)

    overlay = np.zeros((h, w + w, 3), np.uint8)
    overlay[:, w:, :] = img_fixed
    for data in overlay_pts:
        x, y, b, g, r = data
        x += 0.5
        y += 0.5
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        x, y = int(x), int(y)

        overlay[y, x] = np.array([b, g, r], np.uint8)

    print("save path: ", save_path)
    cv2.imwrite(save_path, overlay)


def registration_pipeline(img_path: Tuple[str], args):
    verbose = args.verbose
    method = get_method(args.method)
    save_dir = Path("outputs") / args.method / Path(img_path[0]).parts[-2]

    save_dir.mkdir(parents=True, exist_ok=True)

    img_moving = load_image(img_path[0], verbose, downsize=args.downsize)
    print("img_moving shape ", img_moving.shape)
    img_fixed = load_image(img_path[1], verbose, downsize=args.downsize)
    print("img_fixed shape ", img_fixed.shape)

    feat_moving = feature_detect(
        img_moving, method, num_feat=args.features, verbose=verbose
    )
    print(f"feat_moving len {len(feat_moving)}")
    show_keypoints(img_moving, feat_moving, str(save_dir / "keypoints_pano.tif"))

    feat_fixed = feature_detect(
        img_fixed, method, is_he=True, num_feat=args.features, verbose=verbose
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
        choices=["orb", "sift", "spp"],
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
