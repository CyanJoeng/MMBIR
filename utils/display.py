import cv2
import numpy as np
from typing import List, Tuple

from keypoints.feature import PointFeature


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
