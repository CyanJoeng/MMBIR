from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple

from keypoints.feature import PointFeature


def show_keypoints(
    img: np.ndarray, keypoints: List[PointFeature], save_path: str = None
):
    print("show keypoints")
    scale = 1

    radius = max(min(img.shape[0], img.shape[1]) // 100, 1)
    thin = 2 if radius > 4 else 1

    img_show = cv2.resize(img, np.array(img.shape[:2])[::-1] // scale)
    for kp in keypoints:
        pt = kp.keypoint.pt
        pt = np.array(pt).astype(np.int32) // scale
        img_show = cv2.circle(img_show, pt, radius, (255, 0, 0), thin)

    print(f"\tkeypoints image shape {img_show.shape}")
    print(f"\tkeypoints count {len(keypoints)}")
    if save_path is not None:
        print("\tkeypoints save path: ", save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, img_show)


def show_matches(
    img_pano: np.ndarray,
    img_he: np.ndarray,
    matches: List[Tuple[PointFeature]],
    save_path: str,
    display_count: int = 50,
    verbose=False,
) -> int:
    """
    pano image is shows on the left
    """
    assert img_pano.shape[2] == img_he.shape[2]
    assert img_pano.dtype == img_he.dtype and img_he.dtype == np.uint8

    scale = 1

    img_pano = cv2.resize(img_pano, np.array(img_pano.shape[:2])[::-1] // scale)
    img_he = cv2.resize(img_he, np.array(img_he.shape[:2])[::-1] // scale)

    new_h = max(img_pano.shape[0], img_he.shape[0])
    new_w = img_pano.shape[1] + img_he.shape[1]
    new_c = img_pano.shape[2]

    img_show = np.zeros((new_h, new_w, new_c), dtype=np.uint8)
    img_show[: img_pano.shape[0], : img_pano.shape[1], :] = img_pano
    img_show[: img_he.shape[0], img_pano.shape[1] :, :] = img_he

    num_matches = len(matches)
    for idx in np.random.choice(
        num_matches, min(display_count, num_matches), replace=False
    ):
        match = matches[idx]
        pt0, pt1 = match[0].keypoint.pt, match[1].keypoint.pt
        pt0, pt1 = (
            np.array(pt0).astype(np.int32) // scale,
            np.array(pt1).astype(np.int32) // scale,
        )
        pt1[0] += img_pano.shape[1]

        img_show = cv2.line(img_show, pt0, pt1, (255, 255, 0), 2)

    is_exit = False
    if verbose:
        print("show matches")
        cv2.imshow("matches", img_show)
        code = cv2.waitKey()
        print(code)
        is_exit = code == 113  # 'q'

    print("save path: ", save_path)
    cv2.imwrite(save_path, img_show)


def show_overlay_img(img_fix: np.ndarray, sample_data: np.ndarray, save_path: str):
    h, w, _ = img_fix.shape
    print("show_trans_img")

    sampling_img = np.zeros((h, w, 3), np.uint8)
    overlay_img = np.zeros((h, w, 3), np.uint8)

    overlay_img[:, :, :] = img_fix
    for data in sample_data:
        # fix image x, y
        # sample on moving image b, g, r
        x, y, b, g, r = data
        x, y = int(x), int(y)

        sampling_img[y, x] = [b, g, r]
        overlay_img[y, x] = (
            np.array([b, g, r]) * 0.5 + overlay_img[y, x] * 0.5
        ).astype(np.uint8)

    print("\tsave path: ", save_path)
    cv2.imwrite(save_path + ".sampling_img.tif", sampling_img)
    cv2.imwrite(save_path, overlay_img)
