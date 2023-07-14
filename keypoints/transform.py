from typing import List, Tuple
import cv2
from feature_spp import SppFeature
import numpy as np


MIN_HOMO_COUNT = 10
MIN_FUNDA_COUNT = 8


def filter_by_fundamental(matches: List[Tuple[SppFeature]]):
    if len(matches) > MIN_HOMO_COUNT:
        src_pts = np.float32([m[0].keypoint.pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([m[1].keypoint.pt for m in matches]).reshape(-1, 1, 2)
        F_M, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 2, 0.99)
        matchesMask = mask.ravel().tolist()
    else:
        print(
            "Not enough matches are found - {}/{}".format(len(matches), MIN_FUNDA_COUNT)
        )
        F_M = None
        matchesMask = None

    print("match mask:", matchesMask)
    print("fundamental matrix:\n", F_M)

    refined_matches = [match for match, mask in zip(matches, matchesMask) if mask == 1]
    print(f"\tfinal matches size {len(matches)} -> {len(refined_matches)}")

    return F_M, refined_matches


def find_trans_matrix(matches: List[Tuple[SppFeature]]):
    if len(matches) > MIN_HOMO_COUNT:
        src_pts = np.float32([m[0].keypoint.pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([m[1].keypoint.pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
        matchesMask = mask.ravel().tolist()
    else:
        print(
            "Not enough matches are found - {}/{}".format(len(matches), MIN_HOMO_COUNT)
        )
        matchesMask = None

    # print("match mask:", matchesMask)
    # print("homography matrix:\n", M)

    homo_matches = [match for match, mask in zip(matches, matchesMask) if mask == 1]
    print(f"\tfinal matches size {len(matches)} -> {len(homo_matches)}")

    return M, homo_matches


def trans_image_by(trans_matrix: np.ndarray, image) -> np.ndarray:
    print("trans image")
    h, w, _ = image.shape
    pts = np.float32([[c, r] for r in range(h) for c in range(w)]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, trans_matrix)

    colour = np.float32([image[r, c] for r in range(h) for c in range(w)]).reshape(
        -1, 1, 3
    )
    print(f"\tdata shape {dst.shape} {colour.shape}")
    data = np.concatenate((dst, colour), axis=2)
    data = data.squeeze(axis=1)
    print("\ttrans_image data shape ", data.shape)
    return data
