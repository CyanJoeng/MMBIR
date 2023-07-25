from typing import List, Tuple
import cv2
import numpy as np
from .feature import PointFeature


MIN_HOMO_COUNT = 10
MIN_FUNDA_COUNT = 8


def filter_by_fundamental(matches: List[Tuple[PointFeature]]):
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


def find_trans_matrix(matches: List[Tuple[PointFeature]]):
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


def calc_trans_matrix_by_lstsq(data_pano, data_he):
    input_pano_pos = np.array(data_pano[0]).reshape((-1, 2)).astype(np.float32)
    input_he_pos = np.array(data_he[0]).reshape((-1, 2)).astype(np.float32)

    feat_count = len(input_pano_pos)

    input_pano_pos = np.hstack([input_pano_pos, np.ones((feat_count, 1))])
    input_he_pos = np.hstack([input_he_pos, np.ones((feat_count, 1))])

    affine_matrix, _, _, _ = np.linalg.lstsq(input_pano_pos, input_he_pos, rcond=None)

    print(affine_matrix)

    return affine_matrix


def trans_image_by(trans_matrix, image) -> np.ndarray:
    print("trans image")

    h, w, _ = image.shape
    pts = np.float32([[c, r, 1] for r in range(h) for c in range(w)]).reshape(-1, 3)
    # dst = cv2.perspectiveTransform(pts, trans_matrix)
    dst = pts @ trans_matrix
    dst = dst[:, :2]

    colour = np.float32([image[r, c] for r in range(h) for c in range(w)]).reshape(
        -1, 3
    )
    print(f"\tdata shape  dst {dst.shape} colour {colour.shape}")
    data = np.concatenate((dst, colour), axis=-1)
    print("\ttrans_image data shape ", data.shape)
    return data
