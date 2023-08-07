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


def find_trans_matrix(data_pano, data_he):
    input_pano_pos = np.array(data_pano[0]).reshape((-1, 2)).astype(np.float32)
    input_he_pos = np.array(data_he[0]).reshape((-1, 2)).astype(np.float32)

    if input_pano_pos.shape[0] > MIN_HOMO_COUNT:
        src_pts = input_pano_pos.reshape(-1, 1, 2)
        dst_pts = input_he_pos.reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
        matchesMask = mask.ravel().tolist()
    else:
        print(
            "Not enough matches are found - {}/{}".format(
                input_pano_pos.shape, MIN_HOMO_COUNT
            )
        )
        matchesMask = None

    # print("match mask:", matchesMask)
    # print("homography matrix:\n", M)

    print(f"\tfinal matches size {input_pano_pos.shape} -> {np.sum(matchesMask)}")
    M = M.T
    print("Homo matrix\n", M)

    return M, matchesMask


def calc_trans_matrix_by_lstsq(data_pano, data_he):
    input_pano_pos = np.array(data_pano[0]).reshape((-1, 2)).astype(np.float32)
    input_he_pos = np.array(data_he[0]).reshape((-1, 2)).astype(np.float32)

    feat_count = len(input_pano_pos)

    input_pano_pos = np.hstack([input_pano_pos, np.ones((feat_count, 1))])
    input_he_pos = np.hstack([input_he_pos, np.ones((feat_count, 1))])

    affine_matrix, _, _, _ = np.linalg.lstsq(input_pano_pos, input_he_pos, rcond=None)

    print("trans matrix with lstsq \n", affine_matrix)

    return affine_matrix


def trans_image_by(trans_matrix, images) -> np.ndarray:
    print("trans image")

    img_pano, img_he = images

    h, w, _ = img_pano.shape
    colour = np.float32([img_pano[r, c] for r in range(h) for c in range(w)]).reshape(
        -1, 3
    )

    pts = (
        np.float32([[c, r] for r in range(h) for c in range(w)]).reshape(-1, 2)
        / np.array([w, h])
        * 2
        - 1
    )
    pts = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=-1)
    # dst = cv2.perspectiveTransform(pts, trans_matrix)
    dst = pts @ trans_matrix

    h, w, _ = img_he.shape
    # dst = dst[:, :2]
    dst = (dst[:, :2] + 1) * 0.5 * np.array([w, h])

    print(f"\tdata shape  dst {dst.shape} colour {colour.shape}")
    data = np.concatenate((dst, colour), axis=-1)
    print("\ttrans_image data shape ", data.shape)
    print("\ttrans offset ", trans_matrix[2, :2] / 2 * np.array([w, h]))
    return data
