from typing import List, Tuple
import cv2
import numpy as np
from .feature import PointFeature


MIN_HOMO_COUNT = 10
MIN_FUNDA_COUNT = 8


def unify_img_pts(pts, shape):
    # pts shape (N x 1 x 2)
    assert len(pts.shape) == 3
    assert pts.shape[1] == 1
    return pts / np.array([shape[1], shape[0]], dtype=np.float32).reshape(1, 2) * 2 - 1


def deunify_img_pts(pts, shape):
    # pts shape (N x 1 x 2)
    assert len(pts.shape) == 3
    assert pts.shape[1] == 1
    return (
        (pts + 1) * 0.5 * np.array([shape[1], shape[0]], dtype=np.float32).reshape(1, 2)
    )


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


def filter_by_homography(matches: List[Tuple[PointFeature]]):
    if len(matches) > MIN_HOMO_COUNT:
        src_pts = np.float32([m[0].keypoint.pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([m[1].keypoint.pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
        matchesMask = mask.ravel().tolist()
    else:
        print(
            "Not enough matches are found - {}/{}".format(len(matches), MIN_FUNDA_COUNT)
        )
        M = None
        matchesMask = None

    print("match mask:", matchesMask)
    print("homo matrix:\n", M)

    refined_matches = [match for match, mask in zip(matches, matchesMask) if mask == 1]
    print(f"\tfinal matches size {len(matches)} -> {len(refined_matches)}")

    return M, refined_matches


def matches_to_pts(matches):
    src_pts = np.float32([m[0].keypoint.pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([m[1].keypoint.pt for m in matches]).reshape(-1, 1, 2)
    return src_pts, dst_pts


def find_trans_matrix(pts_mov, pts_fix):
    input_mov_pos = np.array(pts_mov[0]).reshape((-1, 2)).astype(np.float32)
    input_fix_pos = np.array(pts_fix[0]).reshape((-1, 2)).astype(np.float32)

    if input_mov_pos.shape[0] > MIN_HOMO_COUNT:
        src_pts = input_mov_pos.reshape(-1, 1, 2)
        dst_pts = input_fix_pos.reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
        matchesMask = mask.ravel().tolist()
    else:
        print(
            "Not enough matches are found - {}/{}".format(
                input_mov_pos.shape, MIN_HOMO_COUNT
            )
        )
        matchesMask = None

    # print("match mask:", matchesMask)
    # print("homography matrix:\n", M)

    print(f"\tfinal matches size {input_mov_pos.shape} -> {np.sum(matchesMask)}")
    M = M.T
    print("Homo matrix\n", M)

    return M, matchesMask


def calc_trans_matrix_by_lstsq(pts_mov, pts_fix):
    input_mov_pos = np.array(pts_mov).reshape((-1, 2)).astype(np.float32)
    input_fix_pos = np.array(pts_fix).reshape((-1, 2)).astype(np.float32)

    feat_count = len(input_mov_pos)

    input_mov_pos = np.hstack([input_mov_pos, np.ones((feat_count, 1))])
    input_fix_pos = np.hstack([input_fix_pos, np.ones((feat_count, 1))])

    affine_matrix, _, _, _ = np.linalg.lstsq(input_mov_pos, input_fix_pos, rcond=None)

    print("trans matrix with lstsq \n", affine_matrix)

    return affine_matrix


def sample_pix_with(trans_matrix, images) -> np.ndarray:
    print("trans image")

    img_mov, img_fix = images

    mov_h, mov_w, _ = img_mov.shape
    fix_h, fix_w, _ = img_fix.shape

    # N x 3
    pts_fix = (
        np.float32([[c, r] for r in range(fix_h) for c in range(fix_w)]).reshape(-1, 2)
        / np.array([fix_w, fix_h])
        * 2
        - 1
    )
    pts_fix = np.concatenate([pts_fix, np.ones((pts_fix.shape[0], 1))], axis=-1)
    # dst = cv2.perspectiveTransform(pts, trans_matrix)
    # project pts on moving image

    pts_proj = pts_fix @ np.linalg.inv(trans_matrix)

    dst = (pts_proj[:, :2] + 1) * 0.5 * np.array([mov_w, mov_h])

    def get_data(c, r):
        x, y = dst[r * fix_w + c]
        x, y = int(x + 0.5), int(y + 0.5)
        if x < 0 or y < 0 or x >= mov_w or y >= mov_h:
            return [c, r, 0, 0, 0]
        return [c, r] + list(img_mov[y, x])

    data = np.float32(
        [get_data(c, r) for r in range(fix_h) for c in range(fix_w)]
    ).reshape(-1, 5)

    print("\tsample data shape ", data.shape)
    print("\ttrans offset ", trans_matrix[2, :2] / 2 * np.array([mov_w, mov_h]))
    return data
