from typing import List, Tuple
import cv2
from feature_spp import SppFeature
import numpy as np


MIN_MATCH_COUNT = 10


def find_trans_matrix(matches: List[Tuple[SppFeature]]):
    if len(matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([m[0].keypoint.pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([m[1].keypoint.pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10)
        matchesMask = mask.ravel().tolist()
    else:
        print(
            "Not enough matches are found - {}/{}".format(len(matches), MIN_MATCH_COUNT)
        )
        matchesMask = None

    # print("match mask:", matchesMask)
    # print("homography matrix:\n", M)

    homo_matches = [match for match, mask in zip(matches, matchesMask) if mask == 1]
    print(f"\tfinal matches size {len(matches)} -> {len(homo_matches)}")

    return M, homo_matches
