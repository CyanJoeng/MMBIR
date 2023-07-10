from typing import List, Tuple
from typing_extensions import override
from feature import Feature, PointFeature

import numpy as np
import cv2


class SiftFeature(PointFeature):
    def __init__(self, kp, descriptor: None) -> None:
        super().__init__(kp, descriptor)

    @override
    def distance_to(self, another_feature) -> float:
        return np.array(self.desc).astype(np.float32) - np.array(
            another_feature.desc
        ).astype(np.float32)


class Sift:
    def __init__(self, img) -> None:
        self.img = img
        self.features = None

    def compute(self, num_feat=300):
        sift = cv2.SIFT_create(nfeatures=300)

        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_img, None)

        self.features = [
            SiftFeature(kp, desc) for kp, desc in zip(keypoints, descriptors)
        ]

    @staticmethod
    def match(
        sift_moving: List[SiftFeature], sift_fixed: List[SiftFeature], top_count=30
    ) -> List[Tuple[Feature]]:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        desc_moving = np.array([feat.desc for feat in sift_moving])
        desc_fixed = np.array([feat.desc for feat in sift_fixed])

        # Perform matching
        matches = bf.match(desc_moving, desc_fixed)

        matched_feats = [
            (sift_moving[m.queryIdx], sift_fixed[m.trainIdx])
            for m in matches[:top_count]
        ]
        return matched_feats
