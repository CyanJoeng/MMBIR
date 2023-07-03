from typing import List, Tuple
from typing_extensions import override
from feature import Feature, PointFeature

import numpy as np
import cv2


class OrbFeature(PointFeature):
    def __init__(self, kp, descriptor: None) -> None:
        super().__init__(kp, descriptor)

    @override
    def distance_to(self, another_feature) -> float:
        return np.array(self.desc) - np.array(another_feature.desc)


class Orb:
    def __init__(self, img) -> None:
        self.img = img
        self.features = None

    def compute(self):
        orb = cv2.ORB_create(nfeatures=300)

        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray_img, None)

        self.features = [
            OrbFeature(kp, desc) for kp, desc in zip(keypoints, descriptors)
        ]

    @staticmethod
    def match(
        orb_moving: List[OrbFeature], orb_fixed: List[OrbFeature], top_count=30
    ) -> List[Tuple[Feature]]:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        desc_moving = np.array([feat.desc for feat in orb_moving])
        desc_fixed = np.array([feat.desc for feat in orb_fixed])

        # Perform matching
        matches = bf.match(desc_moving, desc_fixed)

        matched_feats = [
            (orb_moving[m.queryIdx], orb_fixed[m.trainIdx]) for m in matches[:top_count]
        ]
        return matched_feats
