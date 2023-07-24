from abc import abstractmethod
from typing import List, Tuple
import cv2
import numpy as np
from pathlib import Path


class Feature:
    def __init__(self, descriptor) -> None:
        self.desc = descriptor

    @abstractmethod
    def distance_to(self, another_feature) -> float:
        return None


class PointFeature(Feature):
    def __init__(self, keypoint, descriptor: None) -> None:
        super().__init__(descriptor)
        self.keypoint = keypoint
        self.img = None


def feature_match(
    feat_moving: List[Feature],
    feat_fixed: List[Feature],
    top_count=30,
    filter_fun=None,
    cache_dir="",
) -> List[Tuple[PointFeature, PointFeature, float]]:
    print("feature match")
    print(f"\tcache_dir {cache_dir}")

    len_fix, len_move = len(feat_fixed), len(feat_moving)
    print(f"\tlen fix {len_fix}  move {len_move}")

    query_map = np.zeros((len_fix, len_move), dtype=np.float32)
    for r in range(len_fix):
        for c in range(len_move):
            query_map[r, c] = feat_fixed[r].distance_to(feat_moving[c])

    print(f"\tmax of query_map {query_map.max()}")
    if cache_dir != "":
        cv2.imwrite(str(Path(cache_dir) / "query_map.tif"), query_map / query_map.max())

    matches = [
        (r, np.argpartition(query_map[r, :], (1, 2, 3))[:3]) for r in range(len_fix)
    ]
    print("\tmatches", f"len {len(matches)}")

    if filter_fun is not None:
        matches = [(r, cs[0]) for r, cs in matches if filter_fun(query_map, r, cs)]
        print("\tmatches", f"len filtered {len(matches)}")
    else:
        matches = [(r, cs[0]) for r, cs in matches]

    assert len(matches) > 10

    print(f"\tmatched feat moving desc example r 10 <-> c {matches[10][1]}")

    matched_feats = [
        (feat_moving[m[1]], feat_fixed[m[0]], query_map[m[0], m[1]]) for m in matches
    ]
    print(f"\tfinal matches size {len_fix} -> {len(matched_feats)}")

    return matched_feats
