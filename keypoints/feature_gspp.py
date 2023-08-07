from pathlib import Path
import pickle
from typing import List, Tuple
from typing_extensions import override

import cv2
import numpy as np

if __name__ != "__main__":
    from .feature import PointFeature, feature_match
    from .feature_spp import Spp, SppFeature
else:
    from sys import argv, path as sys_path

    sys_path.insert(0, str(Path(sys_path[0]).parent))
    from keypoints.feature import PointFeature, feature_match
    from keypoints.feature_spp import Spp, SppFeature


class GsppFeature(PointFeature):
    N_EDGE = 3

    def __init__(self, center_spp: SppFeature, edge_spps: List[SppFeature]) -> None:
        self._center = center_spp
        self._edge_feats = edge_spps
        keypoint = self._center.keypoint
        desc = self._create_graph_desc()
        super().__init__(keypoint, desc)

    def _create_graph_desc(self):
        center_pt = np.array(self._center.keypoint.pt)
        assert len(self._edge_feats) == GsppFeature.N_EDGE

        desc = [
            {
                "vec": np.array(spp_feat.keypoint.pt) - center_pt,
                "desc": np.array(spp_feat.desc),
            }
            for spp_feat in self._edge_feats
        ]
        desc.insert(0, {"vec": np.array([0, 0]), "desc": self._center.desc})
        return desc

    @override
    def distance_to(self, another_feature) -> float:
        point_dist = np.linalg.norm(
            self.desc[0]["desc"] - another_feature.desc[0]["desc"]
        )

        def vec_distance(vec1, vec2):
            diff = (vec2 / np.linalg.norm(vec2)) - (vec1 / np.linalg.norm(vec1))
            # diff = vec2 - vec1
            return np.linalg.norm(diff)

        edge_dist = [
            vec_distance(
                self.desc[idx + 1]["vec"], another_feature.desc[idx + 1]["vec"]
            )
            * np.linalg.norm(
                self.desc[idx + 1]["desc"] - another_feature.desc[idx + 1]["desc"]
            )
            for idx in range(GsppFeature.N_EDGE)
        ]
        edge_dist = np.sum(edge_dist)

        distance = np.sum([point_dist, edge_dist])
        return distance


class Gspp:
    def __init__(self, img) -> None:
        self.img = img
        self.features = None
        self._spp = Spp(self.img)

    def compute(
        self,
        graph_edges=GsppFeature.N_EDGE,
        num_feat=None,
        cache_dir="",
    ):
        print("Gspp compute")

        spp_cache_file = Path(cache_dir) / "spp.pkl"
        if spp_cache_file.exists():
            with open(spp_cache_file, "rb") as f:
                self._spp = pickle.load(f)
                print(f"\tload spp from cache {cache_dir}")
        else:
            spp_cache_dir = cache_dir.replace("gspp", "spp")
            self._spp.compute(num_feat=num_feat, cache_dir=spp_cache_dir)
            if cache_dir != "":
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                with open(spp_cache_file, "wb") as f:
                    pickle.dump(self._spp, f)

        feats = self._spp.features
        len_feat = len(feats)
        print(f"\tspp comput num feat {len_feat}")

        dist_map = np.zeros((len_feat, len_feat), dtype=np.float32)
        for r in range(dist_map.shape[0]):
            r_pt = np.array(feats[r].keypoint.pt)
            for c in range(dist_map.shape[1]):
                c_pt = np.array(feats[c].keypoint.pt)
                dist_map[r, c] = np.linalg.norm(c_pt - r_pt)
        print(f"\tmax of dist_map {dist_map.max()}")
        if cache_dir != "":
            cv2.imwrite(
                str(Path(cache_dir) / "distance_map.tif"), dist_map / dist_map.max()
            )

        nodes = [
            (
                r,
                np.argpartition(dist_map[r, :], range(GsppFeature.N_EDGE + 1))[
                    1 : GsppFeature.N_EDGE + 1
                ],
            )
            for r in range(len_feat)
        ]

        if cache_dir != "":
            img_show = np.copy(self.img)
            for idx in np.random.choice(range(len_feat), 40, replace=False):
                center, edges = nodes[idx]
                center_pt = feats[center].keypoint.pt
                img_show = cv2.circle(
                    img_show, center_pt, 4, (255, 0, 0), thickness=cv2.FILLED
                )
                for edge in edges:
                    edge_pt = feats[edge].keypoint.pt
                    img_show = cv2.line(
                        img_show, center_pt, edge_pt, (255, 255, 0), thickness=2
                    )
            cv2.imwrite(str(Path(cache_dir) / "graph_spp_sample_20.tif"), img_show)

        self.features = [
            GsppFeature(feats[r], [feats[c] for c in cs]) for r, cs in nodes
        ]

    @staticmethod
    def refine_fun(query_map, r, cs):
        return query_map[r, cs[0]] < query_map[r, cs[1]] and query_map[r, cs[0]] < 2

    @staticmethod
    def match(
        feat_moving: List[GsppFeature],
        feat_fixed: List[GsppFeature],
        top_count=30,
        filter_fun=None,
        cache_dir="",
    ) -> List[Tuple[PointFeature, PointFeature, float]]:
        return feature_match(
            feat_moving,
            feat_fixed,
            top_count,
            filter_fun=filter_fun,
            cache_dir=cache_dir,
        )


if __name__ == "__main__":
    from utils.display import show_matches
    from utils.dataset import load_image

    if len(argv) != 3:
        print(f"Usage: {argv[0]} dataset_path data_id")
        exit(-1)

    dataset_path = argv[1]
    data_id = argv[2]
    main_id = data_id.split("_")[0]
    cache_dir = Path("outputs") / "cache" / "gspp" / data_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "he").mkdir(parents=True, exist_ok=True)
    (cache_dir / "pano").mkdir(parents=True, exist_ok=True)

    img_he_path = Path(dataset_path) / "H&E_IMC" / "Pair" / data_id / f"HE{main_id}.tif"
    img_he = load_image(str(img_he_path), downsize=4)
    print(f"he shape {img_he.shape}")
    img_pano_path = (
        Path(dataset_path) / "H&E_IMC" / "Pair" / data_id / f"{main_id}_panorama.tif"
    )
    img_pano = load_image(str(img_pano_path), downsize=4)
    print(f"pano shape {img_pano.shape}")

    gspp = Gspp(img_he)
    gspp.compute(
        num_feat=100,
        cache_dir=str(cache_dir / "he"),
    )
    he_features = gspp.features

    gspp = Gspp(img_pano)
    gspp.compute(
        cache_dir=str(cache_dir / "pano"),
    )
    pano_features = gspp.features

    print(f"feature count: he {len(he_features)}  pano {len(pano_features)}")

    matches = feature_match(
        pano_features, he_features, cache_dir=str(cache_dir), filter_fun=Gspp.refine_fun
    )
    show_matches(img_pano, img_he, matches, save_path=str(cache_dir / "matches.tif"))
