from pathlib import Path
from typing import List, Tuple
from typing_extensions import override

import cv2
import numpy as np
from tensorflow import keras


if __name__ != "__main__":
    from .feature import PointFeature, feature_match
else:
    from sys import argv, path as sys_path

    sys_path.insert(0, str(Path(sys_path[0]).parent))
    from keypoints.feature import PointFeature, feature_match


class SppKeypoint:
    def __init__(self, center) -> None:
        self.pt = center


class SppFeature(PointFeature):
    SIGMOID_DISTANCE = False

    def __init__(self, kp, descriptor: None) -> None:
        super().__init__(SppKeypoint(kp), descriptor)

    @override
    def distance_to(self, another_feature) -> float:
        diff = self.desc - another_feature.desc

        if SppFeature.SIGMOID_DISTANCE:
            len_v = diff.shape[0]
            h_len_v = diff.shape[0] // 3

            scale = 2 * np.e / len_v

            # sigmoid(x) = 1 / (1 + exp(-x)).
            diff = np.array(
                [
                    2 * val / (1 + np.exp(-(idx - h_len_v) * scale))
                    for idx, val in enumerate(diff)
                ]
            )
        distance = np.linalg.norm(diff)
        return distance


class Spp:
    LAYERS = 5

    def __init__(self, img) -> None:
        self.img = img
        self.features = None
        self.feature_map = None

        self.model, args = Spp.create_point_feature_model()
        self.feat_count, self.pool_size, self.block_size = args

    @staticmethod
    def create_point_feature_model():
        # spacial pooling model
        pool_size = [2 ** (pow + 1) for pow in range(Spp.LAYERS)][::-1]
        print("Spp create model")
        print("\tpool size ", pool_size)
        block_size = pool_size[0]

        # feat count in each layer
        # 1x1 + 2x2 + 4x4 + 8x8 + 16x16 + ...
        feat_count = [(block_size // ps) ** 2 for ps in pool_size]
        feature_len = np.sum(feat_count)
        print(f"\tfeat_count {feat_count} sum {feature_len}")

        inputs = keras.Input(shape=(None, None, 1))
        outputs = [
            keras.Sequential(
                [
                    keras.layers.ZeroPadding2D(
                        padding=[
                            (
                                np.ceil((sz - 1) / 2).astype(np.int32),
                                np.floor((sz - 1) / 2).astype(np.int32),
                            )
                        ]
                        * 2
                    ),
                    keras.layers.AveragePooling2D(
                        pool_size=(sz, sz), strides=1, padding="valid"
                    ),
                ]
            )(inputs)
            for sz in pool_size
        ]

        model = keras.Model(inputs, outputs)
        return model, (feat_count, pool_size, block_size)

    def _compute_feature_map(self):
        print(f"\tcompute feature map:")
        img_h, img_w, _ = self.img.shape
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        input_data = np.array(gray_img).astype(np.float32)
        input_data = input_data.reshape([1, img_h, img_w, 1])
        print(f"\tinput data shape {input_data.shape}")

        pool_results = self.model.predict(input_data)
        print("\tlayer shapes ", [layer.shape for layer in pool_results])
        return pool_results

    def _create_intensity_map(self, pool_results):
        print(f"\tcreate feature map:")
        img_h, img_w, _ = self.img.shape
        h_bs = self.block_size // 2
        feature_len = np.sum(self.feat_count)

        # resolve features for each pixel
        feature_map = np.zeros((feature_len, img_h, img_w), np.float32)
        print("\tfeature shape", feature_map.shape)
        print("\tpool size", self.pool_size)
        for idx, (p_sz, feat) in enumerate(zip(self.pool_size, pool_results)):
            feat_st = np.sum(self.feat_count[:idx]).astype(np.int32)
            feat_ed = np.sum(self.feat_count[: idx + 1]).astype(np.int32)

            # block_off_tl = np.floor(2 ** (idx - 1)).astype(np.int32)
            # block_off_br = np.ceil(2 ** (idx - 1)).astype(np.int32)

            range_st_off = (-h_bs) + (p_sz // 2)
            range_ed_off = h_bs
            range_step = p_sz
            print(
                f"\tfeatu map  st {range_st_off} ed {range_ed_off} step {range_step} count {len(range(range_st_off, range_ed_off, range_step)) ** 2}"
            )

            for idx_r in range(h_bs, img_h - h_bs + 1):
                for idx_c in range(h_bs, img_w - h_bs + 1):
                    feature_map[feat_st:feat_ed, idx_r, idx_c] = feat[
                        0,
                        range_st_off + idx_r : range_ed_off + idx_r : range_step,
                        range_st_off + idx_c : range_ed_off + idx_c : range_step,
                        0,
                    ].flatten()

                    # if idx_r == h_bs and idx_c == h_bs:
                    #     print(f"-->feature 0, 0  {feature_map[:, h_bs, h_bs]}")

        feature_map = feature_map[1:, :, :] - feature_map[0, :, :]
        # print(f"\t-->feature 0, 0  {feature_map[:, h_bs, h_bs]}")
        print("\tnew feature_map shape", feature_map.shape)

        # create intensity map
        # the ligher the biger norm of feature vector
        feature_intensity_map = np.linalg.norm(feature_map[:, :, :], axis=0)
        print(
            f"\tfeature_intensity_map \
            {feature_intensity_map.min()} \
            {feature_intensity_map.max()} \
            {feature_intensity_map.mean()}"
        )
        feature_intensity_map /= np.max(feature_intensity_map)
        print(
            f"\tfeature_intensity_map {feature_intensity_map.shape} -> \
            {feature_intensity_map.shape}, \
            {np.max(feature_intensity_map)}, \
            {np.min(feature_intensity_map)}"
        )
        return feature_map, feature_intensity_map

    def _feature_selection(self, num_feat, feature_intensity_map, mask_size, cache_dir):
        print("\tfeature selection")
        top_feature_pos = np.zeros((num_feat, 2), dtype=np.int32)
        masked_intensity_map = np.copy(feature_intensity_map)
        for idx in range(num_feat):
            y, x = np.unravel_index(
                np.argmax(masked_intensity_map), shape=masked_intensity_map.shape
            )
            top_feature_pos[idx, :] = np.array([x, y]).astype(np.int32)
            masked_intensity_map = cv2.circle(
                masked_intensity_map, (x, y), mask_size, 0, -1
            )

        if cache_dir != "":
            cache_dir = Path(cache_dir)
            cv2.imwrite(str(cache_dir / "intensity_map.tif"), feature_intensity_map)
            cv2.imwrite(
                str(cache_dir / "intensity_map_mask.tif"),
                masked_intensity_map,
            )

        return top_feature_pos

    def compute(self, layers=None, num_feat=None, cache_dir=""):
        if layers is None:
            layers = Spp.LAYERS

        print("spp compute")
        print(f"\tnum_feat {num_feat}")
        print(f"\tlayers {layers}")
        print(f"\timg shape {self.img.shape}")

        pool_results = self._compute_feature_map()
        self.feature_map, feature_intensity_map = self._create_intensity_map(
            pool_results
        )

        if num_feat is None:
            num_feat = 300

        img_h, img_w, _ = self.img.shape
        mask_size = max(img_h, img_w) // 50
        top_feature_pos = self._feature_selection(
            num_feat, feature_intensity_map, mask_size, cache_dir
        )

        # ceate spp features
        self.features = [
            SppFeature(
                (x, y),
                self.feature_map[:, y, x] / np.linalg.norm(self.feature_map[:, y, x]),
            )
            for x, y in top_feature_pos
        ]

    @staticmethod
    def refine_fun(query_map, r, cs):
        return query_map[r, cs[0]] < query_map[r, cs[1]] and query_map[r, cs[0]] < 0.75

    @staticmethod
    def match(
        feat_moving: List[SppFeature],
        feat_fixed: List[SppFeature],
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
    from keypoints.feature import PointFeature, feature_match
    from utils.display import show_matches
    from utils.dataset import load_image

    if len(argv) != 3:
        print(f"Usage: {argv[0]} dataset_path data_id")
        exit(-1)

    dataset_path = argv[1]
    data_id = argv[2]
    main_id = data_id.split("_")[0]
    cache_dir = Path("outputs") / "cache" / "spp" / data_id
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

    spp = Spp(img_he)
    spp.compute(num_feat=100, layers=5, cache_dir=(cache_dir / "he"))
    he_features = spp.features

    spp = Spp(img_pano)
    spp.compute(layers=5, cache_dir=(cache_dir / "pano"))
    pano_features = spp.features

    print(f"feature count: he {len(he_features)}  pano {len(pano_features)}")

    matches = feature_match(
        pano_features, he_features, filter_fun=Spp.refine_fun, cache_dir=str(cache_dir)
    )

    show_matches(img_pano, img_he, matches, save_path=str(cache_dir / "matches.tif"))
