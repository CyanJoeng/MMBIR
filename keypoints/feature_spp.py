from typing_extensions import override
from typing import List, Tuple

from .feature import PointFeature, feature_match
from pathlib import Path

import numpy as np
import cv2
from tensorflow import keras


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

    def compute(self, layers=None, num_feat=None, cache_dir=""):
        if layers is None:
            layers = Spp.LAYERS

        print("spp compute")
        print(f"\tnum_feat {num_feat}")
        print(f"\tlayers {layers}")

        img_h, img_w, _ = self.img.shape
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        input_data = np.array(gray_img).astype(np.float32)
        input_data = input_data.reshape([1, img_h, img_w, 1])
        print(f"\tinput data shape {input_data.shape}")

        mask_size = max(img_h, img_w) // 50

        # feat count in each layer
        # 1x1 + 2x2 + 4x4 + 8x8 + 16x16 + ...
        pool_size = [2 ** (pow + 1) for pow in range(layers)][::-1]
        print("\tpool size ", pool_size)
        block_size = pool_size[0]
        h_bs = block_size // 2
        feat_count = [(block_size // ps) ** 2 for ps in pool_size]
        feature_len = np.sum(feat_count)
        print(f"\tfeat_count {feat_count} sum {feature_len}")

        # spacial pooling model
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
        pool_results = model.predict(input_data)
        print("\tlayer shapes ", [layer.shape for layer in pool_results])

        # resolve features for each pixel
        feature_map = np.zeros((feature_len, img_h, img_w), np.float32)
        print("\tfeature shape", feature_map.shape)
        for idx, (p_sz, feat) in enumerate(zip(pool_size, pool_results)):
            feat_st = np.sum(feat_count[:idx]).astype(np.int32)
            feat_ed = np.sum(feat_count[: idx + 1]).astype(np.int32)

            block_off_tl = np.floor(2 ** (idx - 1)).astype(np.int32)
            block_off_br = np.ceil(2 ** (idx - 1)).astype(np.int32)

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
        print("\tnew feature_map shape", feature_map.shape)
        # print(f"\t-->feature 0, 0  {feature_map[:, h_bs, h_bs]}")
        self.feature_map = feature_map

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
            f"\tfeature_intensity_map {feature_map.shape} -> \
            {feature_intensity_map.shape}, \
            {np.max(feature_intensity_map)}, \
            {np.min(feature_intensity_map)}"
        )
        self.feature_intensity_map = feature_intensity_map

        # select features based on intensity of features
        def select_feature(feature_pos, intensity_map):
            return intensity_map

        if num_feat is None:
            num_feat = 300

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

        # ceate spp features
        self.features = [
            SppFeature(
                (x, y),
                feature_map[:, y, x] / np.linalg.norm(feature_map[:, y, x]),
            )
            for x, y in top_feature_pos
        ]

    @staticmethod
    def refine_fun(query_map, r, cs):
        return query_map[r, cs[0]] < query_map[r, cs[1]] and query_map[r, cs[0]] < 1

    @staticmethod
    def match(
        feat_moving: List[SppFeature],
        feat_fixed: List[SppFeature],
        top_count=30,
        cache_dir="",
    ) -> List[Tuple[PointFeature, PointFeature, float]]:
        return feature_match(
            feat_moving,
            feat_fixed,
            top_count,
            filter_fun=Spp.refine_fun,
            cache_dir=cache_dir,
        )


if __name__ == "__main__":
    from sys import argv, path as sys_path
    from feature import feature_match

    if len(argv) != 3:
        print(f"Usage: {argv[0]} image_he image_panorama")
        exit(-1)

    data_id = Path(argv[1]).parts[-2]
    cache_dir = Path(sys_path[0]).parent / "outputs" / "cache" / "spp" / data_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "he").mkdir(parents=True, exist_ok=True)
    (cache_dir / "pano").mkdir(parents=True, exist_ok=True)

    img_he = cv2.imread(argv[1], cv2.IMREAD_ANYCOLOR)
    img_pano = cv2.imread(argv[2], cv2.IMREAD_ANYCOLOR)

    spp = Spp(img_he)
    spp.compute(num_feat=100, layers=5, cache_dir=(cache_dir / "he"))
    he_features = spp.features

    spp = Spp(img_pano)
    spp.compute(layers=5, cache_dir=(cache_dir / "pano"))
    pano_features = spp.features

    print(f"feature count: he {len(he_features)}  pano {len(pano_features)}")

    feature_match(
        pano_features, he_features, filter_fun=Spp.refine_fun, cache_dir=str(cache_dir)
    )
