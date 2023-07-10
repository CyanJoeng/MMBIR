from typing import List, Tuple
from typing_extensions import override

from keras import layers
from feature import Feature, PointFeature

import numpy as np
from sklearn.preprocessing import normalize
import cv2
import tensorflow as tf
from tensorflow import keras


class SppKeypoint:
    def __init__(self, center) -> None:
        self.pt = center


class SppFeature(PointFeature):
    def __init__(self, kp, descriptor: None) -> None:
        super().__init__(SppKeypoint(kp), descriptor)

    @override
    def distance_to(self, another_feature) -> float:
        diff = self.desc - another_feature.desc
        return np.linalg.norm(diff)


class Spp:
    def __init__(self, img) -> None:
        self.img = img
        self.features = None

    def compute(self, layers=3, mask_r=2, num_feat=None):
        img_h, img_w, _ = self.img.shape

        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        input_data = np.array(gray_img).astype(np.float32)
        input_data = input_data.reshape([1, img_h, img_w, 1])
        print(f"input data shape {input_data.shape}")

        pool_size = [2**pow for pow in range(1, layers + 1)]
        print("pool size ", pool_size)

        block_width = [int(pool_size[-1] / sz) for sz in pool_size]
        area_size = [int(w**2) for w in block_width]
        feature_len = np.sum(area_size[:-1])
        print(f"block width: {block_width}  feature len {feature_len}")

        inputs = keras.Input(shape=(None, None, 1))
        outputs = [
            keras.layers.AveragePooling2D(pool_size=(sz, sz))(inputs)
            for sz in pool_size
        ]
        model = keras.Model(inputs, outputs)
        pool_results = model.predict(input_data)
        print([layer.shape for layer in pool_results])

        feat_map_h, feat_map_w = pool_results[-1].shape[1:-1]
        feature_map = np.zeros((feature_len, feat_map_h, feat_map_w), np.float32)

        for idx, feat in enumerate(pool_results[:-1]):
            feat_img = feat[0, :, :, 0]
            b_w = block_width[idx]
            print(f"layer {idx} b_w {b_w} feat_shape {feat_img.shape}")
            for idx_r in range(feat_img.shape[0] // b_w):
                for idx_c in range(feat_img.shape[1] // b_w):
                    st = np.sum(area_size[:idx]).astype(np.int32)
                    ed = np.sum(area_size[: idx + 1]).astype(np.int32)
                    feature_map[st:ed, idx_r, idx_c] = np.array(
                        feat_img[idx_r : idx_r + b_w, idx_c : idx_c + b_w]
                    ).flatten()

                    # if idx_r == 0 and idx_c == 0:
                    #     print(f"area_size {area_size}  st{st} ed{ed}")
                    #     print(f"feature {idx_r},{idx_c} {feature_map[:, idx_r, idx_c]}")

        feature_map -= pool_results[-1][0, :, :, 0]
        feature_intensity_map = np.linalg.norm(feature_map[:, :, :], axis=0)
        feature_intensity_map /= np.max(feature_intensity_map)
        print(
            f"feature_intensity_map {feature_intensity_map.min()} {feature_intensity_map.max()} {feature_intensity_map.mean()}"
        )
        self.feature_intensity_map = feature_intensity_map

        for idx_r in range(feat_img.shape[0] // b_w):
            for idx_c in range(feat_img.shape[1] // b_w):
                feature_map[:, idx_r, idx_c] = (
                    feature_map[:, idx_r, idx_c] - pool_results[-1][0, idx_r, idx_c, 0]
                )
                feature_map[:, idx_r, idx_c] = normalize(
                    [feature_map[:, idx_r, idx_c]]
                )[0]
        self.feature_map = feature_map

        print(f"mean {pool_results[-1][0, 0, 0, 0]}")
        print(f"feature {0},{0} {feature_map[:10, 0, 0]}")

        print(
            f"feature_intensity_map {feature_map.shape} -> {feature_intensity_map.shape}, {np.max(feature_intensity_map)}, {np.min(feature_intensity_map)}"
        )

        if num_feat is not None:
            cv2.imwrite("/tmp/feature_intensity_map_he.tif", feature_intensity_map)
            top_feature_pos = np.zeros((num_feat, 2), dtype=np.int32)
            for idx in range(num_feat):
                loc = np.unravel_index(
                    np.argmax(feature_intensity_map), shape=feature_intensity_map.shape
                )
                top_feature_pos[idx, :] = np.array(loc[::-1]).astype(np.int32)
                feature_intensity_map = cv2.circle(
                    feature_intensity_map, (loc[1], loc[0]), mask_r, 0, -1
                )
            cv2.imwrite("/tmp/feature_intensity_map_mask.tif", feature_intensity_map)
        else:
            cv2.imwrite("/tmp/feature_intensity_map_pano.tif", feature_intensity_map)
            feat_h, feat_w = feature_map.shape[1:]
            n_features = int(np.ceil(feat_h / mask_r) * np.ceil(feat_w / mask_r))
            top_feature_pos = np.zeros((n_features, 2), dtype=np.int32)
            idx = 0
            for r in range(0, feat_h, mask_r):
                for c in range(0, feat_w, mask_r):
                    # print(
                    #     f"{feature_map.shape} {top_feature_pos.shape} {feat_h}x{feat_w} {n_features}, maskr {mask_r} idx {idx}  r{r}  c{c}"
                    # )
                    top_feature_pos[idx, :] = [c, r]
                    idx += 1

        # print(f"feature size {top_feature_pos.shape}")
        self.features = [
            SppFeature(
                (pos[0] * pool_size[-1], pos[1] * pool_size[-1]),
                feature_map[:, pos[1], pos[0]],
            )
            for pos in top_feature_pos
        ]

    @staticmethod
    def match(
        spp_moving: List[SppFeature], spp_fixed: List[SppFeature], top_count=30
    ) -> List[Tuple[Feature]]:
        len_fix, len_move = len(spp_fixed), len(spp_moving)
        query_map = np.zeros((len_fix, len_move), dtype=np.float32)
        for r in range(len_fix):
            for c in range(len_move):
                # print("spp feat fix ", spp_fixed[r].desc[:10])
                # print("spp feat mov ", spp_moving[c].desc[:10])
                query_map[r, c] = spp_fixed[r].distance_to(spp_moving[c])

        print(f"max {query_map.max()}")
        cv2.imwrite("/tmp/query_map.tif", query_map / query_map.max())

        matches = [(r, np.argpartition(query_map[r, :], 10)[9]) for r in range(len_fix)]
        print(matches)

        print(spp_fixed[10].desc[:20])
        print(spp_moving[matches[10][1]].desc[:20])

        matched_feats = [
            (spp_moving[m[1]], spp_fixed[m[0]]) for m in matches[:top_count]
        ]
        return matched_feats


if __name__ == "__main__":
    from sys import argv

    if len(argv) != 3:
        print(f"Usage: {argv[0]} image_he image_panorama")
        exit(-1)

    img_he = cv2.imread(argv[1], cv2.IMREAD_ANYCOLOR)
    img_pano = cv2.imread(argv[2], cv2.IMREAD_ANYCOLOR)

    spp = Spp(img_he)
    spp.compute(num_feat=100, layers=5)
    he_features = spp.features

    spp = Spp(img_pano)
    spp.compute(layers=5)
    pano_features = spp.features

    print(f"feature count: he {len(he_features)}  pano {len(pano_features)}")
    Spp.match(pano_features, he_features)
