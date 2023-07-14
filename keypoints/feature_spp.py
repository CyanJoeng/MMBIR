from typing import List, Tuple
from typing_extensions import override

from keras import layers
from feature import Feature, PointFeature

import numpy as np
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
        return np.linalg.norm(diff)


class Spp:
    def __init__(self, img) -> None:
        self.img = img
        self.features = None

    def compute(self, layers=5, num_feat=None):
        print("spp compute")
        print(f"\tnum_feat {num_feat}")

        img_h, img_w, _ = self.img.shape
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        input_data = np.array(gray_img).astype(np.float32)
        input_data = input_data.reshape([1, img_h, img_w, 1])
        print(f"\tinput data shape {input_data.shape}")

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
            print(
                f"\tfeatu map  st {feat_st} ed {feat_ed} off {block_off_tl},{block_off_br}"
            )

            for idx_r in range(h_bs, img_h - h_bs + 1):
                for idx_c in range(h_bs, img_w - h_bs + 1):
                    feature_map[feat_st:feat_ed, idx_r, idx_c] = pool_results[idx][
                        0,
                        idx_r - block_off_tl : idx_r + block_off_br,
                        idx_c - block_off_tl : idx_c + block_off_br,
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
            f"\tfeature_intensity_map {feature_intensity_map.min()} {feature_intensity_map.max()} {feature_intensity_map.mean()}"
        )
        feature_intensity_map /= np.max(feature_intensity_map)
        print(
            f"\tfeature_intensity_map {feature_map.shape} -> {feature_intensity_map.shape}, {np.max(feature_intensity_map)}, {np.min(feature_intensity_map)}"
        )
        self.feature_intensity_map = feature_intensity_map

        # select features based on intensity of features
        def select_feature(feature_pos, intensity_map):
            for idx in range(feature_pos.shape[0]):
                y, x = np.unravel_index(
                    np.argmax(intensity_map), shape=intensity_map.shape
                )
                feature_pos[idx, :] = np.array([x, y]).astype(np.int32)
                intensity_map = cv2.circle(intensity_map, (x, y), min(h_bs, 8), 0, -1)
            return intensity_map

        if num_feat is not None:
            cv2.imwrite("/tmp/feature_intensity_map_he.tif", feature_intensity_map)
            top_feature_pos = np.zeros((num_feat, 2), dtype=np.int32)
            masked_intensity_map = select_feature(
                top_feature_pos, feature_intensity_map
            )
            cv2.imwrite("/tmp/feature_intensity_map_mask_he.tif", feature_intensity_map)
        else:
            cv2.imwrite("/tmp/feature_intensity_map_pano.tif", feature_intensity_map)
            top_feature_pos = np.zeros((200, 2), dtype=np.int32)
            masked_intensity_map = select_feature(
                top_feature_pos, feature_intensity_map
            )
            cv2.imwrite(
                "/tmp/feature_intensity_map_mask_pano.tif", feature_intensity_map
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
    def match(
        spp_moving: List[SppFeature], spp_fixed: List[SppFeature], top_count=30
    ) -> List[Tuple[SppFeature]]:
        print("spp match")
        len_fix, len_move = len(spp_fixed), len(spp_moving)
        query_map = np.zeros((len_fix, len_move), dtype=np.float32)
        for r in range(len_fix):
            for c in range(len_move):
                # print("spp feat fix ", spp_fixed[r].desc[:10])
                # print("spp feat mov ", spp_moving[c].desc[:10])
                query_map[r, c] = spp_fixed[r].distance_to(spp_moving[c])

        print(f"\tmax of query_map {query_map.max()}")
        cv2.imwrite("/tmp/query_map.tif", query_map / query_map.max())

        matches = [
            (r, np.argpartition(query_map[r, :], (1, 3))[:3]) for r in range(len_fix)
        ]
        print("\tmatches", f"len {len(matches)}")
        refined_matches = [
            (r, cs[0])
            # [f"{query_map[r, c]:.1f}" for c in cs]
            for r, cs in matches
            if query_map[r, cs[0]] < query_map[r, cs[1]] and query_map[r, cs[0]] < 1
        ]
        print("\tmatches", f"len {len(refined_matches)}")

        print("\tspp fixed desc example ", spp_fixed[10].desc[:20])
        print(
            "\tmatched spp moving desc example ",
            spp_moving[refined_matches[10][1]].desc[:20],
        )

        matched_feats = [(spp_moving[m[1]], spp_fixed[m[0]]) for m in refined_matches]
        print(f"\tfinal matches size {len(matches)} -> {len(matched_feats)}")

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
