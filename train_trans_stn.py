import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sys import argv
import numpy as np
import cv2

from utils.dataset import load_image
from dnn.model_parameters import *

SAMPLE_WIDTH = 2
EPOCHS = 40
LR = 1e-4


def data_pipeline(data_id):
    data_dir = Path(argv[0]).parent.parent / "datasets" / "H&E_IMC" / "Pair" / data_id
    data_dir = Path(data_dir).absolute()
    main_id = str(data_id).split("_")[0]

    img_path = str(data_dir / f"HE{main_id}.tif")
    img_he = load_image(img_path, verbose=True, downsize=DOWN_SIZE)

    img_path = str(data_dir / f"{main_id}_panorama.tif")
    img_pano = load_image(img_path, verbose=True, downsize=DOWN_SIZE)

    print(f"Data id {data_id} image size he:{img_he.shape} pano:{img_pano.shape}")

    return img_pano, img_he


class GridLayer(keras.layers.Layer):
    def __init__(self, target_shape, **kwargs):
        super(GridLayer, self).__init__(**kwargs)
        self.target_shape = target_shape

        self.idx = tf.constant(
            GridLayer.idx_initializer((3, target_shape[0], target_shape[1])).reshape(
                3, -1
            )
        )

    def call(self, inputs):
        """
        input shape: 2x3
        output shape: 2xN
        """
        out = tf.matmul(inputs, self.idx)
        return out

    @staticmethod
    def idx_initializer(shape, dtype=None):
        c, h, w = shape
        matrix = np.zeros((c, h, w), dtype=np.float32)

        for r in range(h):
            for c in range(w):
                matrix[:, r, c] = np.array([r * SAMPLE_WIDTH, c * SAMPLE_WIDTH, 1])
        return matrix

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "target_shape": self.target_shape,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        target_shape = config.pop("target_shape")
        return cls(target_shape)


class SampleLayer(keras.layers.Layer):
    def __init__(self, target_shape, batch_size, **kwargs):
        super(SampleLayer, self).__init__(**kwargs)
        self.target_shape = target_shape
        self.batch_sz = batch_size

    def call(self, inputs):
        """
        input shape: 2xN, H'xW'xC
        output shape: HxWxC

        where N = HxW
        """
        idx, source = inputs
        print(f"SampleLayer idx{idx.shape} source{source.shape}")
        out = tf.zeros(
            (self.batch_sz, self.target_shape[0], self.target_shape[1], 3),
            dtype=tf.uint8,
        )

        for bz in range(self.batch_sz):
            i = -1
            for r in range(out.shape[1]):
                for c in range(out.shape[2]):
                    i += 1
                    idxr, idxc = idx[bz, :, i]
                    idxr = int(idxr + 0.5)
                    idxc = int(idxc + 0.5)
                    if (
                        idxr < 0
                        or idxc < 0
                        or idxr >= source[bz].shape[0]
                        or idxc >= source[bz].shape[1]
                    ):
                        continue
                    out[bz, r, c] = source[idxr, idxc]
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "target_shape": self.target_shape,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        target_shape = config.pop("target_shape")
        return cls(target_shape)


def get_pixel_value(img, x, y):
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def affine_grid_generator(height, width, theta):
    num_batch = tf.shape(theta)[0]

    # create normalized 2D grid
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    x_t, y_t = tf.meshgrid(x, y)

    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    theta = tf.cast(theta, "float32")
    sampling_grid = tf.cast(sampling_grid, "float32")

    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, H, W, 2)
    batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])

    return batch_grids


def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, "int32")
    max_x = tf.cast(W - 1, "int32")
    zero = tf.zeros([], dtype="int32")

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, "float32")
    y = tf.cast(y, "float32")
    x = 0.5 * ((x + 1.0) * tf.cast(max_x - 1, "float32"))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y - 1, "float32"))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), "int32")
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), "int32")
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, "float32")
    x1 = tf.cast(x1, "float32")
    y0 = tf.cast(y0, "float32")
    y1 = tf.cast(y1, "float32")

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    return out


def stn(matrix, source, target_size):
    matrix = keras.layers.Reshape((2, 3), name="matrix_reshape")(matrix)
    # matrix = tf.reshape(matrix, (-1, 2, 3))

    grid = keras.layers.Lambda(
        lambda x: affine_grid_generator(target_size[0], target_size[1], x),
        name="affine_grid_generator",
    )(matrix)
    # grid = affine_grid_generator(target_size[0], target_size[1], matrix)

    x = keras.layers.Lambda(
        lambda x: bilinear_sampler(x[0], x[1][:, 0, :, :], x[1][:, 1, :, :]),
        name="bilinear_sampler",
    )((source, grid))
    # x_s = grid[:, 0, :, :]
    # y_s = grid[:, 1, :, :]
    # x = bilinear_sampler(source, x_s, y_s)

    return x


def metric_corelation(result, target):
    """
    the input shape should be BxHxWxC
    """
    m, f = result, target
    B, H, W, C = m.shape

    mean_f = tf.reduce_mean(target, axis=(1, 2))
    mean_m = tf.reduce_mean(result, axis=(1, 2))
    print(f"metric_corelation  mean m {mean_m}  mean_f {mean_f}")

    diff_f = target - mean_f
    diff_m = result - mean_m

    print(f"metric_corelation  diff m {diff_m.shape}  diff f {diff_f.shape}")

    diff = tf.square(tf.reduce_sum(diff_m * diff_f, axis=-1))
    distance = tf.reduce_sum(tf.square(diff_m), axis=-1) * tf.reduce_sum(
        tf.square(diff_f), axis=-1
    )
    distance += 1e-4

    print(
        f"metric_corelation  diff {tf.reduce_min(diff, axis=(1, 2))},{tf.reduce_max(diff, axis=(1, 2))}"
    )
    print(
        f"metric_corelation  dist {tf.reduce_min(distance, axis=(1, 2))},{tf.reduce_max(distance, axis=(1,2))}"
    )

    corss_correlation = diff / distance

    return corss_correlation


def get_model(img_size_pano, img_size_he, batch_size):
    input_pano = keras.Input(img_size_pano, name="img_pano")
    input_he = keras.Input(img_size_he, name="img_he")

    dense_input_shape = (224, 224, 3)
    localisation_model = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=dense_input_shape
    )
    for layer in localisation_model.layers:
        layer.trainable = False

    def resize_img(x, target_size):
        img = tf.image.resize_with_crop_or_pad(x, target_size[0], target_size[1])
        return img / 255.0

    output_bias = tf.keras.initializers.Constant([1, 0, 0, 0, 1, 0])
    out_matrix_weight = keras.Sequential(
        [
            keras.layers.Lambda(
                resize_img, arguments={"target_size": dense_input_shape}
            ),
            localisation_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(6, bias_initializer=output_bias),
        ],
        name="localisation",
    )(input_pano)

    source_img_sz = (img_size_pano[0] // SAMPLE_WIDTH, img_size_pano[1] // SAMPLE_WIDTH)
    target_img_sz = (img_size_he[0] // SAMPLE_WIDTH, img_size_he[1] // SAMPLE_WIDTH)

    source = keras.layers.Lambda(
        resize_img, arguments={"target_size": source_img_sz}, name="resize_source"
    )(input_pano)
    target = keras.layers.Lambda(
        resize_img, arguments={"target_size": target_img_sz}, name="resize_targe"
    )(input_he)
    # out_target = SampleLayer(target_img_sz, batch_size)((out_sample_idx, source))
    out_target = stn(out_matrix_weight, source, target_img_sz)

    diff_correlation = keras.layers.Lambda(
        lambda x: metric_corelation(x[0], x[1]), name="metric_corelation"
    )((out_target, target))

    diff = tf.reduce_mean(diff_correlation, axis=(1, 2))

    # diff = tf.reduce_sum(tf.square(out_target - target), axis=-1)
    # diff = tf.reduce_mean(diff, axis=-1)
    # diff = tf.reduce_mean(diff, axis=-1)

    model = keras.Model(inputs=(input_pano, input_he), outputs=diff)
    model.summary()
    return model


if __name__ == "__main__":
    if len(argv) != 2:
        print(f"Usage: {argv[0]} data_id")
        exit(-1)

    data_id = argv[1]

    if EPOCHS == 1:
        tf.config.run_functions_eagerly(True)

    img_pano, img_he = data_pipeline(data_id)

    model = get_model(img_pano.shape, img_he.shape, 1)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR), loss="mse")

    if EPOCHS == 1:
        model.run_eagerly = True

    dataset = tf.data.Dataset.zip(
        (
            tf.data.Dataset.zip(
                (
                    tf.data.Dataset.from_tensor_slices(
                        np.expand_dims(img_pano, axis=0)
                    ),
                    tf.data.Dataset.from_tensor_slices(np.expand_dims(img_he, axis=0)),
                )
            ),
            tf.data.Dataset.from_tensor_slices([0]),
        )
    ).batch(1)

    model.fit(
        dataset,
        epochs=EPOCHS,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                "outputs/dnn/model_trans_stn.h5",
                monitor="loss",
                save_best_only=True,
                mode="min",
            )
        ],
    )

    model = keras.models.load_model(
        "outputs/dnn/model_trans_stn.h5",
    )
    trans_model = keras.Model(
        inputs=(
            model.get_layer("localisation").input,
            model.get_layer("resize_source").input,
        ),
        outputs=(
            model.get_layer("localisation").output,
            model.get_layer("bilinear_sampler").output,
        ),
    )

    dataset = tf.data.Dataset.zip(
        (
            tf.data.Dataset.zip(
                (
                    tf.data.Dataset.from_tensor_slices(
                        np.expand_dims(img_pano, axis=0)
                    ),
                    tf.data.Dataset.from_tensor_slices(
                        np.expand_dims(img_pano, axis=0)
                    ),
                )
            ),
            tf.data.Dataset.from_tensor_slices(np.array([0])),
        )
    ).batch(1)

    weight_trans, img_out = trans_model.predict(dataset)
    print(tf.reshape(weight_trans, (2, 3)))
    print(img_out.shape)

    cv2.imwrite(
        "outputs/dnn/stn_trans_pano.tif", np.array(img_out[0] * 255).astype(np.uint8)
    )
