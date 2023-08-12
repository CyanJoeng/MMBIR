import SimpleITK as sitk
import cv2
import numpy as np


def calc_score(moving, fixed, R):
    inv_R = np.linalg.inv(R[:2, :2])

    def sample(r, c, default_v=100):
        pos = (
            np.array(
                [
                    c / fixed.shape[1] * 2 - 1 - R[2, 0],
                    r / fixed.shape[0] * 2 - 1 - R[2, 1],
                ]
            ).reshape(1, 2)
            @ inv_R
        ).flatten()
        r_m = int((pos[1] + 1) / 2 * moving.shape[0] + 0.5)
        c_m = int((pos[0] + 1) / 2 * moving.shape[1] + 0.5)

        # print(f"{r},{c}  -> {r_m},{c_m}")

        if r_m < 0 or r_m >= moving.shape[0] or c_m < 0 or c_m >= moving.shape[1]:
            return [default_v] * 3

        return moving[r_m, c_m]

    fixed_array = np.array(fixed)
    transformed_array = np.array(
        [sample(r, c) for r in range(fixed.shape[0]) for c in range(fixed.shape[1])]
    )
    transformed_array = transformed_array.reshape(fixed_array.shape)
    # cv2.imwrite("transformed_array.tif", transformed_array.astype(np.uint8))

    print(f"fixed {fixed_array.shape}  transed {transformed_array.shape}")

    # Calculate mean squared error
    mse = np.mean((fixed_array - transformed_array) ** 2)
    # Calculate normalized cross correlation
    ncc = np.corrcoef(fixed_array.flatten(), transformed_array.flatten())[0, 1]
    ncc = -(ncc**2)

    return {"mse": mse, "ncc": ncc}
