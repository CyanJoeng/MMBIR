import argparse
import numpy as np
import cv2


def helper():
    parser = argparse.ArgumentParser(
        prog="Dice_Coeff",
        description="Use this program to calculate the dice coeffient of registration result",
    )
    parser.add_argument("fixed_image", help="fixed image in dataset folder")
    parser.add_argument("sampling_result", help="sampling result in output folder")
    args = parser.parse_args()
    return args


def dice_coefficient(image1, image2):
    intersection = np.sum(image1 * image2)
    total_pixels_image1 = np.sum(image1)
    total_pixels_image2 = np.sum(image2)

    dice = (2.0 * intersection) / (total_pixels_image1 + total_pixels_image2)

    return dice


def jaccard_index(image1, image2):
    intersection = np.sum(image1 * image2)
    union = np.sum(np.maximum(image1, image2))

    jaccard = intersection / union

    return jaccard


def main(args: argparse.Namespace):
    img1 = load_image(args.fixed_image)
    img2 = load_image(args.sampling_result)
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    print(f"Dice Coeffient: {dice_coefficient(img1, img2)}")
    print(f"Jaccard Index: {jaccard_index(img1, img2)}")


if __name__ == "__main__":
    from sys import path as sys_path
    from pathlib import Path

    sys_path.insert(0, str(Path(sys_path[0]).parent))
    print(sys_path)

    from utils.dataset import load_image

    args = helper()
    main(args)
