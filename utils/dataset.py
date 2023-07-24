from pathlib import Path
import cv2


def read_dataset_folder(folder, sample_name=None, verbose=False):
    assert Path(folder).exists() and Path(folder).is_dir()

    folder_pattern = "[0-9]*_[0-9]*" if sample_name is None else sample_name

    for path in sorted(Path(folder).glob(folder_pattern)):
        files = sorted(path.glob("*"))
        if verbose:
            print(f"load file {(str(files[0]), str(files[1]))}")
        yield (str(files[0]), str(files[1]))


def load_image(path: str, verbose=False, downsize=1, to_gray=False):
    if verbose:
        print(f"read img {path}")

    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR)
    h, w, _ = img.shape
    img = cv2.resize(img, (w // downsize, h // downsize))

    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


if __name__ == "__main__":
    from sys import argv

    if len(argv) != 2:
        print(f"Usage: {argv[0]} dataset_folder")
        exit()

    dataset_folder = argv[1]
    data = read_dataset_folder(dataset_folder)

    print(list(data))
