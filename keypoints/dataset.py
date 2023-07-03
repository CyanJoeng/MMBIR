from pathlib import Path
from cv2 import imread


def read_dataset_folder(folder, verbose=False):
    assert Path(folder).exists() and Path(folder).is_dir()

    for path in sorted(Path(folder).glob("[0-9]*_[0-9]*")):
        files = sorted(path.glob("*"))
        if verbose:
            print(f"load file {str(path)}")
        yield (str(files[0]), str(files[1]))


def load_image(path: str, verbose=False):
    if verbose:
        print(f"read img {path}")
    return imread(path)


if __name__ == "__main__":
    from sys import argv

    if len(argv) != 2:
        print(f"Usage: {argv[0]} dataset_folder")
        exit()

    dataset_folder = argv[1]
    data = read_dataset_folder(dataset_folder)

    print(list(data))
