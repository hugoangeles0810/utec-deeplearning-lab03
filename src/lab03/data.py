from pathlib import Path

import h5py
import torch


class UnlabeledReIDDataset(torch.utils.data.Dataset):
    """Unlabeled images. Returns (image, index, camera_id)"""

    def __init__(self, path: Path | str, key: str = "images", transform=None):
        self.path = Path(path)
        self.key = key
        self.transform = transform

        with h5py.File(self.path, "r") as file:
            self._length = _get_dataset(file, key).shape[0]
            camera_ids_key = _sibling_key(self.key, "camera_ids")
            self.camera_ids = _get_dataset(file, camera_ids_key)[:]

        self._file = None

    def __getitem__(self, index):
        image = torch.from_numpy(self._images()[index])
        if self.transform is not None:
            image = self.transform(image)
        camera_id = int(self.camera_ids[index])
        return image, index, camera_id

    def __len__(self) -> int:
        return self._length

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def _images(self) -> h5py.Dataset:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return _get_dataset(self._file, self.key)


class ReIDEvalDataset(UnlabeledReIDDataset):
    def __init__(self, path: Path | str, key: str, transform=None):
        super().__init__(path, key, transform)
        with h5py.File(self.path, "r") as file:
            person_ids_key = _sibling_key(self.key, "person_ids")
            self.person_ids = _get_dataset(file, person_ids_key)[:]


def _get_dataset(file: h5py.File, key: str) -> h5py.Dataset:
    node = file[key]
    if not isinstance(node, h5py.Dataset):
        raise TypeError(f"{key!r} in {file.filename} is not an h5py.Dataset")
    return node


def _sibling_key(key: str, name: str) -> str:
    prefix, sep, _ = key.rpartition("/")
    return f"{prefix}{sep}{name}"
