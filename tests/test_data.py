import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from lab03.data import UnlabeledReIDDataset

# Same alias as in conftest.py; conftest is not importable with importlib mode.
type MakeH5 = Callable[..., Path]

SIZES = {"": 10, "gallery": 4}
TRAIN_PATH = Path(__file__).parents[1] / "data" / "training.h5"

KEYS = [
    pytest.param("images", 0, id="root"),
    pytest.param("gallery/images", 10, id="prefixed"),
]


def test_len_matches_number_of_images(make_h5: MakeH5) -> None:
    dataset = UnlabeledReIDDataset(make_h5({"": 7}))

    assert len(dataset) == 7


def test_invalid_key_raises_type_error(make_h5: MakeH5) -> None:
    path = make_h5(SIZES)

    with pytest.raises(TypeError, match="'gallery' .* is not an h5py.Dataset"):
        UnlabeledReIDDataset(path, key="gallery")


@pytest.mark.parametrize(("key", "first_value"), KEYS)
def test_camera_ids_are_read_from_same_group_as_images(
    make_h5: MakeH5, key: str, first_value: int
) -> None:
    dataset = UnlabeledReIDDataset(make_h5(SIZES), key=key)

    values = np.arange(first_value, first_value + len(dataset))
    np.testing.assert_array_equal(dataset.camera_ids, values % 6 + 1)


def test_getitem_returns_uint8_chw_image_without_transform(make_h5: MakeH5) -> None:
    dataset = UnlabeledReIDDataset(make_h5(shape=(3, 16, 8)))

    image, _, _ = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 16, 8)
    assert image.dtype == torch.uint8


def test_getitem_returns_requested_index_and_int_camera_id(make_h5: MakeH5) -> None:
    dataset = UnlabeledReIDDataset(make_h5())

    _, index, camera_id = dataset[8]

    assert index == 8
    assert type(camera_id) is int
    assert camera_id == 8 % 6 + 1


@pytest.mark.parametrize(("key", "first_value"), KEYS)
def test_getitem_reads_the_requested_image(
    make_h5: MakeH5, key: str, first_value: int
) -> None:
    dataset = UnlabeledReIDDataset(make_h5(SIZES), key=key)

    for index in range(len(dataset)):
        image, _, _ = dataset[index]
        assert image.unique().tolist() == [first_value + index]


def test_transform_is_applied_to_image(make_h5: MakeH5) -> None:
    dataset = UnlabeledReIDDataset(make_h5(), transform=lambda x: x.float() / 255)

    image, _, _ = dataset[5]

    assert image.dtype == torch.float32
    torch.testing.assert_close(image, torch.full_like(image, 5 / 255))


def test_dataset_is_picklable_after_access(make_h5: MakeH5) -> None:
    dataset = UnlabeledReIDDataset(make_h5())
    _ = dataset[0]  # opens the HDF5 file

    restored = pickle.loads(pickle.dumps(dataset))
    image, index, _ = restored[5]

    assert index == 5
    assert image.unique().tolist() == [5]


def test_works_with_dataloader_workers(make_h5: MakeH5) -> None:
    dataset = UnlabeledReIDDataset(make_h5())
    _ = dataset[0]  # workers must still get a picklable copy
    loader = DataLoader(dataset, batch_size=4, num_workers=2)

    batches = list(loader)
    images = torch.cat([batch_images for batch_images, _, _ in batches])
    indices = torch.cat([batch_indices for _, batch_indices, _ in batches])
    camera_ids = torch.cat([batch_cameras for _, _, batch_cameras in batches])

    assert indices.tolist() == list(range(10))
    assert torch.equal(images[:, 0, 0, 0], indices.to(torch.uint8))
    assert camera_ids.dtype == torch.int64


@pytest.mark.integration
@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="data/training.h5 not found")
def test_real_training_file() -> None:
    dataset = UnlabeledReIDDataset(TRAIN_PATH)

    image, _, camera_id = dataset[0]

    assert len(dataset) == 11043
    assert image.shape == (3, 128, 64)
    assert image.dtype == torch.uint8
    assert 1 <= camera_id <= 6
