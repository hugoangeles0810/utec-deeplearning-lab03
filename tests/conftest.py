from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest

type MakeH5 = Callable[..., Path]


@pytest.fixture
def make_h5(tmp_path: Path) -> MakeH5:
    """Factory for small fake Re-ID files.

    Each image is filled with a single pixel value, unique across groups and
    assigned in insertion order of ``sizes``: with ``{"": 10, "gallery": 4}``,
    ``images`` holds 0..9 and ``gallery/images`` holds 10..13. The camera id of
    an image with value ``v`` is ``v % 6 + 1``.

    ``person_ids=True`` adds labels ``v // 2`` (pairs of images share an id).
    ``low_res_shape`` adds ``images_low_resolution`` with the same pixel values.
    """

    def _make(
        sizes: dict[str, int] | None = None,
        shape: tuple[int, int, int] = (3, 8, 4),
        person_ids: bool = False,
        low_res_shape: tuple[int, int, int] | None = None,
    ) -> Path:
        sizes = sizes or {"": 10}
        assert sum(sizes.values()) <= 256, "pixel values must fit in uint8"

        path = tmp_path / "fake.h5"
        start = 0
        with h5py.File(path, "w") as file:
            for prefix, size in sizes.items():
                values = np.arange(start, start + size, dtype=np.uint8)
                images = np.broadcast_to(values[:, None, None, None], (size, *shape))
                file.create_dataset(_join(prefix, "images"), data=images)
                file.create_dataset(_join(prefix, "camera_ids"), data=values % 6 + 1)
                if person_ids:
                    labels = (values // 2).astype(np.int32)
                    file.create_dataset(_join(prefix, "person_ids"), data=labels)
                if low_res_shape is not None:
                    low_res = np.broadcast_to(
                        values[:, None, None, None], (size, *low_res_shape)
                    )
                    key = _join(prefix, "images_low_resolution")
                    file.create_dataset(key, data=low_res)
                start += size
        return path

    return _make


def _join(prefix: str, name: str) -> str:
    return "/".join(filter(None, [prefix, name]))
