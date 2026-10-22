from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from lab03.data import UnlabeledReIDDataset
from lab03.transforms import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transform,
    build_low_res_transform,
    build_train_transform,
)

# Same alias as in conftest.py; conftest is not importable with importlib mode.
type MakeH5 = Callable[..., Path]

LOW_RES_SIZE = (32, 16)
MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)
# A black pixel (padding) after normalization.
BLACK = -MEAN / STD


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def _random_image(size: tuple[int, int] = IMAGE_SIZE) -> torch.Tensor:
    return torch.randint(0, 256, (3, *size), dtype=torch.uint8)


def _constant_image(value: int, size: tuple[int, int] = IMAGE_SIZE) -> torch.Tensor:
    return torch.full((3, *size), value, dtype=torch.uint8)


def _normalize(image: torch.Tensor) -> torch.Tensor:
    return (image.float() / 255 - MEAN) / STD


def _close(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    # allclose's default atol is too strict for float32 values near zero.
    return torch.allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize(
    ("transform", "input_size"),
    [
        pytest.param(build_train_transform(), IMAGE_SIZE, id="train"),
        pytest.param(build_eval_transform(), IMAGE_SIZE, id="eval"),
        pytest.param(build_low_res_transform(), LOW_RES_SIZE, id="low-res"),
    ],
)
def test_transform_returns_float_image_of_model_size(
    transform: Callable[[torch.Tensor], torch.Tensor], input_size: tuple[int, int]
) -> None:
    out = transform(_random_image(input_size))

    assert out.dtype == torch.float32
    assert out.shape == (3, *IMAGE_SIZE)


@pytest.mark.parametrize(
    "build", [build_train_transform, build_eval_transform, build_low_res_transform]
)
def test_transforms_resize_to_the_requested_size(
    build: Callable[..., Callable[[torch.Tensor], torch.Tensor]],
) -> None:
    out = build(image_size=(256, 128))(_random_image())

    assert out.shape == (3, 256, 128)


def test_eval_transform_upsamples_like_pil_bicubic() -> None:
    image = _random_image()
    upsampled = [
        np.asarray(Image.fromarray(channel.numpy()).resize((128, 256), Image.BICUBIC))
        for channel in image
    ]
    expected = _normalize(torch.from_numpy(np.stack(upsampled)))

    out = build_eval_transform((256, 128))(image)

    torch.testing.assert_close(out, expected, rtol=0, atol=1.01 / 255 / STD.min())


def test_eval_transform_scales_and_normalizes_with_imagenet_stats() -> None:
    image = _random_image()

    out = build_eval_transform()(image)

    torch.testing.assert_close(out, _normalize(image))


def test_eval_transform_maps_imagenet_mean_close_to_zero() -> None:
    mean_pixel = (MEAN * 255).round().to(torch.uint8).expand(3, *IMAGE_SIZE)

    out = build_eval_transform()(mean_pixel)

    assert out.abs().max() < 0.01


def test_train_transform_is_random() -> None:
    transform = build_train_transform()
    image = _random_image()

    outputs = [transform(image) for _ in range(5)]

    assert any(not torch.equal(outputs[0], out) for out in outputs[1:])


def test_train_transform_without_crop_or_erasing_only_flips() -> None:
    transform = build_train_transform(padding=0, erasing_p=0)
    image = _random_image()
    original, flipped = _normalize(image), _normalize(image.flip(-1))

    outputs = [transform(image) for _ in range(20)]

    assert all(_close(out, original) or _close(out, flipped) for out in outputs)
    assert any(_close(out, original) for out in outputs)
    assert any(_close(out, flipped) for out in outputs)


def test_train_transform_shifts_image_into_black_padding() -> None:
    transform = build_train_transform(erasing_p=0)
    image = _constant_image(200)

    outputs = [transform(image) for _ in range(20)]

    # Only the padding is black, so a black border means the crop was shifted.
    is_black = [torch.isclose(out, BLACK.expand_as(out)).all(dim=0) for out in outputs]
    assert any(black.any() for black in is_black)
    # The crop never leaves the padded image: the person is always visible.
    assert all(not black.all() for black in is_black)


def test_train_transform_erases_with_zeros_after_normalization() -> None:
    transform = build_train_transform(padding=0, erasing_p=1)
    image = _constant_image(200)  # normalizes to a nonzero value

    out = transform(image)

    # Zero in normalized space is the ImageNet mean color.
    erased = (out == 0).all(dim=0)
    assert erased.any()
    assert not erased.all()


def test_train_transform_without_erasing_has_no_zeros() -> None:
    transform = build_train_transform(padding=0, erasing_p=0)
    image = _constant_image(200)

    outputs = [transform(image) for _ in range(20)]

    assert all((out != 0).all() for out in outputs)


def test_low_res_transform_upsamples_like_pil_bicubic() -> None:
    image = _random_image(LOW_RES_SIZE)
    # torchvision's uint8 bicubic matches PIL, which SPCL-era pipelines used.
    height, width = IMAGE_SIZE
    upsampled = [
        np.asarray(
            Image.fromarray(channel.numpy()).resize((width, height), Image.BICUBIC)
        )
        for channel in image
    ]
    expected = _normalize(torch.from_numpy(np.stack(upsampled)))

    out = build_low_res_transform()(image)

    # Rounding may differ by one gray level.
    torch.testing.assert_close(out, expected, rtol=0, atol=1.01 / 255 / STD.min())


def test_low_res_transform_keeps_constant_image_constant() -> None:
    out = build_low_res_transform()(_constant_image(200, LOW_RES_SIZE))

    torch.testing.assert_close(out, _normalize(_constant_image(200)))


def test_dataloader_yields_normalized_batches(make_h5: MakeH5) -> None:
    dataset = UnlabeledReIDDataset(
        make_h5({"": 8}, shape=(3, *IMAGE_SIZE)), transform=build_train_transform()
    )
    loader = DataLoader(dataset, batch_size=4)

    images, indices, camera_ids = next(iter(loader))

    assert images.shape == (4, 3, *IMAGE_SIZE)
    assert images.dtype == torch.float32
    assert indices.tolist() == [0, 1, 2, 3]
    assert camera_ids.shape == (4,)
