import torch
from torchvision.transforms import InterpolationMode, v2

IMAGE_SIZE = (128, 64)  # native size of the dataset images
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resize(image_size: tuple[int, int]) -> v2.Resize:
    """Bicubic, as in SPCL; a no-op on images that already have ``image_size``."""
    return v2.Resize(
        image_size, interpolation=InterpolationMode.BICUBIC, antialias=True
    )


def build_train_transform(
    image_size: tuple[int, int] = IMAGE_SIZE,
    padding: int = 10,
    erasing_p: float = 0.5,
) -> v2.Compose:
    """Augmentations from SPCL: resize, flip, pad + crop, random erasing."""
    return v2.Compose(
        [
            _resize(image_size),
            v2.RandomHorizontalFlip(),
            v2.Pad(padding),
            v2.RandomCrop(image_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            v2.RandomErasing(p=erasing_p, value=0),
        ]
    )


def build_eval_transform(image_size: tuple[int, int] = IMAGE_SIZE) -> v2.Compose:
    """Deterministic: resize, scale to [0, 1] and normalize with ImageNet stats."""
    return v2.Compose(
        [
            _resize(image_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_low_res_transform(image_size: tuple[int, int] = IMAGE_SIZE) -> v2.Compose:
    """Upsamples 32x16 queries to the model input size before eval normalization."""
    return build_eval_transform(image_size)
