import torch
from torchvision.transforms import InterpolationMode, v2

IMAGE_SIZE = (128, 64)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(
    image_size: tuple[int, int] = IMAGE_SIZE,
    padding: int = 10,
    erasing_p: float = 0.5,
) -> v2.Compose:
    """Augmentations from SPCL: flip, pad + crop, random erasing."""
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(),
            v2.Pad(padding),
            v2.RandomCrop(image_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            v2.RandomErasing(p=erasing_p, value=0),
        ]
    )


def build_eval_transform() -> v2.Compose:
    """Deterministic: scale to [0, 1] and normalize with ImageNet stats."""
    return v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_low_res_transform(image_size: tuple[int, int] = IMAGE_SIZE) -> v2.Compose:
    """Upsamples 32x16 queries to the gallery size before eval normalization."""
    return v2.Compose(
        [
            v2.Resize(
                image_size, interpolation=InterpolationMode.BICUBIC, antialias=True
            ),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
