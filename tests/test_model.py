import torch

from lab03.model import build_backbone


def test_backbone_uses_last_stride_one() -> None:
    backbone = build_backbone()
    x = torch.randn(2, 3, 128, 64)

    features = backbone(x)

    assert features.shape == (2, 2048, 8, 4)
