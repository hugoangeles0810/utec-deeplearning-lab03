import pytest
import torch

from lab03.model import SpatialAttention, build_backbone

# Small feature map: SpatialAttention has no channel-dependent parameters, so 16
# channels exercise the same code path as the 2048 of the real backbone.
FEATURE_SHAPE = (2, 16, 8, 4)


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def test_backbone_uses_last_stride_one() -> None:
    backbone = build_backbone()
    x = torch.randn(2, 3, 128, 64)

    features = backbone(x)

    assert features.shape == (2, 2048, 8, 4)


def test_spatial_attention_keeps_shape() -> None:
    attention = SpatialAttention()
    x = torch.randn(FEATURE_SHAPE)

    out = attention(x)

    assert out.shape == x.shape


def test_spatial_attention_mask_lies_in_unit_interval() -> None:
    attention = SpatialAttention()
    x = torch.rand(FEATURE_SHAPE) + 0.1  # positive, so out / x recovers the mask

    mask = attention(x) / x

    assert ((mask > 0) & (mask < 1)).all()


def test_spatial_attention_mask_is_shared_across_channels() -> None:
    attention = SpatialAttention()
    x = torch.rand(FEATURE_SHAPE) + 0.1

    mask = attention(x) / x

    torch.testing.assert_close(mask, mask[:, :1].expand_as(mask))


def test_spatial_attention_conv_receives_gradients() -> None:
    attention = SpatialAttention()
    x = torch.randn(FEATURE_SHAPE)

    attention(x).sum().backward()

    assert attention.conv.weight.grad is not None
    assert attention.conv.weight.grad.abs().sum() > 0
