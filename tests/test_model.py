import pytest
import torch

from lab03.model import PoolingFusion, SpatialAttention, build_backbone

# Small feature map: SpatialAttention has no channel-dependent parameters, so 16
# channels exercise the same code path as the 2048 of the real backbone.
FEATURE_SHAPE = (2, 16, 8, 4)
CHANNELS = FEATURE_SHAPE[1]


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


@pytest.mark.parametrize(
    ("height", "width"),
    [
        pytest.param(8, 4, id="128x64-input"),
        pytest.param(16, 8, id="256x128-input"),
        pytest.param(1, 1, id="1x1"),
    ],
)
def test_pooling_fusion_returns_one_vector_per_image(height: int, width: int) -> None:
    fusion = PoolingFusion(CHANNELS, reduction=4)
    x = torch.randn(2, CHANNELS, height, width)

    out = fusion(x)

    assert out.shape == (2, CHANNELS)


def test_pooling_fusion_uses_max_pooling() -> None:
    fusion = PoolingFusion(CHANNELS, reduction=4)
    flat = torch.ones(FEATURE_SHAPE)
    # Same mean as ``flat`` (1) but a different max (2): only GMP can tell them apart.
    peaked = torch.zeros(FEATURE_SHAPE)
    peaked[..., ::2, :] = 2

    assert not torch.allclose(fusion(flat), fusion(peaked))


def test_pooling_fusion_uses_average_pooling() -> None:
    fusion = PoolingFusion(CHANNELS, reduction=4)
    flat = torch.ones(FEATURE_SHAPE)
    # Same max as ``flat`` (1) but a different mean: only GAP can tell them apart.
    sparse = torch.zeros(FEATURE_SHAPE)
    sparse[..., 0, 0] = 1

    assert not torch.allclose(fusion(flat), fusion(sparse))


def test_pooling_fusion_trains_both_se_blocks() -> None:
    fusion = PoolingFusion(CHANNELS, reduction=4)
    x = torch.randn(FEATURE_SHAPE)

    fusion(x).sum().backward()

    for name, parameter in fusion.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.abs().sum() > 0, name
