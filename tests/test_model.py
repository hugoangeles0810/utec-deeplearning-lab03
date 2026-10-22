from pathlib import Path

import pytest
import torch
from torchvision.models import ResNet50_Weights

from lab03.model import (
    CAPF,
    EmbeddingHead,
    GAPHead,
    PoolingFusion,
    ReIDModel,
    SpatialAttention,
    build_backbone,
)

# Small feature map: SpatialAttention has no channel-dependent parameters, so 16
# channels exercise the same code path as the 2048 of the real backbone.
FEATURE_SHAPE = (2, 16, 8, 4)
CHANNELS = FEATURE_SHAPE[1]
# Different sizes, so swapped Linear arguments fail loudly.
IN_FEATURES, EMBED_DIM = 16, 8
IMAGE_SHAPE = (4, 3, 128, 64)

PRETRAINED = ResNet50_Weights.IMAGENET1K_V1
PRETRAINED_PATH = Path(torch.hub.get_dir()) / "checkpoints" / Path(PRETRAINED.url).name


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


@pytest.mark.parametrize("identity_init", [False, True])
def test_spatial_attention_conv_receives_gradients(identity_init: bool) -> None:
    attention = SpatialAttention(identity_init=identity_init)
    x = torch.randn(FEATURE_SHAPE)

    attention(x).sum().backward()

    assert attention.conv.weight.grad is not None
    assert attention.conv.weight.grad.abs().sum() > 0


def test_spatial_attention_identity_init_starts_as_identity() -> None:
    attention = SpatialAttention(identity_init=True)
    x = torch.randn(FEATURE_SHAPE)

    torch.testing.assert_close(attention(x), x)


def test_spatial_attention_identity_init_mask_lies_in_0_2() -> None:
    attention = SpatialAttention(identity_init=True)
    with torch.no_grad():
        attention.conv.weight.normal_(std=0.5)
    x = torch.rand(FEATURE_SHAPE) + 0.1

    mask = attention(x) / x

    assert ((mask > 0) & (mask < 2)).all()
    assert (mask > 1).any()


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


def test_pooling_fusion_without_se_adds_gap_and_gmp() -> None:
    fusion = PoolingFusion(CHANNELS, reduction=4, se=False)
    x = torch.randn(FEATURE_SHAPE)

    out = fusion(x)

    assert list(fusion.parameters()) == []
    torch.testing.assert_close(out, x.mean(dim=(2, 3)) + x.amax(dim=(2, 3)))


def test_balanced_pooling_fusion_weighs_gap_and_gmp_equally() -> None:
    fusion = PoolingFusion(CHANNELS, reduction=4, se=False, balanced=True)
    x = torch.randn(FEATURE_SHAPE)

    out = fusion(x)

    gap = torch.nn.functional.normalize(x.mean(dim=(2, 3)), dim=1)
    gmp = torch.nn.functional.normalize(x.amax(dim=(2, 3)), dim=1)
    torch.testing.assert_close(out, gap + gmp)


def test_embedding_head_output_shape() -> None:
    head = EmbeddingHead(IN_FEATURES, EMBED_DIM)
    x = torch.randn(4, IN_FEATURES)

    out = head(x)

    assert out.shape == (4, EMBED_DIM)


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_embedding_head_outputs_unit_norm(mode: str) -> None:
    head = EmbeddingHead(IN_FEATURES, EMBED_DIM).train(mode == "train")
    x = torch.randn(4, IN_FEATURES)

    out = head(x)

    torch.testing.assert_close(out.norm(dim=1), torch.ones(4))


def test_embedding_head_in_eval_does_not_depend_on_batch() -> None:
    head = EmbeddingHead(IN_FEATURES, EMBED_DIM).eval()
    x = torch.randn(4, IN_FEATURES)

    alone = head(x[:1])
    in_batch = head(x)[:1]

    torch.testing.assert_close(alone, in_batch)


def test_embedding_head_in_train_depends_on_batch() -> None:
    # Documents why feature extraction must run in eval mode.
    head = EmbeddingHead(IN_FEATURES, EMBED_DIM).train()
    x = torch.randn(4, IN_FEATURES)

    small_batch = head(x[:2])[:1]
    full_batch = head(x)[:1]

    assert not torch.allclose(small_batch, full_batch)


def test_embedding_head_trainable_parameters() -> None:
    head = EmbeddingHead(IN_FEATURES, EMBED_DIM)

    trainable = {name for name, p in head.named_parameters() if p.requires_grad}

    # No Linear bias (BN cancels it) and a frozen BN bias, as in SPCL.
    assert trainable == {"fc.weight", "bnorm.weight"}


def test_embedding_head_rejects_single_sample_batch_in_train() -> None:
    head = EmbeddingHead(IN_FEATURES, EMBED_DIM).train()

    with pytest.raises(ValueError, match="more than 1 value per channel"):
        head(torch.randn(1, IN_FEATURES))


def _weighted_loss(out: torch.Tensor) -> torch.Tensor:
    # A plain out.sum() can have zero gradient through BN + L2 normalization.
    return (out * torch.randn_like(out)).sum()


def test_capf_returns_unit_norm_embeddings() -> None:
    capf = CAPF(CHANNELS, EMBED_DIM, reduction=4)
    x = torch.randn(4, CHANNELS, 8, 4)

    out = capf(x)

    assert out.shape == (4, EMBED_DIM)
    torch.testing.assert_close(out.norm(dim=1), torch.ones(4))


def test_capf_trains_all_submodules() -> None:
    capf = CAPF(CHANNELS, EMBED_DIM, reduction=4)
    x = torch.randn(4, CHANNELS, 8, 4)

    _weighted_loss(capf(x)).backward()

    for name, parameter in capf.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert parameter.grad.abs().sum() > 0, name


def test_capf_without_spatial_attention_skips_the_mask() -> None:
    capf = CAPF(CHANNELS, EMBED_DIM, reduction=4, spatial_attention=False).eval()
    x = torch.randn(4, CHANNELS, 8, 4)

    assert not any(name.startswith("attention.") for name in capf.state_dict())
    torch.testing.assert_close(capf(x), capf.head(capf.fusion(x)))


@pytest.mark.parametrize("embed_dim", [2048, 128])
def test_reid_model_returns_unit_norm_embeddings(embed_dim: int) -> None:
    model = ReIDModel(embed_dim=embed_dim)
    x = torch.randn(IMAGE_SHAPE)

    out = model(x)

    assert out.shape == (IMAGE_SHAPE[0], embed_dim)
    torch.testing.assert_close(out.norm(dim=1), torch.ones(IMAGE_SHAPE[0]))


def test_reid_model_in_eval_does_not_depend_on_batch() -> None:
    model = ReIDModel().eval()
    x = torch.randn(IMAGE_SHAPE)

    with torch.inference_mode():
        alone = model(x[:1])
        in_batch = model(x)[:1]

    torch.testing.assert_close(alone, in_batch)


def test_reid_model_trains_backbone_and_capf() -> None:
    model = ReIDModel()
    x = torch.randn(IMAGE_SHAPE)

    _weighted_loss(model(x)).backward()

    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    assert frozen == {"capf.head.bnorm.bias"}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert parameter.grad.abs().sum() > 0, name


def test_gap_head_pools_by_average() -> None:
    head = GAPHead(CHANNELS, EMBED_DIM).eval()
    x = torch.randn(4, CHANNELS, 8, 4)

    torch.testing.assert_close(head(x), head.head(x.mean(dim=(2, 3))))


def test_reid_model_without_capf_is_the_gap_baseline() -> None:
    model = ReIDModel(capf=False)
    x = torch.randn(IMAGE_SHAPE)

    out = model(x)

    assert isinstance(model.gap, GAPHead)
    assert not any(name.startswith("capf.") for name in model.state_dict())
    torch.testing.assert_close(out.norm(dim=1), torch.ones(IMAGE_SHAPE[0]))
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    assert frozen == {"gap.head.bnorm.bias"}


@pytest.mark.parametrize(
    ("spatial_attention", "se", "removed"),
    [
        pytest.param(False, True, "capf.attention.", id="no-spatial-attention"),
        pytest.param(True, False, "capf.fusion.se_", id="no-se"),
    ],
)
def test_reid_model_ablations_drop_their_parameters(
    spatial_attention: bool, se: bool, removed: str
) -> None:
    model = ReIDModel(spatial_attention=spatial_attention, se=se)

    names = set(model.state_dict())

    assert not any(name.startswith(removed) for name in names)
    assert set(ReIDModel().state_dict()) > names


@pytest.mark.parametrize(
    "flags",
    [
        {"spatial_attention": False},
        {"se": False},
        {"identity_attention": True},
        {"balanced_fusion": True},
    ],
    ids=["spatial", "se", "identity", "balanced"],
)
def test_reid_model_capf_flags_require_capf(flags: dict[str, bool]) -> None:
    with pytest.raises(ValueError, match="only apply when capf=True"):
        ReIDModel(capf=False, **flags)


def test_identity_attention_requires_spatial_attention() -> None:
    with pytest.raises(ValueError, match="requires spatial_attention"):
        CAPF(
            CHANNELS,
            EMBED_DIM,
            reduction=4,
            spatial_attention=False,
            identity_attention=True,
        )


def test_capf_fixes_keep_the_parameters() -> None:
    # Checkpoints of the fixed model load into the same state-dict layout.
    fixed = ReIDModel(identity_attention=True, balanced_fusion=True)

    assert set(fixed.state_dict()) == set(ReIDModel().state_dict())


@pytest.mark.integration
@pytest.mark.skipif(
    not PRETRAINED_PATH.exists(),
    reason=f"{PRETRAINED_PATH.name} not cached; ReIDModel({PRETRAINED}) downloads it",
)
def test_reid_model_loads_imagenet_weights() -> None:
    model = ReIDModel(PRETRAINED).eval()
    stem_bn = model.backbone[1]
    assert isinstance(stem_bn, torch.nn.BatchNorm2d)

    with torch.inference_mode():
        out = model(torch.randn(IMAGE_SHAPE))

    # Random init leaves BN running means at zero; ImageNet statistics do not.
    assert stem_bn.running_mean is not None
    assert stem_bn.running_mean.abs().sum() > 0
    assert out.shape == (IMAGE_SHAPE[0], 2048)
