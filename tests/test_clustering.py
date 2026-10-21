import numpy as np
import pytest
import torch

from lab03.clustering import jaccard_distance

BLOB_SIZE = 10
NUM_BLOBS = 3
K1, K2 = 5, 2


def _blobs(noise: float = 0.05, dim: int = 16) -> torch.Tensor:
    """``NUM_BLOBS`` tight groups around orthogonal axes, in blob order."""
    generator = torch.Generator().manual_seed(0)
    centers = torch.eye(dim)[:NUM_BLOBS].repeat_interleave(BLOB_SIZE, dim=0)
    return centers + noise * torch.randn(centers.shape, generator=generator)


def _random(n: int = 40, dim: int = 16) -> torch.Tensor:
    return torch.randn(n, dim, generator=torch.Generator().manual_seed(0))


def test_jaccard_shape_dtype_range() -> None:
    dist = jaccard_distance(_random(), K1, K2)

    assert dist.shape == (40, 40)
    assert dist.dtype == np.float32
    assert dist.min() >= 0
    assert dist.max() <= 1


def test_jaccard_diagonal_is_zero() -> None:
    dist = jaccard_distance(_random(), K1, K2)

    np.testing.assert_allclose(np.diag(dist), 0, atol=1e-6)


def test_jaccard_is_symmetric() -> None:
    dist = jaccard_distance(_random(), K1, K2)

    np.testing.assert_allclose(dist, dist.T, atol=1e-6)


def test_jaccard_without_query_expansion() -> None:
    dist = jaccard_distance(_random(), K1, k2=1)

    np.testing.assert_allclose(np.diag(dist), 0, atol=1e-6)
    np.testing.assert_allclose(dist, dist.T, atol=1e-6)


def test_jaccard_chunk_size_invariant() -> None:
    features = _random()

    np.testing.assert_allclose(
        jaccard_distance(features, K1, K2, chunk_size=7),
        jaccard_distance(features, K1, K2),
        atol=1e-6,
    )


def test_jaccard_is_scale_invariant() -> None:
    features = _random()

    np.testing.assert_allclose(
        jaccard_distance(3 * features, K1, K2),
        jaccard_distance(features, K1, K2),
        atol=1e-6,
    )


def test_jaccard_separates_blobs() -> None:
    # k1 + 1 = BLOB_SIZE: each k-NN list is exactly its own blob.
    dist = jaccard_distance(_blobs(), k1=BLOB_SIZE - 1, k2=K2)
    blob = np.arange(NUM_BLOBS * BLOB_SIZE) // BLOB_SIZE
    same = blob[:, None] == blob[None, :]

    assert dist[same].max() < 1
    np.testing.assert_array_equal(dist[~same], 1)


def test_jaccard_handles_duplicate_rows() -> None:
    # More copies than k1 + 1: ties must not push a row out of its own k-NN.
    features = _blobs(noise=0)

    dist = jaccard_distance(features, K1, K2)

    np.testing.assert_allclose(np.diag(dist), 0, atol=1e-6)


@pytest.mark.parametrize(
    ("k1", "k2"),
    [(40, 2), (5, 0), (5, 6)],
    ids=["k1-not-below-n", "k2-zero", "k2-above-k1"],
)
def test_jaccard_rejects_invalid_k(k1: int, k2: int) -> None:
    with pytest.raises(ValueError, match="need 1 <= k2 <= k1 < N"):
        jaccard_distance(_random(), k1, k2)


def test_jaccard_rejects_non_matrix() -> None:
    with pytest.raises(ValueError, match=r"must be \(N, D\)"):
        jaccard_distance(torch.zeros(40))
