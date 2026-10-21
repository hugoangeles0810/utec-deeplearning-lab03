import numpy as np
import pytest
import torch

from lab03.clustering import (
    PseudoLabels,
    cluster_dbscan,
    generate_pseudo_labels,
    jaccard_distance,
)

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


def _blobs_with_isolated(num_isolated: int = 3, dim: int = 16) -> torch.Tensor:
    """``_blobs()`` followed by ``num_isolated`` points on unused axes."""
    isolated = torch.eye(dim)[NUM_BLOBS : NUM_BLOBS + num_isolated]
    return torch.cat([_blobs(dim=dim), isolated])


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


def test_pseudo_labels_stats() -> None:
    pseudo = PseudoLabels(np.array([0, -1, 1, 0, -1, 2]))

    assert pseudo.num_clusters == 3
    assert pseudo.num_outliers == 2
    assert type(pseudo.num_clusters) is int
    assert type(pseudo.num_outliers) is int
    np.testing.assert_array_equal(pseudo.inlier_indices, [0, 2, 3, 5])


def test_pseudo_labels_all_outliers() -> None:
    pseudo = PseudoLabels(np.array([-1, -1, -1]))

    assert pseudo.num_clusters == 0
    assert pseudo.num_outliers == 3
    assert pseudo.inlier_indices.size == 0


def test_generate_pseudo_labels_recovers_blobs() -> None:
    pseudo = generate_pseudo_labels(_blobs(), k1=BLOB_SIZE - 1, k2=K2)
    labels = pseudo.labels.reshape(NUM_BLOBS, BLOB_SIZE)

    assert pseudo.num_clusters == NUM_BLOBS
    assert pseudo.num_outliers == 0
    assert (labels == labels[:, :1]).all()
    assert len(set(labels[:, 0])) == NUM_BLOBS


def test_generate_pseudo_labels_marks_isolated_points_as_outliers() -> None:
    pseudo = generate_pseudo_labels(_blobs_with_isolated(), k1=BLOB_SIZE - 1, k2=K2)
    blob_size = NUM_BLOBS * BLOB_SIZE

    assert pseudo.num_clusters == NUM_BLOBS
    np.testing.assert_array_equal(pseudo.labels[blob_size:], -1)
    np.testing.assert_array_equal(pseudo.inlier_indices, np.arange(blob_size))


def test_cluster_dbscan_returns_int64_labels() -> None:
    dist = np.array([[0, 0.1], [0.1, 0]], dtype=np.float32)

    pseudo = cluster_dbscan(dist, min_samples=2)

    assert pseudo.labels.dtype == np.int64
    np.testing.assert_array_equal(pseudo.labels, [0, 0])


def test_cluster_dbscan_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="dist must be square"):
        cluster_dbscan(np.zeros((3, 4)))
