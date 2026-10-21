import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lab03.memory import (
    ClusterMemory,
    HybridMemory,
    InstanceMemory,
    cluster_centroids,
)

DIM = 8
TAU = 0.05
# 20 instances in 4 clusters plus outliers at the end.
MEMORY_LABELS = torch.tensor([0] * 5 + [1] * 4 + [2] * 6 + [3] * 3 + [-1] * 2)
NUM_CLUSTERS = 4


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def _unit(*shape: int) -> torch.Tensor:
    return F.normalize(torch.randn(*shape))


def _reference_instance_loss(
    queries: torch.Tensor,
    labels: torch.Tensor,
    memory: torch.Tensor,
    memory_labels: torch.Tensor,
) -> torch.Tensor:
    """Eq. 6 with explicit loops over queries and batch clusters."""
    losses = []
    for q, y in zip(queries, labels.tolist(), strict=True):
        logits = []
        for c in sorted(set(labels.tolist())):
            sims = memory[memory_labels == c] @ q
            logits.append(sims.min() if c == y else sims.max())
        logits = torch.stack(logits) / TAU
        target = sorted(set(labels.tolist())).index(y)
        losses.append(-F.log_softmax(logits, dim=0)[target])
    return torch.stack(losses).mean()


def test_cluster_centroids_are_normalized_means() -> None:
    features = torch.randn(len(MEMORY_LABELS), DIM)

    centroids = cluster_centroids(features, MEMORY_LABELS)

    assert centroids.shape == (NUM_CLUSTERS, DIM)
    for c in range(NUM_CLUSTERS):
        expected = F.normalize(features[MEMORY_LABELS == c].mean(0), dim=0)
        torch.testing.assert_close(centroids[c], expected)


def test_cluster_centroids_rejects_only_outliers() -> None:
    with pytest.raises(ValueError, match="only outliers"):
        cluster_centroids(torch.randn(3, DIM), torch.tensor([-1, -1, -1]))


def test_cluster_centroids_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match=r"features \(N, D\) and labels \(N,\)"):
        cluster_centroids(torch.randn(3, DIM), torch.tensor([0, 1]))


def test_cluster_loss_is_infonce_over_centroids() -> None:
    centroids = _unit(NUM_CLUSTERS, DIM)
    queries, labels = _unit(6, DIM), torch.tensor([0, 1, 1, 2, 3, 3])
    memory = ClusterMemory(centroids, temperature=TAU)

    logits = queries @ centroids.T / TAU
    expected = -logits.log_softmax(dim=1)[torch.arange(6), labels].mean()
    torch.testing.assert_close(memory(queries, labels), expected)


def test_cluster_loss_prefers_own_centroid() -> None:
    centroids = torch.eye(NUM_CLUSTERS, DIM)
    memory = ClusterMemory(centroids, temperature=TAU)
    queries = centroids[[0, 1]]

    right = memory(queries, torch.tensor([0, 1]))
    wrong = memory(queries, torch.tensor([1, 0]))

    assert right < 1e-6
    assert wrong > 1 / TAU - 1


def test_cluster_update_blends_batch_mean_with_momentum() -> None:
    centroids = _unit(NUM_CLUSTERS, DIM)
    memory = ClusterMemory(centroids, momentum=0.2)
    features, labels = _unit(3, DIM), torch.tensor([1, 1, 3])

    memory.update(features, labels)

    mean_1 = features[:2].mean(0)
    expected_1 = F.normalize(0.2 * centroids[1] + 0.8 * mean_1, dim=0)
    expected_3 = F.normalize(0.2 * centroids[3] + 0.8 * features[2], dim=0)
    torch.testing.assert_close(memory.centroids[1], expected_1)
    torch.testing.assert_close(memory.centroids[3], expected_3)
    torch.testing.assert_close(memory.centroids[[0, 2]], centroids[[0, 2]])


def test_instance_loss_matches_reference() -> None:
    bank = _unit(len(MEMORY_LABELS), DIM)
    queries, labels = _unit(5, DIM), torch.tensor([2, 0, 2, 3, 0])
    memory = InstanceMemory(bank, MEMORY_LABELS, temperature=TAU)

    expected = _reference_instance_loss(queries, labels, bank, MEMORY_LABELS)
    torch.testing.assert_close(memory(queries, labels), expected)


def test_instance_loss_ignores_outliers_and_absent_clusters() -> None:
    bank = _unit(len(MEMORY_LABELS), DIM)
    queries, labels = _unit(2, DIM), torch.tensor([0, 2])
    memory = InstanceMemory(bank, MEMORY_LABELS, temperature=TAU)
    before = memory(queries, labels)

    # Outliers and cluster 1 (absent from the batch) are never negatives.
    bank_moved = bank.clone()
    bank_moved[(MEMORY_LABELS == -1) | (MEMORY_LABELS == 1)] = queries[0]
    after = InstanceMemory(bank_moved, MEMORY_LABELS, temperature=TAU)(queries, labels)

    torch.testing.assert_close(after, before)


def test_instance_loss_uses_farthest_positive() -> None:
    # Cluster 0 = {e0, e1}, cluster 1 = {e2}; queries are e0 and e2.
    bank = torch.eye(3, DIM)
    memory = InstanceMemory(bank, torch.tensor([0, 0, 1]), temperature=TAU)

    loss = memory(bank[[0, 2]], torch.tensor([0, 1]))

    # e0 is mined against e1 (sim 0), not its own copy (sim 1): logits tie at 0
    # and its loss is log 2. e2 has positive sim 1 and loss ~0.
    assert math.isclose(loss.item(), math.log(2) / 2, rel_tol=1e-5)


def test_instance_update_replaces_rows() -> None:
    bank = _unit(len(MEMORY_LABELS), DIM)
    memory = InstanceMemory(bank, MEMORY_LABELS)
    new, indices = 3 * torch.randn(2, DIM), torch.tensor([4, 11])

    memory.update(new, indices)

    torch.testing.assert_close(memory.features[indices], F.normalize(new))
    keep = torch.ones(len(bank), dtype=torch.bool)
    keep[indices] = False
    torch.testing.assert_close(memory.features[keep], bank[keep])


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_hybrid_loss_weights_both_terms(weight: float) -> None:
    bank = _unit(len(MEMORY_LABELS), DIM)
    memory = HybridMemory(bank, MEMORY_LABELS.numpy(), temperature=TAU, weight=weight)
    indices = torch.tensor([0, 6, 9, 16, 3])

    loss = memory(bank[indices], indices)

    labels = MEMORY_LABELS[indices]
    torch.testing.assert_close(loss.cluster, memory.cluster(bank[indices], labels))
    torch.testing.assert_close(loss.instance, memory.instance(bank[indices], labels))
    torch.testing.assert_close(
        loss.total, (1 - weight) * loss.cluster + weight * loss.instance
    )


def test_hybrid_gradient_reaches_features_only() -> None:
    bank = _unit(len(MEMORY_LABELS), DIM)
    memory = HybridMemory(bank, MEMORY_LABELS)
    indices = torch.tensor([0, 1, 6, 10, 15])
    features = _unit(len(indices), DIM).requires_grad_()

    memory(features, indices).total.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0
    assert all(not buffer.requires_grad for buffer in memory.buffers())


def test_hybrid_update_after_backward() -> None:
    bank = _unit(len(MEMORY_LABELS), DIM)
    memory = HybridMemory(bank, MEMORY_LABELS)
    indices = torch.tensor([0, 1, 6])
    features = _unit(len(indices), DIM).requires_grad_()
    centroids = memory.cluster.centroids.clone()

    memory(features, indices).total.backward()
    memory.update(features, indices)

    torch.testing.assert_close(memory.instance.features[indices], features.detach())
    assert not torch.equal(memory.cluster.centroids[:2], centroids[:2])
    torch.testing.assert_close(memory.cluster.centroids[2:], centroids[2:])


def test_hybrid_accepts_numpy_labels() -> None:
    bank = _unit(len(MEMORY_LABELS), DIM)
    memory = HybridMemory(bank, MEMORY_LABELS.numpy().astype(np.int64))

    torch.testing.assert_close(memory.labels, MEMORY_LABELS)


def test_hybrid_rejects_outlier_indices() -> None:
    memory = HybridMemory(_unit(len(MEMORY_LABELS), DIM), MEMORY_LABELS)
    indices = torch.tensor([0, len(MEMORY_LABELS) - 1])

    with pytest.raises(ValueError, match="outliers"):
        memory(_unit(2, DIM), indices)


@pytest.mark.parametrize("weight", [-0.1, 1.5])
def test_hybrid_rejects_weight_out_of_range(weight: float) -> None:
    with pytest.raises(ValueError, match=r"weight must be in \[0, 1\]"):
        HybridMemory(_unit(len(MEMORY_LABELS), DIM), MEMORY_LABELS, weight=weight)
