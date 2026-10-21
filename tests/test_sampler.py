from collections import Counter

import numpy as np
import pytest
import torch

from lab03.sampler import PseudoLabelBatchSampler

# Cluster sizes 6, 2, 5, 4, 3 (interleaved) plus outliers.
LABELS = np.array([0, 1, 2, 0, 3, -1, 0, 2, 4, 1, 0, 3, 2, 4, -1, 0, 2, 3, 0, 4, 2, 3])
P, Z = 2, 4


def _sampler(labels: np.ndarray = LABELS, seed: int = 0) -> PseudoLabelBatchSampler:
    generator = torch.Generator().manual_seed(seed)
    return PseudoLabelBatchSampler(labels, P, Z, generator)


def _clusters(batch: list[int]) -> Counter:
    return Counter(LABELS[batch].tolist())


def test_len_counts_full_batches_of_clusters() -> None:
    assert len(_sampler()) == 5 // P
    assert len(list(_sampler())) == len(_sampler())


def test_batches_hold_p_clusters_with_z_instances_each() -> None:
    for batch in _sampler():
        assert len(batch) == P * Z
        assert list(_clusters(batch).values()) == [Z] * P


def test_batches_group_indices_by_cluster() -> None:
    for batch in _sampler():
        labels = LABELS[batch].reshape(P, Z)
        assert (labels == labels[:, :1]).all()


def test_never_samples_outliers() -> None:
    for seed in range(10):
        for batch in _sampler(seed=seed):
            assert (LABELS[batch] >= 0).all()


def test_each_cluster_appears_at_most_once_per_pass() -> None:
    for seed in range(10):
        seen = [c for batch in _sampler(seed=seed) for c in _clusters(batch)]
        assert len(seen) == len(set(seen))


def test_large_clusters_are_sampled_without_replacement() -> None:
    for seed in range(10):
        for batch in _sampler(seed=seed):
            for c in _clusters(batch):
                picks = [i for i in batch if LABELS[i] == c]
                if (LABELS == c).sum() >= Z:
                    assert len(set(picks)) == Z


def test_small_clusters_are_sampled_with_replacement() -> None:
    labels = np.array([0, 0, 1, 1, 1, 1])

    batch = next(iter(_sampler(labels)))

    small = [i for i in batch if labels[i] == 0]
    assert len(small) == Z
    assert set(small) <= {0, 1}


def test_passes_shuffle_clusters() -> None:
    sampler = _sampler()
    passes = {tuple(c for b in sampler for c in _clusters(b)) for _ in range(20)}
    assert len(passes) > 1


def test_same_seed_gives_same_batches() -> None:
    assert list(_sampler(seed=1)) == list(_sampler(seed=1))


def test_cluster_ids_need_not_be_contiguous() -> None:
    labels = np.array([7, 7, 3, 3, 7, 3])

    batch = next(iter(_sampler(labels)))

    assert set(labels[batch].tolist()) == {3, 7}


def test_rejects_fewer_clusters_than_a_batch() -> None:
    with pytest.raises(ValueError, match="at least 2 clusters"):
        _sampler(np.array([0, 0, 0, -1]))


def test_rejects_only_outliers() -> None:
    with pytest.raises(ValueError, match="got 0"):
        _sampler(np.full(5, -1))
