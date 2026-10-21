from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def cluster_centroids(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """(C, D) L2-normalized mean feature of each cluster. Outliers (-1) are skipped."""
    _check_pairs(features, labels)
    inliers = labels >= 0
    if not inliers.any():
        raise ValueError("labels has no clusters, only outliers")

    features, labels = features[inliers], labels[inliers]
    num_clusters = int(labels.max()) + 1
    sums = features.new_zeros(num_clusters, features.shape[1])
    sums.index_add_(0, labels, features)
    return F.normalize(sums)  # same direction as the mean


class ClusterMemory(nn.Module):
    """Cluster-level memory ``Mc``: InfoNCE against centroids (paper eq. 4-5)."""

    def __init__(
        self, centroids: torch.Tensor, temperature: float = 0.05, momentum: float = 0.2
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.momentum = momentum
        self.register_buffer("centroids", F.normalize(centroids.detach().float()))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = features @ self.centroids.T / self.temperature
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Momentum update with the batch mean of each cluster (eq. 5).

        Call after ``backward()``: the loss graph holds the old centroids.
        """
        _check_pairs(features, labels)
        sums = torch.zeros_like(self.centroids).index_add_(0, labels, features)
        counts = torch.bincount(labels, minlength=len(self.centroids))
        present = counts > 0
        mean = sums[present] / counts[present, None]
        blended = self.momentum * self.centroids[present] + (1 - self.momentum) * mean
        self.centroids[present] = F.normalize(blended)


class InstanceMemory(nn.Module):
    """Instance-level memory ``Mi`` with hard mining (paper eq. 6-7).

    Holds one row per training image, indexed like the dataset. For each query
    the positive is the farthest instance of its own cluster and the negatives
    are the nearest instance of every other cluster in the batch. Outliers (-1)
    stay in memory but are never mined.
    """

    def __init__(
        self, features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.05
    ) -> None:
        super().__init__()
        _check_pairs(features, labels)
        self.temperature = temperature
        self.num_clusters = int(labels.max()) + 1
        self.register_buffer("features", F.normalize(features.detach().float()))
        # Outliers go to an extra column that is dropped after the reductions.
        self.register_buffer("_columns", labels.where(labels >= 0, self.num_clusters))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sim = features @ self.features.T
        columns = self._columns.expand_as(sim)
        shape = (len(features), self.num_clusters + 1)
        hardest_neg = sim.new_full(shape, -torch.inf).scatter_reduce(
            1, columns, sim, "amax", include_self=False
        )
        hardest_pos = sim.new_full(shape, torch.inf).scatter_reduce(
            1, columns, sim, "amin", include_self=False
        )

        own = F.one_hot(labels, shape[1]).bool()
        logits = hardest_neg.where(~own, hardest_pos)

        # Eq. 6 sums over the P identities of the batch, not over all clusters.
        batch_clusters, targets = torch.unique(labels, return_inverse=True)
        return F.cross_entropy(logits[:, batch_clusters] / self.temperature, targets)

    @torch.no_grad()
    def update(self, features: torch.Tensor, indices: torch.Tensor) -> None:
        """Replace the batch rows with the new features (eq. 7)."""
        self.features[indices] = F.normalize(features.float())


@dataclass(frozen=True)
class HybridLoss:
    """``total = (1 - weight) * cluster + weight * instance`` (paper eq. 3)."""

    total: torch.Tensor
    cluster: torch.Tensor
    instance: torch.Tensor


class HybridMemory(nn.Module):
    """Both memories, built from one epoch's features and pseudo-labels.

    ``weight`` is ``a`` in eq. 3: 0 keeps only the cluster loss, 1 only the
    instance loss.
    """

    def __init__(
        self,
        features: torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        temperature: float = 0.05,
        momentum: float = 0.2,
        weight: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0 <= weight <= 1:
            raise ValueError(f"weight must be in [0, 1], got {weight}")
        labels = torch.as_tensor(labels, dtype=torch.int64, device=features.device)
        self.weight = weight
        self.register_buffer("labels", labels)
        self.cluster = ClusterMemory(
            cluster_centroids(features, labels), temperature, momentum
        )
        self.instance = InstanceMemory(features, labels, temperature)

    def forward(self, features: torch.Tensor, indices: torch.Tensor) -> HybridLoss:
        """Loss for a batch; ``indices`` are dataset indices of inliers."""
        labels = self._batch_labels(indices)
        cluster = self.cluster(features, labels)
        instance = self.instance(features, labels)
        total = (1 - self.weight) * cluster + self.weight * instance
        return HybridLoss(total, cluster, instance)

    @torch.no_grad()
    def update(self, features: torch.Tensor, indices: torch.Tensor) -> None:
        """Update both memories (eq. 5 and 7). Call after ``backward()``."""
        features = features.detach()
        self.cluster.update(features, self._batch_labels(indices))
        self.instance.update(features, indices)

    def _batch_labels(self, indices: torch.Tensor) -> torch.Tensor:
        labels = self.labels[indices.to(self.labels.device)]
        if (labels < 0).any():
            raise ValueError("indices include outliers; sample only inliers")
        return labels


def _check_pairs(features: torch.Tensor, labels: torch.Tensor) -> None:
    if features.ndim != 2 or labels.shape != features.shape[:1]:
        raise ValueError(
            f"expected features (N, D) and labels (N,), got "
            f"{tuple(features.shape)} and {tuple(labels.shape)}"
        )
