from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler


class PseudoLabelBatchSampler(Sampler[list[int]]):
    """PK batches: ``num_clusters`` pseudo-identities x ``num_instances`` images.

    One pass visits every cluster at most once, in random order, and drops the
    clusters left over after the last full batch. Clusters smaller than
    ``num_instances`` are sampled with replacement. Outliers (-1) are never
    sampled. Yields dataset indices grouped by cluster.
    """

    def __init__(
        self,
        labels: np.ndarray,
        num_clusters: int = 16,
        num_instances: int = 16,
        generator: torch.Generator | None = None,
    ) -> None:
        labels = np.asarray(labels)
        inliers = np.flatnonzero(labels >= 0)
        inliers = inliers[np.argsort(labels[inliers], kind="stable")]
        _, starts = np.unique(labels[inliers], return_index=True)
        self._members = np.split(inliers, starts[1:]) if len(inliers) else []
        if len(self._members) < num_clusters:
            raise ValueError(
                f"need at least {num_clusters} clusters per batch, "
                f"got {len(self._members)}"
            )

        self.num_clusters = num_clusters
        self.num_instances = num_instances
        self.generator = generator

    def __len__(self) -> int:
        return len(self._members) // self.num_clusters

    def __iter__(self) -> Iterator[list[int]]:
        order = torch.randperm(len(self._members), generator=self.generator)
        for start in range(0, len(self) * self.num_clusters, self.num_clusters):
            batch = []
            for c in order[start : start + self.num_clusters].tolist():
                batch.extend(self._pick(self._members[c]))
            yield batch

    def _pick(self, members: np.ndarray) -> list[int]:
        n, k = len(members), self.num_instances
        if n >= k:
            picks = torch.randperm(n, generator=self.generator)[:k]
        else:
            picks = torch.randint(n, (k,), generator=self.generator)
        return members[picks.numpy()].tolist()
