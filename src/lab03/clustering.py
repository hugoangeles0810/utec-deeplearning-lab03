from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN


@dataclass(frozen=True)
class PseudoLabels:
    """DBSCAN output. ``labels[i] == -1`` marks an outlier."""

    labels: np.ndarray

    @property
    def num_clusters(self) -> int:
        return int(self.labels.max(initial=-1)) + 1

    @property
    def num_outliers(self) -> int:
        return int((self.labels == -1).sum())

    @property
    def inlier_indices(self) -> np.ndarray:
        return np.flatnonzero(self.labels >= 0)


def generate_pseudo_labels(
    features: torch.Tensor,
    k1: int = 30,
    k2: int = 6,
    eps: float = 0.6,
    min_samples: int = 4,
) -> PseudoLabels:
    """Jaccard distance + DBSCAN, with the paper's defaults."""
    return cluster_dbscan(jaccard_distance(features, k1, k2), eps, min_samples)


def cluster_dbscan(
    dist: np.ndarray, eps: float = 0.6, min_samples: int = 4
) -> PseudoLabels:
    """DBSCAN on a precomputed (N, N) distance matrix."""
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError(f"dist must be square, got shape {dist.shape}")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1)
    return PseudoLabels(dbscan.fit_predict(dist).astype(np.int64))


@torch.inference_mode()
def jaccard_distance(
    features: torch.Tensor, k1: int = 30, k2: int = 6, chunk_size: int = 4096
) -> np.ndarray:
    """(N, N) float32 k-reciprocal Jaccard distance (Zhong et al. 2017).

    Rows are L2-normalized first, so squared Euclidean distance is ``2 - 2 * sim``.
    The k-NN search runs on ``features.device`` in chunks of ``chunk_size`` rows;
    the rest runs on CPU and holds two dense (N, N) float32 arrays at its peak.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be (N, D), got shape {tuple(features.shape)}")
    n = features.shape[0]
    if not 1 <= k2 <= k1 < n:
        raise ValueError(f"need 1 <= k2 <= k1 < N, got k1={k1}, k2={k2}, N={n}")

    features = F.normalize(features.float())
    rank = _knn(features, k1 + 1, chunk_size)
    feats = features.cpu().numpy()

    # Encode each sample as weights over its expanded k-reciprocal set.
    mask = _reciprocal_mask(rank, k1)
    half_k = round(k1 / 2)
    half_forward, half_mask = rank[:, : half_k + 1], _reciprocal_mask(rank, half_k)
    v = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        reciprocal = rank[i][mask[i]]
        expanded = [reciprocal]
        for c in reciprocal:
            candidate = half_forward[c][half_mask[c]]
            if np.isin(candidate, reciprocal).sum() > 2 / 3 * len(candidate):
                expanded.append(candidate)
        support = np.unique(np.concatenate(expanded))
        weights = np.exp(-(2 - 2 * feats[support] @ feats[i]))
        v[i, support] = weights / weights.sum()

    # Local query expansion: average the encodings of the k2 nearest neighbours.
    if k2 > 1:
        v_qe = np.empty_like(v)
        for i in range(n):
            v_qe[i] = v[rank[i, :k2]].mean(axis=0)
        v = v_qe
        del v_qe

    # Rows sum to 1, so sum(max) = 2 - sum(min) and only the minimums are needed.
    rows, cols = np.nonzero(v)
    row_split = np.cumsum(np.bincount(rows, minlength=n))[:-1]
    row_cols = np.split(cols, row_split)
    order = np.argsort(cols, kind="stable")
    col_split = np.cumsum(np.bincount(cols, minlength=n))[:-1]
    col_rows = np.split(rows[order], col_split)
    col_values = np.split(v[rows, cols][order], col_split)
    del rows, cols, order

    dist = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        overlap = np.zeros(n, dtype=np.float32)
        for j in row_cols[i]:
            overlap[col_rows[j]] += np.minimum(v[i, j], col_values[j])
        dist[i] = 1 - overlap / (2 - overlap)
    return np.clip(dist, 0, 1, out=dist)


def _knn(features: torch.Tensor, k: int, chunk_size: int) -> np.ndarray:
    """(N, k) neighbour indices by cosine similarity; column 0 is the row itself."""
    n = features.shape[0]
    rank = torch.empty(n, k, dtype=torch.int64)
    for start in range(0, n, chunk_size):
        sim = features[start : start + chunk_size] @ features.T
        rows = torch.arange(sim.shape[0], device=sim.device)
        sim[rows, rows + start] = 2  # above any cosine, so duplicates can't tie
        rank[start : start + len(sim)] = sim.topk(k).indices.cpu()
    return rank.numpy()


def _reciprocal_mask(rank: np.ndarray, k: int) -> np.ndarray:
    """(N, k + 1) mask over ``rank[:, :k + 1]``: True where i is also in j's k-NN."""
    forward = rank[:, : k + 1]
    backward = rank[forward, : k + 1]
    return (backward == np.arange(len(rank))[:, None, None]).any(axis=2)
