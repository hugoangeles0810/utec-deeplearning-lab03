from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from lab03.data import ReIDEvalDataset


@dataclass(frozen=True)
class ReIDMetrics:
    """mAP and CMC curve; ``cmc[k - 1]`` is the rank-k accuracy."""

    mean_ap: float
    cmc: np.ndarray

    def rank(self, k: int) -> float:
        return float(self.cmc[k - 1])

    def summary(self, ranks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
        """JSON-friendly ``{"mAP": ..., "rank1": ..., ...}``."""
        return {"mAP": self.mean_ap} | {f"rank{k}": self.rank(k) for k in ranks}


@dataclass(frozen=True)
class MapDrop:
    """mAP lost when going from ``original`` to ``degraded`` queries."""

    absolute: float  # in mAP units, e.g. 0.12 = 12 points
    relative: float  # fraction of the original mAP


def map_drop(original: ReIDMetrics, degraded: ReIDMetrics) -> MapDrop:
    # mean_ap > 0 always: every scored query has at least one match in the gallery.
    absolute = original.mean_ap - degraded.mean_ap
    return MapDrop(absolute, absolute / original.mean_ap)


@torch.inference_mode()
def extract_features(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> torch.Tensor:
    """(N, D) float32 embeddings on CPU, in dataset order. Restores model mode."""
    was_training = model.training
    model.eval()
    try:
        features = None
        for images, indices, _ in loader:
            out = model(images.to(device, non_blocking=True)).float().cpu()
            if features is None:
                features = torch.empty(len(loader.dataset), out.shape[1])
            features[indices] = out
    finally:
        model.train(was_training)

    if features is None:
        raise ValueError("loader yielded no batches")
    return features


def pairwise_distance(query: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
    """(Q, G) cosine distance between L2-normalized rows."""
    return (1 - F.normalize(query) @ F.normalize(gallery).T).clamp_(min=0)


@dataclass(frozen=True)
class QueryScores:
    """Per-query AP and first-hit rank; NaN and -1 for queries without a match."""

    ap: np.ndarray
    first_hit: np.ndarray  # 0-based rank of the first true match

    @property
    def valid(self) -> np.ndarray:
        return self.first_hit >= 0


def valid_ranking(
    row: np.ndarray,
    q_pid: int,
    q_cam: int,
    g_pids: np.ndarray,
    g_cams: np.ndarray,
) -> np.ndarray:
    """Gallery indices by increasing distance, without junk: same id and camera
    as the query, or id -1."""
    order = np.argsort(row, kind="stable")
    pids, cams = g_pids[order], g_cams[order]
    junk = ((pids == q_pid) & (cams == q_cam)) | (pids == -1)
    return order[~junk]


def score_queries(
    dist: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_cams: np.ndarray,
    g_cams: np.ndarray,
) -> QueryScores:
    if dist.shape != (len(q_pids), len(g_pids)):
        raise ValueError(
            f"dist has shape {dist.shape}, expected {(len(q_pids), len(g_pids))}"
        )

    ap = np.full(len(q_pids), np.nan)
    first_hit = np.full(len(q_pids), -1)
    for i, (q_pid, q_cam, row) in enumerate(zip(q_pids, q_cams, dist, strict=True)):
        ranked = valid_ranking(row, q_pid, q_cam, g_pids, g_cams)
        hits = np.flatnonzero(g_pids[ranked] == q_pid)  # 0-based ranks of matches
        if len(hits):
            first_hit[i] = hits[0]
            ap[i] = (np.arange(1, len(hits) + 1) / (hits + 1)).mean()
    return QueryScores(ap, first_hit)


def compute_cmc_map(
    dist: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_cams: np.ndarray,
    g_cams: np.ndarray,
    max_rank: int = 10,
) -> ReIDMetrics:
    scores = score_queries(dist, q_pids, g_pids, q_cams, g_cams)
    if not scores.valid.any():
        raise ValueError("no query has a valid gallery match")

    first_hit = scores.first_hit[scores.valid]
    cmc = (first_hit[:, None] < np.arange(1, max_rank + 1)).mean(axis=0)
    return ReIDMetrics(float(scores.ap[scores.valid].mean()), cmc)


def evaluate(
    model: nn.Module,
    query_loader: DataLoader,
    gallery_loader: DataLoader,
    device: torch.device,
    max_rank: int = 10,
) -> ReIDMetrics:
    """Extracts both splits and scores them. Reads pids and cams from the datasets."""
    q_pids, q_cams = _labels(query_loader)
    g_pids, g_cams = _labels(gallery_loader)

    query = extract_features(model, query_loader, device)
    gallery = extract_features(model, gallery_loader, device)
    dist = pairwise_distance(query, gallery).numpy()
    return compute_cmc_map(dist, q_pids, g_pids, q_cams, g_cams, max_rank)


def _labels(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    dataset = loader.dataset
    if not isinstance(dataset, ReIDEvalDataset):
        raise TypeError(f"expected a ReIDEvalDataset, got {type(dataset).__name__}")
    return dataset.person_ids, dataset.camera_ids
