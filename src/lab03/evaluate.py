from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ReIDMetrics:
    """mAP and CMC curve; ``cmc[k - 1]`` is the rank-k accuracy."""

    mean_ap: float
    cmc: np.ndarray

    def rank(self, k: int) -> float:
        return float(self.cmc[k - 1])


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


def compute_cmc_map(
    dist: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_cams: np.ndarray,
    g_cams: np.ndarray,
    max_rank: int = 10,
) -> ReIDMetrics:
    if dist.shape != (len(q_pids), len(g_pids)):
        raise ValueError(
            f"dist has shape {dist.shape}, expected {(len(q_pids), len(g_pids))}"
        )

    cmc_sum = np.zeros(max_rank)
    aps = []
    for q_pid, q_cam, row in zip(q_pids, q_cams, dist, strict=True):
        order = np.argsort(row, kind="stable")
        pids, cams = g_pids[order], g_cams[order]
        junk = ((pids == q_pid) & (cams == q_cam)) | (pids == -1)
        matches = (pids == q_pid)[~junk]
        if not matches.any():
            continue

        hits = np.flatnonzero(matches)  # 0-based ranks of the true matches
        cmc_sum[hits[0] :] += 1
        precisions = np.arange(1, len(hits) + 1) / (hits + 1)
        aps.append(precisions.mean())

    if not aps:
        raise ValueError("no query has a valid gallery match")
    return ReIDMetrics(float(np.mean(aps)), cmc_sum / len(aps))
