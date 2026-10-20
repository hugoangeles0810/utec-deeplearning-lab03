from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ReIDMetrics:
    """mAP and CMC curve; ``cmc[k - 1]`` is the rank-k accuracy."""

    mean_ap: float
    cmc: np.ndarray

    def rank(self, k: int) -> float: ...


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
