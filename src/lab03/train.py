from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lab03.clustering import generate_pseudo_labels
from lab03.evaluate import ReIDMetrics, extract_features
from lab03.memory import HybridMemory
from lab03.sampler import PseudoLabelBatchSampler


@dataclass(frozen=True)
class TrainConfig:
    """Paper defaults (same as SPCL). A batch is ``P * Z`` = 16 x 16 images."""

    epochs: int = 50
    iters_per_epoch: int = 400
    clusters_per_batch: int = 16  # P
    instances_per_cluster: int = 16  # Z
    lr: float = 3.5e-4
    weight_decay: float = 5e-4
    lr_step: int = 20
    lr_gamma: float = 0.1
    temperature: float = 0.05
    momentum: float = 0.2
    loss_weight: float = 0.5  # a in eq. 3
    k1: int = 30
    k2: int = 6
    eps: float = 0.6
    min_samples: int = 4
    extract_batch_size: int = 256
    num_workers: int = 4
    eval_every: int = 5


@dataclass(frozen=True)
class EpochLosses:
    """Mean losses over an epoch's iterations."""

    total: float
    cluster: float
    instance: float


@dataclass(frozen=True)
class EpochStats:
    epoch: int
    lr: float
    num_clusters: int
    num_outliers: int
    losses: EpochLosses
    metrics: ReIDMetrics | None


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Adam:
    """Adam over trainable parameters only (skips the frozen BN bias)."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)


def train_one_epoch(
    model: nn.Module,
    memory: HybridMemory,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    iters: int,
) -> EpochLosses:
    """``iters`` steps, cycling ``loader``; updates both memories after each step."""
    model.train()
    batches = _cycle(loader)
    sums = torch.zeros(3, device=device)
    for _ in range(iters):
        images, indices, _ = next(batches)
        images = images.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)

        features = model(images)
        loss = memory(features, indices)
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        optimizer.step()
        memory.update(features, indices)

        sums += torch.stack([loss.total, loss.cluster, loss.instance]).detach()

    total, cluster, instance = (sums / iters).tolist()
    return EpochLosses(total, cluster, instance)


def train(
    model: nn.Module,
    train_set: Dataset,
    cluster_set: Dataset,
    device: torch.device,
    config: TrainConfig | None = None,
    validate: Callable[[nn.Module], ReIDMetrics] | None = None,
    on_epoch: Callable[[EpochStats], None] | None = None,
) -> list[EpochStats]:
    """The full loop of paper Fig. 1. Moves ``model`` to ``device``.

    ``train_set`` and ``cluster_set`` hold the same images in the same order:
    the first with training augmentations, the second deterministic. Each
    epoch re-extracts features from ``cluster_set``, re-clusters them and
    rebuilds the hybrid memory before training. ``validate`` runs every
    ``config.eval_every`` epochs and after the last one.
    """
    config = config or TrainConfig()
    model.to(device)
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_step, gamma=config.lr_gamma
    )
    pin_memory = device.type == "cuda"
    cluster_loader = DataLoader(
        cluster_set,
        batch_size=config.extract_batch_size,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    history = []
    for epoch in range(1, config.epochs + 1):
        features = extract_features(model, cluster_loader, device).to(device)
        pseudo = generate_pseudo_labels(
            features, config.k1, config.k2, config.eps, config.min_samples
        )
        memory = HybridMemory(
            features,
            pseudo.labels,
            config.temperature,
            config.momentum,
            config.loss_weight,
        )
        del features

        sampler = PseudoLabelBatchSampler(
            pseudo.labels, config.clusters_per_batch, config.instances_per_cluster
        )
        loader = DataLoader(
            train_set,
            batch_sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        )
        lr = scheduler.get_last_lr()[0]
        losses = train_one_epoch(
            model, memory, loader, optimizer, device, config.iters_per_epoch
        )
        scheduler.step()
        del memory

        evaluate_now = epoch % config.eval_every == 0 or epoch == config.epochs
        metrics = validate(model) if validate is not None and evaluate_now else None
        stats = EpochStats(
            epoch, lr, pseudo.num_clusters, pseudo.num_outliers, losses, metrics
        )
        history.append(stats)
        if on_epoch is not None:
            on_epoch(stats)
    return history


def _cycle(loader: DataLoader) -> Iterator:
    """Like ``itertools.cycle`` but re-iterates, so batches are resampled."""
    while True:
        yield from loader
