import math
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lab03.evaluate import ReIDMetrics
from lab03.memory import HybridMemory
from lab03.sampler import PseudoLabelBatchSampler
from lab03.train import TrainConfig, build_optimizer, train, train_one_epoch

CPU = torch.device("cpu")
NUM_BLOBS, BLOB_SIZE = 4, 8
IMAGE_SHAPE = (3, 4, 2)
DIM = math.prod(IMAGE_SHAPE)
CONFIG = TrainConfig(
    epochs=2,
    iters_per_epoch=3,
    clusters_per_batch=2,
    instances_per_cluster=4,
    lr_step=1,
    k1=BLOB_SIZE - 1,
    k2=2,
    extract_batch_size=8,
    num_workers=0,
    eval_every=1,
)


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


class _Blobs(Dataset):
    """Images around ``NUM_BLOBS`` orthogonal axes, in blob order."""

    def __init__(self, noise: float = 0.05) -> None:
        generator = torch.Generator().manual_seed(0)
        centers = torch.eye(DIM)[:NUM_BLOBS].repeat_interleave(BLOB_SIZE, dim=0)
        images = centers + noise * torch.randn(centers.shape, generator=generator)
        self.images = images.reshape(-1, *IMAGE_SHAPE)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        return self.images[index], index, 0

    def __len__(self) -> int:
        return len(self.images)


class _Linear(nn.Module):
    """Flatten -> identity-initialized linear -> L2 norm, plus a frozen bias."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(DIM, DIM)
        self.frozen = nn.Parameter(torch.zeros(1), requires_grad=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(DIM))
            self.fc.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc(x.flatten(1)))


def _blob_labels() -> np.ndarray:
    return np.arange(NUM_BLOBS).repeat(BLOB_SIZE)


def _metrics(mean_ap: float = 0.5) -> ReIDMetrics:
    return ReIDMetrics(mean_ap, np.ones(10))


def _setup() -> tuple[_Linear, HybridMemory, DataLoader]:
    model, dataset = _Linear(), _Blobs()
    with torch.no_grad():
        features = model(dataset.images)
    memory = HybridMemory(features, _blob_labels())
    sampler = PseudoLabelBatchSampler(_blob_labels(), 2, 4)
    return model, memory, DataLoader(dataset, batch_sampler=sampler)


def test_build_optimizer_skips_frozen_parameters() -> None:
    model = _Linear()

    optimizer = build_optimizer(model, CONFIG)

    params = optimizer.param_groups[0]["params"]
    assert {id(p) for p in params} == {id(model.fc.weight), id(model.fc.bias)}
    assert optimizer.param_groups[0]["lr"] == CONFIG.lr
    assert optimizer.param_groups[0]["weight_decay"] == CONFIG.weight_decay


def test_train_one_epoch_cycles_loader_and_updates_model() -> None:
    model, memory, loader = _setup()
    before = model.fc.weight.clone()
    optimizer = build_optimizer(model, CONFIG)
    iters = 3 * len(loader)

    losses = train_one_epoch(model, memory, loader, optimizer, CPU, iters)

    assert all(math.isfinite(v) for v in (losses.total, losses.cluster))
    assert math.isfinite(losses.instance)
    assert losses.total == pytest.approx(0.5 * (losses.cluster + losses.instance))
    assert not torch.equal(model.fc.weight, before)
    assert model.training


def test_train_one_epoch_updates_memory_with_new_features() -> None:
    model, memory, loader = _setup()
    memory.instance.features.copy_(F.normalize(torch.randn(NUM_BLOBS * BLOB_SIZE, DIM)))
    centroids = memory.cluster.centroids.clone()
    instances = memory.instance.features.clone()
    optimizer = build_optimizer(model, CONFIG)

    train_one_epoch(model, memory, loader, optimizer, CPU, 1)

    changed = (memory.instance.features != instances).any(dim=1)
    assert changed.sum() == 2 * 4
    assert not torch.equal(memory.cluster.centroids, centroids)


def test_train_clusters_blobs_and_reports_each_epoch() -> None:
    seen = []

    history = train(_Linear(), _Blobs(), _Blobs(), CPU, CONFIG, on_epoch=seen.append)

    assert [s.epoch for s in history] == [1, 2]
    assert seen == history
    for stats in history:
        assert stats.num_clusters == NUM_BLOBS
        assert stats.num_outliers == 0
        assert math.isfinite(stats.losses.total)
        assert stats.metrics is None


def test_train_decays_learning_rate_every_lr_step() -> None:
    config = replace(CONFIG, epochs=3, lr_step=2)

    history = train(_Linear(), _Blobs(), _Blobs(), CPU, config)

    lrs = [s.lr for s in history]
    assert lrs == pytest.approx([config.lr, config.lr, config.lr * config.lr_gamma])


def test_train_validates_every_eval_every_epochs_and_at_the_end() -> None:
    config = replace(CONFIG, epochs=5, eval_every=2)
    calls = []

    def validate(model: nn.Module) -> ReIDMetrics:
        calls.append(model.training)
        return _metrics(len(calls))

    history = train(_Linear(), _Blobs(), _Blobs(), CPU, config, validate)

    evaluated = [s.epoch for s in history if s.metrics is not None]
    assert evaluated == [2, 4, 5]
    assert [history[e - 1].metrics.mean_ap for e in evaluated] == [1, 2, 3]


def test_train_fails_when_there_are_too_few_clusters() -> None:
    config = replace(CONFIG, clusters_per_batch=NUM_BLOBS + 1)

    with pytest.raises(ValueError, match="clusters per batch"):
        train(_Linear(), _Blobs(), _Blobs(), CPU, config)
