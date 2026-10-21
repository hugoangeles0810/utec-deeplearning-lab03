from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from lab03.evaluate import ReIDMetrics
from lab03.main import (
    build_parser,
    evaluate_split,
    format_metrics,
    get_device,
    run_evaluate,
    run_train,
)
from lab03.train import TrainConfig

# Same alias as in conftest.py; conftest is not importable with importlib mode.
type MakeH5 = Callable[..., Path]


class _PixelValue(nn.Module):
    """Maps each constant ``make_h5`` image to its normalized pixel values."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3))


def test_get_device_returns_torch_device():
    assert isinstance(get_device(), torch.device)


def test_train_defaults_follow_the_paper() -> None:
    args = build_parser().parse_args(["train"])

    assert args.run is run_train
    assert args.data_dir == Path("data")
    assert args.epochs == TrainConfig.epochs == 50
    assert args.iters == TrainConfig.iters_per_epoch
    assert args.eval_every == TrainConfig.eval_every


def test_evaluate_defaults_to_testing_split() -> None:
    args = build_parser().parse_args(["evaluate", "runs/best.pt"])

    assert args.run is run_evaluate
    assert args.checkpoint == Path("runs/best.pt")
    assert args.split == "testing"


def test_evaluate_rejects_training_split() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["evaluate", "x.pt", "--split", "training"])


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_format_metrics() -> None:
    metrics = ReIDMetrics(0.5, np.array([0.25, 0.5, 0.5, 0.5, 0.75] + [1] * 5))

    line = format_metrics("query", metrics)

    assert line == "query: mAP=50.0% R1=25.0% R5=75.0% R10=100.0%"


def test_evaluate_split_scores_both_query_resolutions(make_h5: MakeH5) -> None:
    path = make_h5({"query": 4, "gallery": 6}, person_ids=True, low_res_shape=(3, 2, 1))
    # make_h5 gives disjoint query and gallery ids; overwrite them to get matches.
    with h5py.File(path, "r+") as file:
        file["gallery/person_ids"][:] = [1, 2, 0, 1, 2, 3]

    results = evaluate_split(_PixelValue(), path, torch.device("cpu"), num_workers=0)

    assert set(results) == {"query", "low_res"}
    # Constant images keep their value after upsampling, so both rankings agree.
    np.testing.assert_allclose(results["low_res"].cmc, results["query"].cmc)
    assert results["low_res"].mean_ap == pytest.approx(results["query"].mean_ap)
