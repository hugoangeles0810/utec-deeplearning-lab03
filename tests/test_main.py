import json
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from lab03.evaluate import MapDrop, ReIDMetrics
from lab03.main import (
    build_parser,
    epoch_record,
    evaluate_split,
    format_drop,
    format_metrics,
    get_device,
    load_checkpoint,
    main,
    model_args,
    run_evaluate,
    run_train,
    save_checkpoint,
    split_datasets,
    summarize,
)
from lab03.model import ReIDModel
from lab03.train import EpochLosses, EpochStats, TrainConfig
from lab03.transforms import IMAGE_SIZE

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
    assert args.lr == TrainConfig.lr
    assert args.loss_weight == TrainConfig.loss_weight == 0.5
    assert (args.k1, args.k2) == (TrainConfig.k1, TrainConfig.k2)
    assert args.eps == TrainConfig.eps
    assert args.min_samples == TrainConfig.min_samples
    assert model_args(args) == {
        "capf": True,
        "spatial_attention": True,
        "se": True,
        "identity_attention": False,
        "balanced_fusion": False,
        "sam_kernel": 7,
    }


@pytest.mark.parametrize(
    ("flag", "disabled"),
    [
        ("--no-capf", "capf"),
        ("--no-spatial-attention", "spatial_attention"),
        ("--no-se", "se"),
    ],
)
def test_ablation_flags_disable_one_component(flag: str, disabled: str) -> None:
    default = model_args(build_parser().parse_args(["train"]))
    args = build_parser().parse_args(["train", flag])

    assert model_args(args) == default | {disabled: False}


@pytest.mark.parametrize(
    ("flag", "enabled"),
    [
        ("--identity-attention", "identity_attention"),
        ("--balanced-fusion", "balanced_fusion"),
    ],
)
def test_capf_fix_flags_enable_one_fix(flag: str, enabled: str) -> None:
    default = model_args(build_parser().parse_args(["train"]))
    args = build_parser().parse_args(["train", flag])

    assert model_args(args) == default | {enabled: True}


def test_sam_kernel_flag_sets_the_kernel_size() -> None:
    default = model_args(build_parser().parse_args(["train"]))
    args = build_parser().parse_args(["train", "--sam-kernel", "3"])

    assert model_args(args) == default | {"sam_kernel": 3}


def test_hyperparameter_flags_are_parsed() -> None:
    argv = ["train", "--lr", "1e-4", "--loss-weight", "1", "--eps", "0.5"]
    argv += ["--k1", "20", "--k2", "4", "--min-samples", "3"]

    args = build_parser().parse_args(argv)

    assert (args.lr, args.loss_weight, args.eps) == (1e-4, 1.0, 0.5)
    assert (args.k1, args.k2, args.min_samples) == (20, 4, 3)


@pytest.mark.parametrize(
    "flags",
    [
        ["--no-capf", "--no-se"],
        ["--no-capf", "--balanced-fusion"],
        ["--no-spatial-attention", "--identity-attention"],
        ["--no-spatial-attention", "--sam-kernel", "3"],
        ["--sam-kernel", "4"],
    ],
)
def test_incompatible_model_flags_are_rejected(flags: list[str]) -> None:
    with pytest.raises(SystemExit):
        main(["train", *flags])


def test_evaluate_defaults_to_testing_split() -> None:
    args = build_parser().parse_args(["evaluate", "runs/best.pt"])

    assert args.run is run_evaluate
    assert args.checkpoint == Path("runs/best.pt")
    assert args.split == "testing"


def test_evaluate_output_defaults_to_none() -> None:
    args = build_parser().parse_args(["evaluate", "runs/best.pt"])

    assert args.output is None


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


def test_format_drop() -> None:
    line = format_drop(MapDrop(absolute=0.123, relative=0.25))

    assert line == "mAP drop: 12.3 pts (25.0% relative)"


def _results() -> dict[str, ReIDMetrics]:
    return {
        "query": ReIDMetrics(0.8, np.linspace(0.5, 1, 10)),
        "low_res": ReIDMetrics(0.6, np.linspace(0.3, 1, 10)),
    }


def test_summarize_reports_both_conditions_and_the_drop() -> None:
    summary = summarize(_results())

    assert set(summary) == {"query", "low_res", "map_drop"}
    assert summary["query"]["mAP"] == 0.8
    assert summary["low_res"]["rank1"] == pytest.approx(0.3)
    assert summary["map_drop"] == {
        "absolute": pytest.approx(0.2),
        "relative": pytest.approx(0.25),
    }
    json.dumps(summary)  # serializable as is


@pytest.mark.parametrize("validated", [True, False])
def test_epoch_record(validated: bool) -> None:
    losses = EpochLosses(total=1.0, cluster=1.5, instance=0.5)
    stats = EpochStats(
        3, 1e-4, 10, 2, losses, _results()["query"] if validated else None
    )

    record = epoch_record(stats, _results() if validated else None)

    assert record["epoch"] == 3
    assert (record["num_clusters"], record["num_outliers"]) == (10, 2)
    assert record["losses"] == {"total": 1.0, "cluster": 1.5, "instance": 0.5}
    if validated:
        assert record["validation"] == summarize(_results())
    else:
        assert record["validation"] is None
    json.dumps(record)


def test_checkpoint_rebuilds_the_saved_variant(tmp_path: Path) -> None:
    margs = {"capf": False, "spatial_attention": True, "se": True}
    model = ReIDModel(**margs).eval()
    save_checkpoint(model, margs, tmp_path / "model.pt")

    loaded, loaded_args, image_size = load_checkpoint(tmp_path / "model.pt")

    assert loaded_args == margs
    assert image_size == IMAGE_SIZE
    assert not loaded.use_capf
    x = torch.randn(2, 3, 128, 64)
    with torch.inference_mode():
        torch.testing.assert_close(loaded.eval()(x), model(x))


def test_checkpoint_loads_a_bare_state_dict(tmp_path: Path) -> None:
    torch.save(ReIDModel().state_dict(), tmp_path / "old.pt")

    loaded, loaded_args, image_size = load_checkpoint(tmp_path / "old.pt")

    assert loaded_args == {}
    assert loaded.use_capf
    assert image_size == IMAGE_SIZE


def test_checkpoint_keeps_the_training_image_size(tmp_path: Path) -> None:
    save_checkpoint(ReIDModel(), {}, tmp_path / "model.pt", image_size=(256, 128))

    _, _, image_size = load_checkpoint(tmp_path / "model.pt")

    assert image_size == (256, 128)


def test_parser_reads_the_image_size() -> None:
    parser = build_parser()

    assert parser.parse_args(["train"]).image_size == list(IMAGE_SIZE)
    args = parser.parse_args(["train", "--image-size", "256", "128"])
    assert args.image_size == [256, 128]


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


def test_split_datasets_resize_every_image_to_the_model_size(make_h5: MakeH5) -> None:
    path = make_h5({"query": 2, "gallery": 3}, person_ids=True, low_res_shape=(3, 2, 1))

    queries, gallery = split_datasets(path, image_size=(256, 128))

    for dataset in [*queries.values(), gallery]:
        assert dataset[0][0].shape == (3, 256, 128)


def test_run_evaluate_writes_metrics_json(make_h5: MakeH5, tmp_path: Path) -> None:
    path = make_h5({"query": 4, "gallery": 6}, person_ids=True, low_res_shape=(3, 2, 1))
    with h5py.File(path, "r+") as file:
        file["gallery/person_ids"][:] = [1, 2, 0, 1, 2, 3]
    path.rename(tmp_path / "validation.h5")
    margs = {"capf": True, "spatial_attention": False, "se": True}
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(ReIDModel(**margs), margs, checkpoint)
    argv = ["evaluate", str(checkpoint), "--data-dir", str(tmp_path)]
    args = build_parser().parse_args(argv + ["--split", "validation", "--workers", "0"])

    run_evaluate(args)

    report = json.loads((tmp_path / "best_validation.json").read_text())
    assert report["split"] == "validation"
    assert report["model"] == margs
    assert set(report) >= {"query", "low_res", "map_drop"}
    assert set(report["query"]) == {"mAP", "rank1", "rank5", "rank10"}
