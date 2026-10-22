import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from lab03.analysis import (
    describe,
    load_runs,
    markdown_table,
    metric_row,
    report_table,
    validation_table,
)
from lab03.train import TrainConfig

AHCL = {"capf": True, "spatial_attention": True, "se": True}


def _summary(orig: float, low: float) -> dict[str, Any]:
    def metrics(m: float) -> dict[str, float]:
        return {"mAP": m, "rank1": m + 0.1, "rank5": m + 0.2, "rank10": m + 0.3}

    drop = orig - low
    return {
        "query": metrics(orig),
        "low_res": metrics(low),
        "map_drop": {"absolute": drop, "relative": drop / orig},
    }


def _config(model: dict[str, bool] = AHCL, **train: Any) -> dict[str, Any]:
    return {"model": model, "train": asdict(TrainConfig(**train)), "seed": 0}


def _write_run(
    root: Path, name: str, val_maps: list[float | None], **train: Any
) -> Path:
    path = root / name
    path.mkdir()
    (path / "config.json").write_text(json.dumps(_config(**train)))
    history = [
        {
            "epoch": epoch,
            "lr": 3.5e-4,
            "num_clusters": 100,
            "num_outliers": 10,
            "losses": {"total": 1.0, "cluster": 1.0, "instance": 1.0},
            "validation": None if m is None else _summary(m, m / 2),
        }
        for epoch, m in enumerate(val_maps, start=1)
    ]
    (path / "history.json").write_text(json.dumps(history))
    return path


def test_describe_paper_setup() -> None:
    assert describe(_config()) == "AHCL (paper)"


def test_describe_lists_what_differs_from_the_paper() -> None:
    model = {"capf": False, "spatial_attention": True, "se": True}

    label = describe(_config(model, loss_weight=1.0, eps=0.5))

    assert label == "sin CAPF, a=1, eps=0.5"


@pytest.mark.parametrize(
    ("flag", "label"),
    [("spatial_attention", "sin atención espacial"), ("se", "sin SE")],
)
def test_describe_attention_ablations(flag: str, label: str) -> None:
    assert describe(_config(AHCL | {flag: False})) == label


def test_describe_shows_the_capf_fixes() -> None:
    model = AHCL | {"identity_attention": True, "balanced_fusion": True}

    label = describe(_config(model))

    assert label == "atención desde identidad, GAP/GMP normalizados"


def test_describe_shows_a_non_default_sam_kernel() -> None:
    label = describe(_config(AHCL | {"sam_kernel": 3}))

    assert label == "SAM 3x3"
    assert describe(_config(AHCL | {"sam_kernel": 7})) == "AHCL (paper)"


def test_describe_shows_a_non_native_image_size() -> None:
    config = _config(loss_weight=1.0) | {"image_size": [256, 128]}

    assert describe(config) == "a=1, 256x128"
    assert describe(config | {"image_size": [128, 64]}) == "a=1"


def test_load_runs_skips_directories_without_history(tmp_path: Path) -> None:
    _write_run(tmp_path, "b", [None, 0.4])
    _write_run(tmp_path, "a", [0.3])
    (tmp_path / "empty").mkdir()

    runs = load_runs(tmp_path)

    assert [run.name for run in runs] == ["a", "b"]


def test_load_runs_reads_evaluation_reports(tmp_path: Path) -> None:
    path = _write_run(tmp_path, "ahcl", [0.3])
    (path / "best_testing.json").write_text(json.dumps(_summary(0.5, 0.25)))

    (run,) = load_runs(tmp_path)

    assert set(run.reports) == {"best_testing"}


def test_best_is_the_first_epoch_with_max_validation_map(tmp_path: Path) -> None:
    _write_run(tmp_path, "run", [None, 0.3, 0.5, 0.2, 0.5])

    (run,) = load_runs(tmp_path)

    assert [r["epoch"] for r in run.validated] == [2, 3, 4, 5]
    assert run.best is not None
    assert run.best["epoch"] == 3


def test_best_is_none_without_validation(tmp_path: Path) -> None:
    _write_run(tmp_path, "run", [None, None])

    (run,) = load_runs(tmp_path)

    assert run.best is None
    assert validation_table([run]) == []


def test_metric_row_flattens_both_conditions_and_the_drop() -> None:
    row = metric_row(_summary(0.8, 0.6))

    assert row["orig mAP"] == 0.8
    assert row["low rank1"] == pytest.approx(0.7)
    assert row["mAP drop"] == pytest.approx(0.2)
    assert row["drop %"] == pytest.approx(0.25)


def test_validation_table_uses_the_best_epoch(tmp_path: Path) -> None:
    _write_run(tmp_path, "baseline", [0.2, 0.4], loss_weight=1.0)

    (row,) = validation_table(load_runs(tmp_path))

    assert row["run"] == "baseline"
    assert row["variant"] == "a=1"
    assert row["epoch"] == 2
    assert row["orig mAP"] == 0.4


def test_report_table_skips_runs_without_the_report(tmp_path: Path) -> None:
    path = _write_run(tmp_path, "a", [0.3])
    _write_run(tmp_path, "b", [0.3])
    (path / "best_testing.json").write_text(json.dumps(_summary(0.5, 0.25)))

    rows = report_table(load_runs(tmp_path))

    assert [row["run"] for row in rows] == ["a"]
    assert rows[0]["low mAP"] == 0.25


def test_markdown_table_formats_percentages_and_points() -> None:
    rows = [{"run": "a", "epoch": 5, "orig mAP": 0.5, "mAP drop": 0.123}]

    table = markdown_table(rows)

    assert table.splitlines() == [
        "| run | epoch | orig mAP | mAP drop |",
        "|---|---|---|---|",
        "| a | 5 | 50.0% | 12.3 pts |",
    ]


def test_markdown_table_without_rows() -> None:
    assert markdown_table([]) == "_(sin datos)_"
