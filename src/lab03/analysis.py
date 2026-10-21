"""Reads the JSON written by ``lab03 train`` and ``lab03 evaluate``."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from lab03.train import TrainConfig

# TrainConfig fields exposed as CLI flags; the rest never change between runs.
TUNED = ("lr", "loss_weight", "k1", "k2", "eps", "min_samples")


@dataclass(frozen=True)
class Run:
    """One ``lab03 train --out`` directory."""

    name: str
    path: Path
    config: dict[str, Any]
    history: list[dict[str, Any]]
    reports: dict[str, dict[str, Any]]  # "<checkpoint>_<split>" -> evaluate JSON

    @property
    def label(self) -> str:
        return describe(self.config)

    @property
    def validated(self) -> list[dict[str, Any]]:
        """Epochs with validation metrics, in order."""
        return [r for r in self.history if r["validation"] is not None]

    @property
    def best(self) -> dict[str, Any] | None:
        """The epoch saved as ``best.pt``: first maximum of validation mAP."""
        validated = self.validated
        if not validated:
            return None
        return max(validated, key=lambda r: r["validation"]["query"]["mAP"])


def load_run(path: Path) -> Run:
    path = Path(path)
    reports = {}
    for report in sorted(path.glob("*_*.json")):
        reports[report.stem] = json.loads(report.read_text())
    return Run(
        name=path.name,
        path=path,
        config=json.loads((path / "config.json").read_text()),
        history=json.loads((path / "history.json").read_text()),
        reports=reports,
    )


def load_runs(root: Path) -> list[Run]:
    """Every run directly under ``root`` (or ``root`` itself), sorted by name."""
    root = Path(root)
    candidates = [root, *sorted(p for p in root.iterdir() if p.is_dir())]
    return [load_run(p) for p in candidates if (p / "history.json").is_file()]


def describe(config: dict[str, Any]) -> str:
    """What sets a run apart from the paper's setup, e.g. ``"sin CAPF, a=1"``."""
    model = config["model"]
    parts = []
    if not model["capf"]:
        parts.append("sin CAPF")
    if not model["spatial_attention"]:
        parts.append("sin atención espacial")
    if not model["se"]:
        parts.append("sin SE")
    defaults = {f.name: f.default for f in fields(TrainConfig)}
    for name in TUNED:
        value = config["train"][name]
        if value != defaults[name]:
            parts.append(f"{'a' if name == 'loss_weight' else name}={value:g}")
    return ", ".join(parts) or "AHCL (paper)"


def metric_row(summary: dict[str, Any]) -> dict[str, float]:
    """Flattens a ``summarize`` dict into one table row."""
    row = {}
    for condition, prefix in (("query", "orig"), ("low_res", "low")):
        for metric, value in summary[condition].items():
            row[f"{prefix} {metric}"] = value
    row["mAP drop"] = summary["map_drop"]["absolute"]
    row["drop %"] = summary["map_drop"]["relative"]
    return row


def validation_table(runs: Sequence[Run]) -> list[dict[str, Any]]:
    """Best validation epoch of each run: the basis for model selection."""
    rows = []
    for run in runs:
        best = run.best
        if best is None:
            continue
        rows.append(
            {"run": run.name, "variant": run.label, "epoch": best["epoch"]}
            | metric_row(best["validation"])
        )
    return rows


def report_table(
    runs: Sequence[Run], split: str = "testing", checkpoint: str = "best"
) -> list[dict[str, Any]]:
    """Rows from ``<checkpoint>_<split>.json``; runs without it are skipped."""
    rows = []
    for run in runs:
        report = run.reports.get(f"{checkpoint}_{split}")
        if report is not None:
            rows.append({"run": run.name, "variant": run.label} | metric_row(report))
    return rows


def markdown_table(rows: Sequence[dict[str, Any]]) -> str:
    """Floats in [0, 1] print as percentages; ``mAP drop`` in points."""
    if not rows:
        return "_(sin datos)_"
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        cells = [_format(column, row.get(column)) for column in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format(column: str, value: Any) -> str:
    if isinstance(value, float):
        if column == "mAP drop":
            return f"{value * 100:.1f} pts"
        return f"{value:.1%}"
    return "—" if value is None else str(value)
