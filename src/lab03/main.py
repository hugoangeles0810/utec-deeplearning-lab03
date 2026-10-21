import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from lab03.data import ReIDEvalDataset, UnlabeledReIDDataset
from lab03.evaluate import MapDrop, ReIDMetrics, evaluate, map_drop
from lab03.model import ReIDModel
from lab03.train import EpochStats, TrainConfig, train
from lab03.transforms import (
    build_eval_transform,
    build_low_res_transform,
    build_train_transform,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_split(
    model: nn.Module,
    path: Path,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 4,
) -> dict[str, ReIDMetrics]:
    """Scores the full-size and the low-resolution queries against the gallery."""
    queries = {
        "query": ReIDEvalDataset(path, "query/images", build_eval_transform()),
        "low_res": ReIDEvalDataset(
            path, "query/images_low_resolution", build_low_res_transform()
        ),
    }
    gallery = ReIDEvalDataset(path, "gallery/images", build_eval_transform())

    def loader(dataset: ReIDEvalDataset) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    return {
        name: evaluate(model, loader(query), loader(gallery), device)
        for name, query in queries.items()
    }


def format_metrics(name: str, metrics: ReIDMetrics) -> str:
    ranks = " ".join(f"R{k}={metrics.rank(k):.1%}" for k in (1, 5, 10))
    return f"{name}: mAP={metrics.mean_ap:.1%} {ranks}"


def format_drop(drop: MapDrop) -> str:
    return f"mAP drop: {drop.absolute * 100:.1f} pts ({drop.relative:.1%} relative)"


def summarize(results: dict[str, ReIDMetrics]) -> dict[str, Any]:
    """``evaluate_split`` output as JSON-friendly metrics plus the mAP drop."""
    summary: dict[str, Any] = {name: m.summary() for name, m in results.items()}
    summary["map_drop"] = asdict(map_drop(results["query"], results["low_res"]))
    return summary


def epoch_record(
    stats: EpochStats, validation: dict[str, ReIDMetrics] | None
) -> dict[str, Any]:
    """One ``history.json`` entry. ``validation`` is None on epochs without it."""
    return {
        "epoch": stats.epoch,
        "lr": stats.lr,
        "num_clusters": stats.num_clusters,
        "num_outliers": stats.num_outliers,
        "losses": asdict(stats.losses),
        "validation": None if validation is None else summarize(validation),
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def model_args(args: argparse.Namespace) -> dict[str, bool]:
    """``ReIDModel`` keyword arguments selected by the train flags."""
    return {
        "capf": args.capf,
        "spatial_attention": args.spatial_attention,
        "se": args.se,
    }


def save_checkpoint(model: nn.Module, args: dict[str, Any], path: Path) -> None:
    """Saves the weights with the ``ReIDModel`` arguments needed to rebuild it."""
    torch.save({"model_args": args, "state_dict": model.state_dict()}, path)


def load_checkpoint(path: Path) -> tuple[ReIDModel, dict[str, Any]]:
    """Rebuilds the model saved by ``save_checkpoint``, on CPU."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" not in checkpoint:  # bare state dict: default ReIDModel
        checkpoint = {"model_args": {}, "state_dict": checkpoint}
    model = ReIDModel(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint["model_args"]


def run_train(args: argparse.Namespace) -> None:
    device = get_device()
    torch.manual_seed(args.seed)
    config = TrainConfig(
        epochs=args.epochs,
        iters_per_epoch=args.iters,
        lr=args.lr,
        loss_weight=args.loss_weight,
        k1=args.k1,
        k2=args.k2,
        eps=args.eps,
        min_samples=args.min_samples,
        num_workers=args.workers,
        eval_every=args.eval_every,
    )
    train_path = args.data_dir / "training.h5"
    val_path = args.data_dir / "validation.h5"
    train_set = UnlabeledReIDDataset(train_path, transform=build_train_transform())
    cluster_set = UnlabeledReIDDataset(train_path, transform=build_eval_transform())
    margs = model_args(args)
    model = ReIDModel(weights=ResNet50_Weights.IMAGENET1K_V1, **margs)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out / "config.json",
        {"model": margs, "train": asdict(config), "seed": args.seed},
    )
    best_map = -1.0
    last_metrics: dict[str, ReIDMetrics] = {}
    history: list[dict[str, Any]] = []

    def validate(model: nn.Module) -> ReIDMetrics:
        nonlocal last_metrics
        last_metrics = evaluate_split(model, val_path, device, num_workers=args.workers)
        return last_metrics["query"]

    def on_epoch(stats: EpochStats) -> None:
        nonlocal best_map
        losses = stats.losses
        print(
            f"epoch {stats.epoch}/{config.epochs} lr={stats.lr:.2e} "
            f"clusters={stats.num_clusters} outliers={stats.num_outliers} "
            f"loss={losses.total:.3f} (cluster={losses.cluster:.3f} "
            f"instance={losses.instance:.3f})",
            flush=True,
        )
        save_checkpoint(model, margs, args.out / "last.pt")
        validation = None if stats.metrics is None else last_metrics
        history.append(epoch_record(stats, validation))
        write_json(args.out / "history.json", history)
        if validation is None:
            return
        for name, metrics in validation.items():
            print(f"  val {format_metrics(name, metrics)}", flush=True)
        drop = map_drop(validation["query"], validation["low_res"])
        print(f"  val {format_drop(drop)}", flush=True)
        if stats.metrics.mean_ap > best_map:
            best_map = stats.metrics.mean_ap
            save_checkpoint(model, margs, args.out / "best.pt")

    print(f"training on {device}, {len(train_set)} images", flush=True)
    train(model, train_set, cluster_set, device, config, validate, on_epoch)


def run_evaluate(args: argparse.Namespace) -> None:
    device = get_device()
    model, margs = load_checkpoint(args.checkpoint)
    model.to(device)
    path = args.data_dir / f"{args.split}.h5"
    results = evaluate_split(model, path, device, num_workers=args.workers)
    for name, metrics in results.items():
        print(f"{args.split} {format_metrics(name, metrics)}")
    print(f"{args.split} {format_drop(map_drop(results['query'], results['low_res']))}")

    output = args.output or args.checkpoint.with_name(
        f"{args.checkpoint.stem}_{args.split}.json"
    )
    write_json(
        output,
        {
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "model": margs,
            **summarize(results),
        },
    )
    print(f"wrote {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab03", description="AHCL unsupervised person re-identification."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_cmd = commands.add_parser("train", help="unsupervised training (AHCL)")
    train_cmd.add_argument("--data-dir", type=Path, default=Path("data"))
    train_cmd.add_argument("--out", type=Path, default=Path("runs"))
    train_cmd.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    train_cmd.add_argument("--iters", type=int, default=TrainConfig.iters_per_epoch)
    train_cmd.add_argument("--eval-every", type=int, default=TrainConfig.eval_every)
    train_cmd.add_argument("--workers", type=int, default=TrainConfig.num_workers)
    train_cmd.add_argument("--seed", type=int, default=0)

    hparams = train_cmd.add_argument_group("hyperparameters (tune on validation)")
    hparams.add_argument("--lr", type=float, default=TrainConfig.lr)
    hparams.add_argument(
        "--loss-weight",
        type=float,
        default=TrainConfig.loss_weight,
        help="a in eq. 3: 0 = cluster loss only, 1 = instance loss only",
    )
    hparams.add_argument("--k1", type=int, default=TrainConfig.k1)
    hparams.add_argument("--k2", type=int, default=TrainConfig.k2)
    hparams.add_argument("--eps", type=float, default=TrainConfig.eps)
    hparams.add_argument("--min-samples", type=int, default=TrainConfig.min_samples)

    ablation = train_cmd.add_argument_group("baseline and ablation")
    ablation.add_argument(
        "--no-capf",
        dest="capf",
        action="store_false",
        help="replace CAPF with global average pooling (paper baseline)",
    )
    ablation.add_argument(
        "--no-spatial-attention", dest="spatial_attention", action="store_false"
    )
    ablation.add_argument("--no-se", dest="se", action="store_false")
    train_cmd.set_defaults(run=run_train)

    eval_cmd = commands.add_parser("evaluate", help="score a checkpoint")
    eval_cmd.add_argument("checkpoint", type=Path)
    eval_cmd.add_argument("--data-dir", type=Path, default=Path("data"))
    eval_cmd.add_argument(
        "--split", choices=["validation", "testing"], default="testing"
    )
    eval_cmd.add_argument("--workers", type=int, default=TrainConfig.num_workers)
    eval_cmd.add_argument(
        "--output",
        type=Path,
        help="metrics JSON (default: <checkpoint>_<split>.json next to it)",
    )
    eval_cmd.set_defaults(run=run_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train" and not args.capf:
        if not (args.spatial_attention and args.se):
            parser.error("--no-spatial-attention and --no-se require CAPF")
    args.run(args)


if __name__ == "__main__":
    main()
