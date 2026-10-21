import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from lab03.data import ReIDEvalDataset, UnlabeledReIDDataset
from lab03.evaluate import ReIDMetrics, evaluate
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


def run_train(args: argparse.Namespace) -> None:
    device = get_device()
    torch.manual_seed(args.seed)
    config = TrainConfig(
        epochs=args.epochs,
        iters_per_epoch=args.iters,
        num_workers=args.workers,
        eval_every=args.eval_every,
    )
    train_path = args.data_dir / "training.h5"
    val_path = args.data_dir / "validation.h5"
    train_set = UnlabeledReIDDataset(train_path, transform=build_train_transform())
    cluster_set = UnlabeledReIDDataset(train_path, transform=build_eval_transform())
    model = ReIDModel(weights=ResNet50_Weights.IMAGENET1K_V1)
    args.out.mkdir(parents=True, exist_ok=True)
    best_map = -1.0
    last_metrics: dict[str, ReIDMetrics] = {}

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
        torch.save(model.state_dict(), args.out / "last.pt")
        if stats.metrics is None:
            return
        for name, metrics in last_metrics.items():
            print(f"  val {format_metrics(name, metrics)}", flush=True)
        if stats.metrics.mean_ap > best_map:
            best_map = stats.metrics.mean_ap
            torch.save(model.state_dict(), args.out / "best.pt")

    print(f"training on {device}, {len(train_set)} images", flush=True)
    train(model, train_set, cluster_set, device, config, validate, on_epoch)


def run_evaluate(args: argparse.Namespace) -> None:
    device = get_device()
    model = ReIDModel()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device)
    path = args.data_dir / f"{args.split}.h5"
    for name, metrics in evaluate_split(
        model, path, device, num_workers=args.workers
    ).items():
        print(f"{args.split} {format_metrics(name, metrics)}")


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
    train_cmd.set_defaults(run=run_train)

    eval_cmd = commands.add_parser("evaluate", help="score a checkpoint")
    eval_cmd.add_argument("checkpoint", type=Path)
    eval_cmd.add_argument("--data-dir", type=Path, default=Path("data"))
    eval_cmd.add_argument(
        "--split", choices=["validation", "testing"], default="testing"
    )
    eval_cmd.add_argument("--workers", type=int, default=TrainConfig.num_workers)
    eval_cmd.set_defaults(run=run_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
