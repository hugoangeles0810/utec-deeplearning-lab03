from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from lab03.data import ReIDEvalDataset, UnlabeledReIDDataset
from lab03.evaluate import (
    ReIDMetrics,
    compute_cmc_map,
    evaluate,
    extract_features,
    map_drop,
    pairwise_distance,
    score_queries,
    valid_ranking,
)

# Same alias as in conftest.py; conftest is not importable with importlib mode.
type MakeH5 = Callable[..., Path]

CPU = torch.device("cpu")
EVAL_SIZES = {"query": 4, "gallery": 6}


def _to_float(image: torch.Tensor) -> torch.Tensor:
    return image.float()


class _PixelValue(nn.Module):
    """Maps each constant ``make_h5`` image to its pixel value, per channel."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_training: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen_training.append(self.training)
        return x.mean(dim=(2, 3))


class _PixelAngle(nn.Module):
    """Maps pixel value v to a unit vector at angle 0.1 * v.

    Angles stay below pi, so cosine distance grows with |v - w|.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angle = 0.1 * x.mean(dim=(1, 2, 3))
        return torch.stack([angle.cos(), angle.sin()], dim=1)


class _Failing(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("forward failed")


def _ranked(
    g_pids: list[int],
    g_cams: list[int] | None = None,
    q_pid: int = 1,
    q_cam: int = 1,
    max_rank: int = 3,
) -> ReIDMetrics:
    """Scores one query against a gallery already sorted by distance."""
    g_cams = g_cams or [2] * len(g_pids)  # other camera: nothing is junk
    dist = np.arange(len(g_pids), dtype=np.float32)[None]
    return compute_cmc_map(
        dist,
        np.array([q_pid]),
        np.array(g_pids),
        np.array([q_cam]),
        np.array(g_cams),
        max_rank,
    )


def _assert_metrics(metrics: ReIDMetrics, mean_ap: float, cmc: list[float]) -> None:
    assert metrics.mean_ap == pytest.approx(mean_ap)
    np.testing.assert_allclose(metrics.cmc, cmc)


def test_cmc_map_is_perfect_when_matches_come_first() -> None:
    _assert_metrics(_ranked([1, 1, 2, 2]), 1.0, [1, 1, 1])


def test_cmc_map_averages_precision_at_each_match() -> None:
    # Matches at ranks 1 and 3: AP = (1/1 + 2/3) / 2.
    _assert_metrics(_ranked([1, 2, 1, 2]), 5 / 6, [1, 1, 1])


def test_cmc_map_counts_rank_of_first_match() -> None:
    _assert_metrics(_ranked([2, 1, 2, 2]), 1 / 2, [0, 1, 1])


def test_cmc_map_match_beyond_max_rank_still_counts_for_ap() -> None:
    _assert_metrics(_ranked([2, 2, 1], max_rank=2), 1 / 3, [0, 0])


def test_cmc_map_sorts_gallery_by_distance() -> None:
    dist = np.array([[0.9, 0.1]])

    metrics = compute_cmc_map(
        dist, np.array([1]), np.array([1, 2]), np.array([1]), np.array([2, 2]), 2
    )

    _assert_metrics(metrics, 1 / 2, [0, 1])


def test_cmc_map_ignores_same_camera_matches() -> None:
    with_junk = _ranked([1, 2, 1, 2], g_cams=[1, 2, 2, 2], q_cam=1)
    without_junk = _ranked([2, 1, 2])

    _assert_metrics(with_junk, without_junk.mean_ap, list(without_junk.cmc))
    _assert_metrics(with_junk, 1 / 2, [0, 1, 1])


def test_cmc_map_keeps_same_camera_negatives() -> None:
    # Only same pid + same camera is junk: a same-camera negative still ranks.
    _assert_metrics(_ranked([2, 1], g_cams=[1, 2], q_cam=1), 1 / 2, [0, 1, 1])


def test_cmc_map_ignores_junk_pid() -> None:
    _assert_metrics(_ranked([-1, 1, 2]), 1.0, [1, 1, 1])


def test_cmc_map_counts_distractors_as_negatives() -> None:
    _assert_metrics(_ranked([0, 1, 2]), 1 / 2, [0, 1, 1])


def test_cmc_map_averages_over_queries() -> None:
    dist = np.tile(np.arange(3, dtype=np.float32), (2, 1))
    g_pids = np.array([1, 2, 3])

    metrics = compute_cmc_map(
        dist, np.array([1, 2]), g_pids, np.array([1, 1]), np.array([2, 2, 2]), 3
    )

    # Query 1: AP 1 and rank-1 hit. Query 2: AP 1/2 and first hit at rank 2.
    _assert_metrics(metrics, 3 / 4, [1 / 2, 1, 1])


def test_cmc_map_skips_queries_without_valid_matches() -> None:
    dist = np.tile(np.arange(3, dtype=np.float32), (2, 1))
    g_pids, g_cams = np.array([3, 1, 2]), np.array([1, 2, 2])

    # Query 2 only matches gallery images from its own camera.
    metrics = compute_cmc_map(
        dist, np.array([1, 2]), g_pids, np.array([1, 2]), g_cams, 3
    )

    _assert_metrics(metrics, 1 / 2, [0, 1, 1])


def test_cmc_map_without_any_valid_match_raises() -> None:
    with pytest.raises(ValueError, match="no query has a valid gallery match"):
        _ranked([1, 2], g_cams=[1, 1], q_cam=1)


def test_cmc_map_rejects_transposed_distances() -> None:
    dist = np.zeros((3, 1))

    with pytest.raises(ValueError, match=r"expected \(1, 3\)"):
        compute_cmc_map(
            dist, np.array([1]), np.array([1, 2, 3]), np.array([1]), np.ones(3), 3
        )


def test_metrics_rank_reads_cmc() -> None:
    metrics = ReIDMetrics(mean_ap=0.5, cmc=np.array([0.25, 0.5, 1.0]))

    assert metrics.rank(1) == 0.25
    assert metrics.rank(3) == 1.0


def test_metrics_summary_reports_map_and_ranks() -> None:
    metrics = ReIDMetrics(mean_ap=0.5, cmc=np.array([0.25, 0.5, 0.5, 0.5, 0.75]))

    assert metrics.summary((1, 5)) == {"mAP": 0.5, "rank1": 0.25, "rank5": 0.75}


def test_map_drop_is_absolute_and_relative_to_the_original() -> None:
    original = ReIDMetrics(mean_ap=0.8, cmc=np.ones(10))
    degraded = ReIDMetrics(mean_ap=0.6, cmc=np.ones(10))

    drop = map_drop(original, degraded)

    assert drop.absolute == pytest.approx(0.2)
    assert drop.relative == pytest.approx(0.25)


def test_pairwise_distance_is_cosine_distance() -> None:
    x = torch.tensor([[1.0, 0.0]])
    gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    dist = pairwise_distance(x, gallery)

    torch.testing.assert_close(dist, torch.tensor([[0.0, 1.0, 2.0]]))


def test_pairwise_distance_ignores_feature_scale() -> None:
    torch.manual_seed(0)
    query, gallery = torch.randn(3, 8), torch.randn(5, 8)

    scaled = pairwise_distance(3 * query, 0.1 * gallery)

    assert scaled.shape == (3, 5)
    torch.testing.assert_close(scaled, pairwise_distance(query, gallery))


def test_pairwise_distance_is_never_negative() -> None:
    torch.manual_seed(0)
    x = torch.randn(512, 256)

    assert pairwise_distance(x, x).min() >= 0


def test_extract_features_returns_rows_in_dataset_order(make_h5: MakeH5) -> None:
    dataset = UnlabeledReIDDataset(make_h5(), transform=_to_float)
    loader = DataLoader(dataset, batch_size=3, shuffle=True)

    features = extract_features(_PixelValue(), loader, CPU)

    expected = torch.arange(10, dtype=torch.float32)[:, None].expand(10, 3)
    torch.testing.assert_close(features, expected)


def test_extract_features_infers_embedding_size(make_h5: MakeH5) -> None:
    dataset = UnlabeledReIDDataset(make_h5(), transform=_to_float)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 4, 5))

    features = extract_features(model, DataLoader(dataset, batch_size=4), CPU)

    assert features.shape == (10, 5)


def test_extract_features_returns_detached_float32_on_cpu(make_h5: MakeH5) -> None:
    dataset = UnlabeledReIDDataset(make_h5(), transform=lambda x: x.double())
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 4, 5, dtype=torch.float64))

    features = extract_features(model, DataLoader(dataset, batch_size=4), CPU)

    assert features.dtype == torch.float32
    assert features.device == CPU
    assert not features.requires_grad


@pytest.mark.parametrize("training", [True, False])
def test_extract_features_runs_in_eval_and_restores_mode(
    make_h5: MakeH5, training: bool
) -> None:
    dataset = UnlabeledReIDDataset(make_h5(), transform=_to_float)
    model = _PixelValue().train(training)

    extract_features(model, DataLoader(dataset, batch_size=4), CPU)

    assert model.seen_training == [False, False, False]
    assert model.training is training


def test_extract_features_restores_mode_when_model_fails(make_h5: MakeH5) -> None:
    dataset = UnlabeledReIDDataset(make_h5(), transform=_to_float)
    model = _Failing().train()

    with pytest.raises(RuntimeError, match="forward failed"):
        extract_features(model, DataLoader(dataset, batch_size=4), CPU)

    assert model.training


def test_extract_features_rejects_empty_loader(make_h5: MakeH5) -> None:
    dataset = UnlabeledReIDDataset(make_h5(), transform=_to_float)
    loader = DataLoader(torch.utils.data.Subset(dataset, []), batch_size=4)

    with pytest.raises(ValueError, match="loader yielded no batches"):
        extract_features(_PixelValue(), loader, CPU)


def _eval_loader(path: Path, key: str, person_ids: list[int]) -> DataLoader:
    dataset = ReIDEvalDataset(path, key, transform=_to_float)
    # make_h5 gives disjoint query and gallery ids; overwrite them to get matches.
    dataset.person_ids = np.array(person_ids)
    return DataLoader(dataset, batch_size=4)


def test_evaluate_applies_market_protocol_end_to_end(make_h5: MakeH5) -> None:
    path = make_h5(EVAL_SIZES, person_ids=True)
    # Pixel values: query 0..3 with cams 1..4, gallery 4..9 with cams 5, 6, 1..4.
    # _PixelAngle ranks the gallery 4..9 for every query.
    query = _eval_loader(path, "query/images", [0, 1, 2, 3])
    gallery = _eval_loader(path, "gallery/images", [1, 2, 0, 1, 2, 3])

    metrics = evaluate(_PixelAngle(), query, gallery, CPU, max_rank=3)

    # Queries 0 and 3 only match same-camera images and are skipped.
    # Query 1 hits at rank 1 (AP 1); query 2 hits at rank 2 (AP 1/2).
    _assert_metrics(metrics, 3 / 4, [1 / 2, 1, 1])


def test_evaluate_rejects_unlabeled_loader_before_extracting(
    make_h5: MakeH5,
) -> None:
    path = make_h5(EVAL_SIZES, person_ids=True)
    unlabeled = DataLoader(UnlabeledReIDDataset(path, "query/images"))
    gallery = DataLoader(ReIDEvalDataset(path, "gallery/images"))

    # _Failing would raise RuntimeError if features were extracted first.
    with pytest.raises(TypeError, match="expected a ReIDEvalDataset"):
        evaluate(_Failing(), unlabeled, gallery, CPU)


def test_score_queries_reports_ap_and_first_hit_per_query() -> None:
    dist = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.1, 0.2, 0.3]])
    q_pids, g_pids = np.array([1, 2, 9]), np.array([3, 1, 2])

    scores = score_queries(dist, q_pids, g_pids, np.zeros(3), np.ones(3))

    np.testing.assert_array_equal(scores.first_hit, [1, 0, -1])
    np.testing.assert_allclose(scores.ap[:2], [0.5, 1.0])
    assert np.isnan(scores.ap[2])
    np.testing.assert_array_equal(scores.valid, [True, True, False])


def test_valid_ranking_drops_junk_and_sorts_by_distance() -> None:
    row = np.array([0.4, 0.1, 0.3, 0.2])
    g_pids, g_cams = np.array([1, 1, -1, 2]), np.array([1, 2, 2, 1])

    ranked = valid_ranking(row, q_pid=1, q_cam=2, g_pids=g_pids, g_cams=g_cams)

    # Index 1 shares id and camera with the query, index 2 has id -1.
    np.testing.assert_array_equal(ranked, [3, 0])
