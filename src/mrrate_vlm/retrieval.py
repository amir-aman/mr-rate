import numpy as np


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Embeddings must be a finite 2D matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError("Cannot normalize a zero embedding")
    return (values / norms).astype(np.float32)


def pool_slices(embeddings: np.ndarray) -> np.ndarray:
    return normalize(normalize(embeddings).mean(axis=0, keepdims=True))[0]


def similarities(images: np.ndarray, reports: np.ndarray) -> np.ndarray:
    images, reports = normalize(images), normalize(reports)
    if images.shape[1] != reports.shape[1]:
        raise ValueError("Image/text embedding dimensions disagree")
    return images @ reports.T


def matched_ranks(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    targets = np.asarray(targets, dtype=int)
    if (
        scores.ndim != 2
        or len(targets) != len(scores)
        or not np.isfinite(scores).all()
        or (targets < 0).any()
        or (targets >= scores.shape[1]).any()
    ):
        raise ValueError("Invalid similarity matrix or matching indices")
    # Average ranks for tied scores: identical reports cannot receive arbitrary ID advantages.
    positives = scores[np.arange(len(scores)), targets, None]
    better = (scores > positives).sum(axis=1)
    equal = (scores == positives).sum(axis=1)
    return 1.0 + better + (equal - 1) / 2.0


def metrics_from_ranks(ranks: np.ndarray, n_candidates: int) -> dict:
    ranks = np.asarray(ranks, dtype=float)
    if not len(ranks) or not np.isfinite(ranks).all() or (ranks < 1).any():
        raise ValueError("Metrics require valid ranks")
    return {
        "n_queries": len(ranks),
        "n_candidates": n_candidates,
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)) if n_candidates >= 5 else None,
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
    }


def chance_metrics(n: int) -> dict:
    if n < 1:
        raise ValueError("Candidate count must be positive")
    return {
        "n_candidates": n,
        "recall_at_1": 1.0 / n,
        "recall_at_5": 5.0 / n if n >= 5 else None,
        "mrr": float(np.sum(1.0 / np.arange(1, n + 1)) / n),
        "median_rank": (n + 1) / 2,
        "note": "Expected uniform random ordering without ties",
    }


def shuffled_targets(n: int, seed: int) -> np.ndarray:
    if n < 2:
        raise ValueError("A shuffled control requires at least two studies")
    # Random cycle is a reproducible derangement: no accidentally correct pairs.
    order = np.random.default_rng(seed).permutation(n)
    targets = np.empty(n, dtype=int)
    targets[order] = np.roll(order, 1)
    return targets
