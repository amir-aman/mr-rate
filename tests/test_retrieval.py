import numpy as np
import pytest

from mrrate_vlm.evaluate import bootstrap_recall
from mrrate_vlm.reports import clean_text, duplicate_diagnostics, token_audit
from mrrate_vlm.retrieval import (
    chance_metrics,
    matched_ranks,
    metrics_from_ranks,
    normalize,
    pool_slices,
    shuffled_targets,
    similarities,
)


def test_exact_rankings_and_controls():
    image = np.eye(6)
    scores = similarities(image, image)
    ranks = matched_ranks(scores, np.arange(6))
    np.testing.assert_array_equal(ranks, np.ones(6))
    metrics = metrics_from_ranks(ranks, 6)
    assert metrics["recall_at_1"] == metrics["recall_at_5"] == metrics["mrr"] == 1
    control = shuffled_targets(6, 42)
    assert (control != np.arange(6)).all()
    assert len(set(control)) == 6
    assert metrics_from_ranks(matched_ranks(scores, control), 6)["recall_at_1"] == 0
    assert chance_metrics(6)["recall_at_1"] == 1 / 6


def test_nontrivial_rank_metrics_and_ties():
    scores = np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    ranks = matched_ranks(scores, np.arange(3))
    np.testing.assert_array_equal(ranks, [1, 2, 3])
    result = metrics_from_ranks(ranks, 3)
    assert result["recall_at_1"] == 1 / 3
    assert result["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)
    assert result["median_rank"] == 2
    assert result["recall_at_5"] is None
    np.testing.assert_array_equal(matched_ranks(np.ones((3, 3)), np.arange(3)), [2, 2, 2])


def test_pool_normalizes_each_slice_before_mean():
    vector = pool_slices(np.array([[100.0, 0.0], [0.0, 1.0]]))
    np.testing.assert_allclose(vector, np.array([1.0, 1.0]) / np.sqrt(2), rtol=1e-6)
    with pytest.raises(ValueError, match="zero"):
        normalize(np.zeros((2, 3)))
    ci = bootstrap_recall(np.array([1, 2, 1, 3]), ["a", "b", "c", "d"], 1000, 42)
    assert 0 <= ci["low"] <= 0.5 <= ci["high"] <= 1
    with pytest.raises(ValueError, match="unique patient"):
        bootstrap_recall(np.ones(2), ["a", "a"], 100, 42)


class Tokenizer:
    def __call__(self, text, truncation=False, max_length=None, **kwargs):
        # Deliberately two tokens per word, testing that audit uses tokenizer outputs.
        ids = [1] * (2 * len(text.split()) + 1)
        return {"input_ids": ids[:max_length] if truncation else ids}


def test_token_audit_empty_and_truncated():
    assert clean_text(None) == clean_text(float("nan")) == ""
    with pytest.raises(ValueError, match="empty"):
        token_audit(Tokenizer(), "  ")
    audit = token_audit(Tokenizer(), "word " * 40)
    assert audit == {"original_tokens": 81, "used_tokens": 64, "truncated": True, "max_tokens": 64}
    assert not token_audit(Tokenizer(), "short report")["truncated"]


def test_duplicate_diagnostics():
    result = duplicate_diagnostics(["Normal brain.", "normal  brain", "Normal brain MRI."])
    assert result["exact_duplicate_members"] == 2
    assert result["unique_normalized_impressions"] == 2
