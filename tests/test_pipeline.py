import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mrrate_vlm.dataset import build_manifest, load_manifest
from mrrate_vlm.embed import embed_manifest
from mrrate_vlm.evaluate import evaluate_arrays, evaluate_bundle
from mrrate_vlm.synthetic import SyntheticEncoder, run_demo


def test_offline_end_to_end_cache_and_changed_inputs(fixture_root, config):
    manifest_path = fixture_root / "manifest.csv"
    build_manifest(fixture_root / "raw", manifest_path, config)
    manifest = load_manifest(manifest_path)
    encoder = SyntheticEncoder(config)
    first = embed_manifest(
        manifest, fixture_root / "raw", fixture_root / "embeddings", config, encoder
    )
    assert first["cache_hits"] == 0
    second = embed_manifest(
        manifest, fixture_root / "raw", fixture_root / "embeddings", config, encoder
    )
    assert second["cache_hits"] == 6
    cache = next((fixture_root / "embeddings/per-study").glob("*.npz"))
    cache.write_bytes(b"interrupted cache fixture")
    recovered = embed_manifest(
        manifest, fixture_root / "raw", fixture_root / "embeddings", config, encoder
    )
    assert recovered["cache_hits"] == 5
    result = evaluate_bundle(
        manifest, fixture_root / "embeddings", fixture_root / "results", config
    )
    assert result["data_kind"] == "synthetic"
    assert result["n_candidates"] == 6
    assert result["image_to_report"]["recall_at_1"] == 1
    assert result["shuffled_pair_control"]["recall_at_1"] == 0
    assert result["subgroups"]["classified_modality"]["FLAIR"]["skipped"]
    assert (fixture_root / "results/figures/retrieval_metrics.png").is_file()
    examples = (fixture_root / "results/retrieval_examples.csv").read_text()
    assert "synthetic-study-" not in examples
    assert "phantom pattern" not in examples
    assert "synthetic" in examples
    manifest.loc[0, "impression"] += " changed"
    with pytest.raises(ValueError, match="Manifest changed"):
        evaluate_bundle(manifest, fixture_root / "embeddings", fixture_root / "results", config)


def test_demo_outputs_labelled(tmp_path, config):
    result = run_demo(tmp_path / "demo", 6, config)
    assert "SYNTHETIC" in result["label"]
    metrics = json.loads((tmp_path / "demo/results/metrics.json").read_text())
    assert metrics["data_kind"] == "synthetic"
    assert "SYNTHETIC" in (tmp_path / "demo/results/limitations.md").read_text()


def test_evaluation_duplicate_sensitivity_and_subgroups(fixture_root, config):
    manifest, _ = build_manifest(fixture_root / "raw", fixture_root / "manifest.csv", config)
    manifest.loc[1, "impression"] = manifest.loc[0, "impression"]
    stats, _, _, _ = evaluate_arrays(
        manifest, np.eye(6), np.eye(6), replace(config, min_subgroup_n=3)
    )
    assert stats["duplicate_free_sensitivity"]["n"] == 5
    assert stats["report_duplicates"]["exact_duplicate_members"] == 2
    assert stats["subgroups"]["classified_modality"]["FLAIR"]["metrics"]["n_queries"] == 6
    manifest.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="mix official"):
        evaluate_arrays(manifest, np.eye(6), np.eye(6), config)


def test_no_model_load_for_synthetic_in_real_path(fixture_root, config):
    manifest, _ = build_manifest(fixture_root / "raw", fixture_root / "manifest.csv", config)
    with pytest.raises(ValueError, match="explicit synthetic"):
        embed_manifest(manifest, fixture_root / "raw", fixture_root / "embeddings", config)


def test_embedding_rejects_unavailable_batches_before_loading_model(fixture_root, config):
    manifest, _ = build_manifest(fixture_root / "raw", fixture_root / "manifest.csv", config)
    with pytest.raises(ValueError, match="outside the configured"):
        embed_manifest(
            manifest,
            fixture_root / "raw",
            fixture_root / "embeddings",
            replace(config, source_batches=("batch26",)),
        )


def test_embedding_alignment_survives_manifest_reordering(fixture_root, config):
    manifest, _ = build_manifest(fixture_root / "raw", fixture_root / "manifest.csv", config)
    embed_manifest(
        manifest,
        fixture_root / "raw",
        fixture_root / "embeddings",
        config,
        SyntheticEncoder(config),
    )
    reversed_manifest = pd.concat([manifest.iloc[3:], manifest.iloc[:3]], ignore_index=True)
    result = evaluate_bundle(
        reversed_manifest, fixture_root / "embeddings", fixture_root / "results", config
    )
    assert result["image_to_report"]["recall_at_1"] == 1
