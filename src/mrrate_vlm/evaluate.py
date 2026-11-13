"""Patient-level retrieval evaluation and aggregate-only research outputs."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .dataset import validate_manifest
from .embed import row_fingerprint
from .reports import duplicate_diagnostics, duplicate_key
from .retrieval import (
    chance_metrics,
    matched_ranks,
    metrics_from_ranks,
    shuffled_targets,
    similarities,
)
from .utils import private_id, write_json

LIMITATIONS = """# Research limitations

{label}

- Retrieval is association matching, not diagnostic accuracy or Alzheimer's prediction.
- One study report is weakly paired with only three slices of one selected sequence.
- Mean-pooled 2D features form a 2.5D baseline, not a native 3D model.
- Native-space RAS reorientation does not register or de-oblique scans. Nonzero-voxel
  foreground is a heuristic; inspect local QC montages for coverage and artifacts.
- Brain keywords in descriptions/technique can misclassify anatomy. Mixed brain/spine
  descriptions are rejected. Manual examination of the local audit is required.
- The encoder receives at most 64 tokenizer tokens including special tokens. Truncation
  counters measure lost context; impressions may not mention every image finding.
- Exact or near-duplicate reports make strict one-pair retrieval ambiguous. Average ranks
  for exact score ties avoid an arbitrary identifier-based ranking advantage.
- The random baseline assumes uniformly random, untied ordering. The negative control
  uses one seeded random-cycle derangement; it is descriptive, not a significance test.
- Bootstrap intervals resample unique-patient query outcomes with the candidate pool
  fixed. They describe this convenience pilot and do not imply population validity.
- Subgroups use the full candidate pool and require the configured minimum query count.
  There are no subgroup significance claims or adjustment for scanner/site confounding.
- Duplicate-free sensitivity keeps one deterministic representative per normalized text,
  rebuilds the candidate pool and therefore changes both N and random chance.
- Official validation data are for pipeline decisions. Reserve official test data for
  the final held-out run; do not tune after inspecting test results.
- Reports were translated/structured using an LLM. Optional report-derived pathology
  labels are not independent clinical truth and are unused by this implementation.
- Images, reports, embeddings, IDs and QC outputs remain private. Hashing IDs does not
  constitute anonymization certification; review dataset terms before sharing aggregates.
"""


def bootstrap_recall(ranks: np.ndarray, patients: list[str], samples: int, seed: int) -> dict:
    if len(set(patients)) != len(patients) or len(patients) != len(ranks):
        raise ValueError("Bootstrap requires one query per unique patient")
    if len(ranks) < 2:
        return {"low": None, "high": None, "reason": "Need at least two patients"}
    rng = np.random.default_rng(seed)
    hits = np.asarray(ranks) <= 1
    means = np.empty(samples)
    for i in range(samples):
        means[i] = hits[rng.integers(0, len(hits), size=len(hits))].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "low": float(low),
        "high": float(high),
        "samples": samples,
        "method": "Percentile bootstrap of unique-patient queries; fixed candidate pool",
    }


def evaluate_arrays(
    manifest: pd.DataFrame, image: np.ndarray, text: np.ndarray, config: Config
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    validate_manifest(manifest, config.source_batches)
    if manifest.split.nunique() != 1:
        raise ValueError("Evaluation cannot mix official splits")
    if len(manifest) < 2:
        raise ValueError("Retrieval evaluation requires at least two unique patients")
    if len(image) != len(manifest) or len(text) != len(manifest):
        raise ValueError("Embeddings and manifest sample counts disagree")
    n = len(manifest)
    permutation = np.random.default_rng(config.seed).permutation(n)
    inverse = np.argsort(permutation)
    scores = similarities(image, text[permutation])
    ranks = matched_ranks(scores, inverse)
    negative_targets = shuffled_targets(n, config.seed + 1)
    control_ranks = matched_ranks(scores, inverse[negative_targets])
    stats = metrics_from_ranks(ranks, n)
    stats["recall_at_1_ci95"] = bootstrap_recall(
        ranks, manifest.patient_uid.tolist(), config.bootstrap_samples, config.seed
    )
    keys = [duplicate_key(t) for t in manifest.impression]
    keep = np.array([i for i, key in enumerate(keys) if key not in keys[:i]])
    sensitivity = {"n": len(keep), "note": "One representative per normalized impression; new pool"}
    if len(keep) >= 2:
        sensitivity["metrics"] = metrics_from_ranks(
            matched_ranks(similarities(image[keep], text[keep]), np.arange(len(keep))), len(keep)
        )
        sensitivity["chance"] = chance_metrics(len(keep))
    subgroups = {}
    for column in ("classified_modality", "Manufacturer"):
        if column not in manifest:
            subgroups[column] = {"skipped": "column unavailable"}
            continue
        groups = {}
        for value in sorted(manifest[column].unique()):
            mask = manifest[column].to_numpy() == value
            count = int(mask.sum())
            groups[value or "unknown"] = {"n": count}
            if count >= config.min_subgroup_n:
                groups[value or "unknown"]["metrics"] = metrics_from_ranks(ranks[mask], n)
            else:
                groups[value or "unknown"]["skipped"] = f"n < {config.min_subgroup_n}"
        subgroups[column] = groups
    stats = {
        "split": str(manifest.split.iloc[0]),
        "n_unique_patients": n,
        "n_studies": n,
        "n_candidates": n,
        "pilot_descriptive": True,
        "image_to_report": stats,
        "random_chance": chance_metrics(n),
        "shuffled_pair_control": metrics_from_ranks(control_ranks, n),
        "control_policy": "Seeded random-cycle derangement of ground truth pairs",
        "tie_policy": "Average ranks for exact tied scores; recall thresholds use these ranks",
        "queries_with_tied_positive": int(
            sum(np.sum(scores[i] == scores[i, inverse[i]]) > 1 for i in range(n))
        ),
        "report_duplicates": duplicate_diagnostics(
            manifest.impression.tolist(), config.near_duplicate_threshold
        ),
        "duplicate_free_sensitivity": sensitivity,
        "subgroups": subgroups,
    }
    return stats, ranks, scores, permutation


def write_figures(stats: dict, ranks: np.ndarray, directory: Path, label: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    names = ["recall_at_1", "recall_at_5", "mrr"]
    names = [key for key in names if stats["image_to_report"][key] is not None]
    x = np.arange(len(names))
    fig, axis = plt.subplots(figsize=(8, 4))
    for offset, (key, title) in enumerate(
        (
            ("image_to_report", "Retrieval"),
            ("random_chance", "Random expectation"),
            ("shuffled_pair_control", "Shuffled pairs"),
        )
    ):
        axis.bar(x + (offset - 1) * 0.25, [stats[key][k] for k in names], 0.25, label=title)
    axis.set(
        xticks=x,
        xticklabels=[n.replace("_", " ") for n in names],
        ylim=(0, 1.12),
        title=f"{label} | N={len(ranks)} | {stats['split']}",
        ylabel="Retrieval metric",
    )
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "retrieval_metrics.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.hist(ranks, bins=min(20, len(ranks)), edgecolor="black")
    axis.set(
        title=f"{label} | Associated-report rank | N={len(ranks)}",
        xlabel="Rank (average rank for ties)",
        ylabel="Number of studies",
    )
    fig.tight_layout()
    fig.savefig(directory / "rank_histogram.png", dpi=160)
    plt.close(fig)


def evaluate_bundle(
    manifest: pd.DataFrame, embeddings_dir: Path, output: Path, config: Config
) -> dict:
    validate_manifest(manifest, config.source_batches)
    with np.load(embeddings_dir / "embeddings.npz", allow_pickle=False) as bundle:
        uids = bundle["study_uids"].tolist()
        fingerprints = bundle["row_fingerprints"].tolist()
        if len(set(uids)) != len(uids):
            raise ValueError("Embedding index contains duplicate studies")
        indexed = manifest.set_index("study_uid", drop=False)
        if not set(uids) <= set(indexed.index):
            raise ValueError("Embedding studies do not match the provided manifest")
        selected = indexed.loc[uids].reset_index(drop=True)
        if selected.split.nunique() != 1 or set(selected.split) != {config.split}:
            raise ValueError("Embedding split differs from the requested evaluation split")
        actual = [row_fingerprint(row) for row in selected.to_dict("records")]
        if actual != fingerprints:
            raise ValueError("Manifest changed since embedding; regenerate embeddings")
        provenance = json.loads(str(bundle["provenance"].item()))
        audits = json.loads(str(bundle["audits"].item()))
        stats, ranks, scores, permutation = evaluate_arrays(
            selected, bundle["image"], bundle["text"], config
        )
    kind = provenance["data_kind"]
    if kind not in {"real", "synthetic"}:
        raise ValueError("Unknown data provenance")
    label = "SYNTHETIC ENGINEERING DEMO" if kind == "synthetic" else "MR-RATE research pilot"
    stats.update(
        {
            "data_kind": kind,
            "label": label,
            "provenance": provenance,
            "evaluation_config": config.to_dict(),
            "text_tokens": {
                "max_context": provenance["encoder"].get("max_tokens"),
                "truncated_reports": sum(a["text"]["truncated"] for a in audits),
                "original_tokens_total": sum(a["text"]["original_tokens"] for a in audits),
            },
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", stats)
    predicted = permutation[np.argmax(scores, axis=1)]
    with (output / "retrieval_examples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "data_kind",
                "query_hash",
                "matched_report_hash",
                "top_report_hash",
                "matched_rank",
                "top_score",
            ],
        )
        writer.writeheader()
        for i in np.argsort(ranks, kind="stable"):
            writer.writerow(
                {
                    "data_kind": kind,
                    "query_hash": private_id(uids[i]),
                    "matched_report_hash": private_id(uids[i]),
                    "top_report_hash": private_id(uids[predicted[i]]),
                    "matched_rank": float(ranks[i]),
                    "top_score": float(scores[i].max()),
                }
            )
    write_figures(stats, ranks, output / "figures", label)
    (output / "limitations.md").write_text(LIMITATIONS.format(label=label), encoding="utf-8")
    metric = stats["image_to_report"]
    (output / "result_table.md").write_text(
        f"{label}\n\n| Split | Patients / candidates | R@1 | R@5 | MRR | Median rank |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"| {stats['split']} | {len(uids)} | {metric['recall_at_1']:.4f} | "
        f"{metric['recall_at_5']} | {metric['mrr']:.4f} | {metric['median_rank']:.2f} |\n",
        encoding="utf-8",
    )
    return stats
