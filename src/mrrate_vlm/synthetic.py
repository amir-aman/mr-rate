"""Offline phantoms and deliberate fake embeddings, never empirical model evidence."""

import re
import tempfile
import zipfile
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image

from .config import Config
from .dataset import build_manifest
from .embed import embed_manifest
from .evaluate import evaluate_bundle
from .imaging import axial_slices
from .retrieval import normalize, pool_slices
from .utils import write_json


def phantom(pattern: int) -> np.ndarray:
    x, y, z = np.indices((24, 26, 20), dtype=np.float32)
    region = ((x - 12) / 10) ** 2 + ((y - 13) / 11) ** 2 + ((z - 10) / 9) ** 2 < 1
    rng = np.random.default_rng(pattern)
    texture = rng.uniform(5, 150, (24, 26, 1))
    return np.where(region, texture + z, 0).astype(np.float32)


class SyntheticEncoder:
    provenance = {
        "encoder": "SYNTHETIC deterministic phantom features",
        "synthetic": True,
        "resolved_revision": "synthetic-v1",
        "max_tokens": None,
        "warning": "Fake encoder reconstructs known phantoms from text pattern IDs",
    }

    def __init__(self, config: Config):
        self.config = config

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        return normalize(
            np.stack(
                [np.asarray(image.resize((8, 8))).mean(axis=-1).ravel() / 255 for image in images]
            )
        )

    def encode_texts(self, texts: list[str]) -> tuple[np.ndarray, list[dict]]:
        vectors, audits = [], []
        for text in texts:
            match = re.search(r"SYNTHETIC phantom pattern (\d+)", text)
            if not match:
                raise ValueError("Synthetic encoder only accepts labelled phantom text")
            images, _ = axial_slices(
                phantom(int(match[1])), self.config.slice_fractions, self.config.percentiles
            )
            vectors.append(pool_slices(self.encode_images(images)))
            audits.append(
                {
                    "original_tokens": len(text.split()),
                    "used_tokens": len(text.split()),
                    "truncated": False,
                    "count_method": "SYNTHETIC whitespace (no tokenizer)",
                }
            )
        return normalize(vectors), audits


def create_fixture(root: Path, n: int = 12) -> None:
    if n < 2:
        raise ValueError("Synthetic demo needs at least two studies")
    raw = root / "raw"
    (raw / "metadata").mkdir(parents=True, exist_ok=True)
    (raw / "reports").mkdir(exist_ok=True)
    (raw / "mri" / "batch01").mkdir(parents=True, exist_ok=True)
    metadata, reports, splits = [], [], []
    for i in range(n):
        uid = f"synthetic-study-{i:03d}"
        patient = f"synthetic-patient-{i:03d}"
        for modality in ("FLAIR", "T1w"):
            metadata.append(
                {
                    "patient_uid": patient,
                    "study_uid": uid,
                    "series_id": f"{modality.lower()}-raw-axi",
                    "classified_modality": modality,
                    "is_derived": "False",
                    "acquisition_plane": "AXIAL",
                    "SeriesNumber": "1",
                    "Manufacturer": "SYNTHETIC scanner",
                }
            )
        reports.append(
            {
                "study_uid": uid,
                "technique": "SYNTHETIC brain MRI phantom",
                "impression": f"SYNTHETIC phantom pattern {i}; engineering fixture only.",
            }
        )
        # Intentionally different patient-first batch: verifies metadata source resolution.
        splits.append(
            {"study_uid": uid, "patient_uid": patient, "split": "validation", "batch_id": "batch00"}
        )
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "phantom.nii.gz"
            # Store LAS; canonicalization restores the same RAS phantom used by the fake text tower.
            affine = np.diag([-1.0, 1.0, 2.0, 1.0])
            affine[0, 3] = 23
            nib.save(nib.Nifti1Image(phantom(i)[::-1].copy(), affine), target)
            with zipfile.ZipFile(
                raw / "mri" / "batch01" / f"{uid}.zip", "w", zipfile.ZIP_DEFLATED
            ) as archive:
                archive.write(target, f"{uid}/img/{uid}_flair-raw-axi.nii.gz")
    pd.DataFrame(metadata).to_csv(raw / "metadata" / "batch01_metadata.csv", index=False)
    pd.DataFrame(reports).to_csv(raw / "reports" / "batch01_reports.csv", index=False)
    pd.DataFrame(splits).to_csv(raw / "splits.csv", index=False)
    write_json(
        raw / "download_provenance.json",
        {"data_kind": "synthetic", "repo_id": "SYNTHETIC", "resolved_revision": "synthetic-v1"},
    )


def run_demo(root: Path, n: int, config: Config) -> dict:
    # Demo always uses its own private data tree and validation split.
    config = replace(
        config,
        data_dir=str(root),
        results_dir=str(root / "results"),
        split="validation",
        source_batches=(),
        max_studies=n,
    )
    config.validate()
    create_fixture(root, n)
    manifest, _ = build_manifest(root / "raw", root / "manifest.csv", config)
    embedding_summary = embed_manifest(
        manifest, root / "raw", root / "embeddings", config, SyntheticEncoder(config)
    )
    stats = evaluate_bundle(manifest, root / "embeddings", root / "results", config)
    return {
        "label": "SYNTHETIC ENGINEERING DEMO - no real model/data performance",
        "n": stats["n_studies"],
        "embedding_summary": embedding_summary,
        "results": str(root / "results"),
        "metrics": stats["image_to_report"],
    }
