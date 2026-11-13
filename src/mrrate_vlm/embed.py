"""Lazy model loading, independent tower inference and per-study provenance caches."""

import json
import zipfile
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from PIL import Image

from .config import Config
from .dataset import validate_manifest
from .download import hub_call
from .imaging import axial_slices, load_zip_volume, save_montage
from .reports import token_audit
from .retrieval import normalize, pool_slices
from .utils import code_hash, digest, environment, file_hash, private_id, write_json


class Encoder(Protocol):
    provenance: dict

    def encode_images(self, images: list[Image.Image]) -> np.ndarray: ...

    def encode_texts(self, texts: list[str]) -> tuple[np.ndarray, list[dict]]: ...


def feature_tensor(output: Any) -> Any:
    """Transformers 4 returns tensors; 5 may return BaseModelOutputWithPooling."""
    if hasattr(output, "detach"):
        result = output
    elif getattr(output, "pooler_output", None) is not None:
        result = output.pooler_output
    else:
        raise ValueError("Unsupported MedSigLIP feature output; expected tensor or pooler_output")
    if result.ndim != 2:
        raise ValueError("Model returned unexpected embedding shape")
    return result


class MedSiglipEncoder:
    def __init__(self, config: Config):
        try:
            import torch
            from huggingface_hub import HfApi
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "Install model dependencies: python -m pip install -e '.[model]'"
            ) from error

        self.torch = torch
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if config.device == "auto"
            else config.device
        )
        if self.device == "cpu" and config.dtype == "float16":
            raise ValueError("Use float32 on CPU; float16 CPU inference is not supported here")
        self.dtype = getattr(torch, config.dtype)
        resolved = hub_call(
            HfApi().repo_info,
            repo_id=config.model_id,
            repo_type="model",
            revision=config.model_revision,
        ).sha
        self.processor = hub_call(
            AutoProcessor.from_pretrained,
            pretrained_model_name_or_path=config.model_id,
            revision=resolved,
            cache_dir=str(Path(config.data_dir) / "model-cache"),
        )
        self.model = hub_call(
            AutoModel.from_pretrained,
            pretrained_model_name_or_path=config.model_id,
            revision=resolved,
            cache_dir=str(Path(config.data_dir) / "model-cache"),
        )
        self.model = self.model.to(device=self.device, dtype=self.dtype).eval()
        self.model.requires_grad_(False)
        self.batch_size = config.batch_size
        if not hasattr(self.model, "get_image_features") or not hasattr(
            self.model, "get_text_features"
        ):
            raise ValueError("Chosen model does not expose the supported SigLIP feature API")
        self.max_tokens = min(64, self.model.config.text_config.max_position_embeddings)
        self.provenance = {
            "encoder": "MedSigLIP",
            "synthetic": False,
            "model_id": config.model_id,
            "requested_revision": config.model_revision,
            "resolved_revision": resolved,
            "max_tokens": self.max_tokens,
            "device": self.device,
            "dtype": config.dtype,
            "resize": "official AutoProcessor PIL resizing",
        }

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        vectors = []
        for start in range(0, len(images), self.batch_size):
            inputs = self.processor(
                images=images[start : start + self.batch_size], return_tensors="pt"
            )
            pixels = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
            with self.torch.inference_mode():
                output = self.model.get_image_features(pixel_values=pixels)
            vectors.append(feature_tensor(output).detach().float().cpu().numpy())
        return normalize(np.concatenate(vectors))

    def encode_texts(self, texts: list[str]) -> tuple[np.ndarray, list[dict]]:
        audits = [token_audit(self.processor.tokenizer, text, self.max_tokens) for text in texts]
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            inputs = self.processor.tokenizer(
                texts[start : start + self.batch_size],
                padding="max_length",
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            )
            # SigLIP pooling uses the final token. Fixed-length padding follows the model card.
            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
                if k in {"input_ids", "attention_mask"}
            }
            with self.torch.inference_mode():
                output = self.model.get_text_features(**inputs)
            vectors.append(feature_tensor(output).detach().float().cpu().numpy())
        return normalize(np.concatenate(vectors)), audits


def row_fingerprint(row: dict) -> str:
    return digest({key: str(value) for key, value in row.items()})


def read_cached(cache_file: Path, audit_file: Path) -> tuple | None:
    if not cache_file.exists() or not audit_file.exists():
        return None
    try:
        with np.load(cache_file, allow_pickle=False) as data:
            image, text = data["image"], data["text"]
        if image.ndim != 1 or image.shape != text.shape:
            return None
        image, text = normalize(image[None, :])[0], normalize(text[None, :])[0]
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
        if not {"study_hash", "volume", "slices", "text"} <= set(audit):
            return None
        if not {"truncated", "original_tokens", "used_tokens"} <= set(audit["text"]):
            return None
        return image, text, audit
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
        # Interrupted or corrupted private cache entries are regenerated from the source.
        return None


def embed_manifest(
    manifest: pd.DataFrame,
    raw_dir: Path,
    output_dir: Path,
    config: Config,
    encoder: Encoder | None = None,
) -> dict:
    validate_manifest(manifest, config.source_batches)
    selected = manifest.loc[manifest.split.eq(config.split)].head(config.max_studies).copy()
    if selected.empty:
        raise ValueError("No manifest studies in the selected split")
    if encoder is None and "data_kind" in selected and selected.data_kind.eq("synthetic").any():
        raise ValueError("Synthetic inputs require the explicit synthetic demo encoder")
    missing = [r for r in selected.itertuples() if not (raw_dir / r.image_repo_path).is_file()]
    if missing:
        raise ValueError(f"{len(missing)} selected study ZIP(s) missing; run download-images first")
    encoder = encoder or MedSiglipEncoder(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "per-study"
    cache_dir.mkdir(exist_ok=True)
    run_provenance = {
        "encoder": encoder.provenance,
        "config": config.to_dict(),
        "code_hash": code_hash(),
        "environment": environment(),
        "split": config.split,
        "data_kind": "synthetic" if encoder.provenance.get("synthetic") else "real",
    }
    image_vectors, text_vectors, index, audits = [], [], [], []
    cached = 0
    for row in selected.to_dict("records"):
        identifier = private_id(row["study_uid"])
        archive = raw_dir / row["image_repo_path"]
        provenance = {
            "run": run_provenance,
            "input_sha256": file_hash(archive),
            "row_fingerprint": row_fingerprint(row),
        }
        key = digest(provenance)
        cache_file, audit_file = cache_dir / f"{key}.npz", cache_dir / f"{key}.json"
        existing_cache = read_cached(cache_file, audit_file)
        if existing_cache is not None:
            image, text, audit = existing_cache
            cached += 1
        else:
            volume, volume_qc = load_zip_volume(archive, row["study_uid"], row["series_id"])
            images, slice_qc = axial_slices(volume, config.slice_fractions, config.percentiles)
            label = (
                "SYNTHETIC - " if encoder.provenance.get("synthetic") else "Local research QC - "
            )
            save_montage(
                images, Path(config.data_dir) / "qc" / f"{identifier}.png", label + identifier
            )
            image = pool_slices(encoder.encode_images(images))
            vectors, text_audits = encoder.encode_texts([row["impression"]])
            text = normalize(vectors)[0]
            audit = {
                "study_hash": identifier,
                "data_kind": run_provenance["data_kind"],
                "volume": volume_qc,
                "slices": slice_qc,
                "text": text_audits[0],
            }
            temporary_cache = cache_file.with_suffix(".tmp.npz")
            np.savez_compressed(temporary_cache, image=image, text=text)
            temporary_cache.replace(cache_file)
            write_json(audit_file, audit)
        image_vectors.append(image)
        text_vectors.append(text)
        audits.append(audit)
        index.append({"study_uid": row["study_uid"], "row_fingerprint": row_fingerprint(row)})
    image_matrix, text_matrix = normalize(image_vectors), normalize(text_vectors)
    if image_matrix.shape != text_matrix.shape:
        raise ValueError("Image/text embedding shapes disagree")
    # A single atomic bundle prevents partial new runs being paired with an old index.
    temporary = output_dir / "embeddings.tmp.npz"
    np.savez_compressed(
        temporary,
        image=image_matrix,
        text=text_matrix,
        study_uids=np.asarray([r["study_uid"] for r in index]),
        row_fingerprints=np.asarray([r["row_fingerprint"] for r in index]),
        provenance=np.asarray(json.dumps(run_provenance)),
        audits=np.asarray(json.dumps(audits)),
    )
    temporary.replace(output_dir / "embeddings.npz")
    write_json(output_dir / "provenance.json", run_provenance)
    write_json(
        output_dir / "qc_summary.json",
        {
            "data_kind": run_provenance["data_kind"],
            "n": len(index),
            "cache_hits": cached,
            "truncated_reports": sum(a["text"]["truncated"] for a in audits),
            "studies": audits,
        },
    )
    return {
        "data_kind": run_provenance["data_kind"],
        "n": len(index),
        "cache_hits": cached,
        "truncated_reports": sum(a["text"]["truncated"] for a in audits),
        "embeddings": str(output_dir / "embeddings.npz"),
    }
