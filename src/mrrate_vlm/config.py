import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    data_dir: str = "data"
    results_dir: str = "results"
    repo_id: str = "Forithmus/MR-RATE"
    dataset_revision: str = "main"
    source_batches: tuple = ("batch26",)
    model_id: str = "google/medsiglip-448"
    model_revision: str = "main"
    seed: int = 42
    split: str = "validation"
    max_studies: int = 30
    max_total_gb: float = 10.0
    brain_only: bool = True
    brain_columns: tuple = (
        "BodyPartExamined",
        "StudyDescription",
        "SeriesDescription",
        "body_part",
        "ProtocolName",
        "PerformedProcedureStepDescription",
        "ProcedureCodeSequence_0_CodeMeaning",
    )
    sequence_priority: tuple = ("FLAIR", "T2w", "T1w")
    min_text_chars: int = 10
    findings_fallback: bool = False
    n_slices: int = 3
    slice_fractions: tuple = (0.2, 0.5, 0.8)
    percentiles: tuple = (1.0, 99.0)
    batch_size: int = 4
    device: str = "auto"
    dtype: str = "float32"
    bootstrap_samples: int = 2000
    min_subgroup_n: int = 10
    near_duplicate_threshold: float = 0.9

    def validate(self) -> "Config":
        if not isinstance(self.source_batches, (tuple, list)):
            raise ValueError(
                "source_batches must be a list of batch names; [] disables the restriction"
            )
        canonical = []
        for batch in self.source_batches:
            value = str(batch).removeprefix("batch")
            if not value.isdigit() or not 0 <= int(value) <= 27:
                raise ValueError("source_batches values must be batch00 through batch27")
            canonical.append(f"batch{int(value):02d}")
        self.source_batches = tuple(sorted(set(canonical)))
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for key in ("brain_only", "findings_fallback"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a YAML boolean")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation or test")
        for key in (
            "max_studies",
            "min_text_chars",
            "n_slices",
            "batch_size",
            "bootstrap_samples",
            "min_subgroup_n",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if (
            not isinstance(self.max_total_gb, (int, float))
            or not math.isfinite(self.max_total_gb)
            or self.max_total_gb <= 0
        ):
            raise ValueError("max_total_gb must be positive and finite")
        if len(self.slice_fractions) != self.n_slices:
            raise ValueError("n_slices must equal the length of slice_fractions")
        if any(not 0 <= x <= 1 for x in self.slice_fractions):
            raise ValueError("slice_fractions must lie in [0, 1]")
        if len(self.percentiles) != 2 or not 0 <= self.percentiles[0] < self.percentiles[1] <= 100:
            raise ValueError("percentiles must be two increasing values in [0, 100]")
        if not 0 < self.near_duplicate_threshold <= 1:
            raise ValueError("near_duplicate_threshold must lie in (0, 1]")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16 or bfloat16")
        if not self.sequence_priority:
            raise ValueError("sequence_priority must not be empty")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Path | None = None, **overrides) -> Config:
    values = {}
    if path:
        values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a YAML mapping")
        unknown = set(values) - {f.name for f in fields(Config)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**values).validate()
