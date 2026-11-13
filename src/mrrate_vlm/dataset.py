"""Validated joins and deterministic study selection; all CSVs stay private."""

import json
import re
from pathlib import Path

import pandas as pd

from .config import Config
from .reports import clean_text
from .utils import digest, safe_component, write_json

BRAIN = re.compile(
    r"\b(brain|head|cranial|cranium|cerebral|intracranial|beyin|kranial|kraniyal)\b", re.I
)
SPINE = re.compile(
    r"\b(spine|spinal|cervical|thoracic|lumbar|lumbosacral|sacral|vertebra|vertebral|"
    r"servikal|torakal|lomber|sakral|lumbosakral)\b",
    re.I,
)
MANIFEST_COLUMNS = {
    "patient_uid",
    "study_uid",
    "source_batch",
    "series_id",
    "classified_modality",
    "image_repo_path",
    "impression",
    "split",
}


def read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    for column in required:
        frame[column] = frame[column].str.strip()
    return frame


def require_keys(frame: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
    frame = frame.drop_duplicates().copy()
    if frame[keys].eq("").any().any():
        raise ValueError(f"{name}: empty identifier")
    if frame.duplicated(keys).any():
        raise ValueError(f"{name}: conflicting duplicate keys {keys}")
    return frame


def validate_splits(splits: pd.DataFrame) -> pd.DataFrame:
    splits = require_keys(splits, ["study_uid"], "splits")
    splits["split"] = splits["split"].str.lower().replace({"val": "validation"})
    if not splits["split"].isin(["train", "validation", "test"]).all():
        raise ValueError("Unknown official split value")
    if splits["patient_uid"].eq("").any():
        raise ValueError("splits: empty patient_uid")
    if splits.groupby("patient_uid")["split"].nunique().gt(1).any():
        raise ValueError("Patient leakage: patient_uid appears in multiple official splits")
    return splits


def image_repo_path(batch: str, study: str) -> str:
    if not re.fullmatch(r"batch(?:0[0-9]|1[0-9]|2[0-7])", batch):
        raise ValueError("source_batch must be batch00 through batch27")
    return f"mri/{batch}/{safe_component(study)}.zip"


def validate_manifest(frame: pd.DataFrame, allowed_batches: tuple | None = None) -> pd.DataFrame:
    if not MANIFEST_COLUMNS <= set(frame.columns):
        raise ValueError(
            f"Manifest missing columns: {sorted(MANIFEST_COLUMNS - set(frame.columns))}"
        )
    if frame.empty:
        raise ValueError("Manifest has no selected studies; inspect selection_summary.json")
    if allowed_batches and not frame.source_batch.isin(allowed_batches).all():
        raise ValueError("Manifest contains studies outside the configured source_batches")
    if frame[list(MANIFEST_COLUMNS)].isna().any().any():
        raise ValueError("Manifest contains missing required values")
    if frame[list(MANIFEST_COLUMNS)].eq("").any().any():
        raise ValueError("Manifest contains empty required values")
    if frame.study_uid.duplicated().any() or frame.patient_uid.duplicated().any():
        raise ValueError("Manifest requires exactly one study per unique patient")
    validate_splits(frame[["study_uid", "patient_uid", "split"]].copy())
    for row in frame.itertuples():
        if row.image_repo_path != image_repo_path(row.source_batch, row.study_uid):
            raise ValueError("image_repo_path does not match metadata-derived source_batch")
        safe_component(row.series_id)
        if not clean_text(row.impression):
            raise ValueError("Manifest contains an empty report target")
    return frame


def load_manifest(path: Path) -> pd.DataFrame:
    frame = read_csv(path, MANIFEST_COLUMNS)
    if "brain_identifiable" in frame:
        values = frame["brain_identifiable"].str.lower()
        if not values.isin(["true", "false"]).all():
            raise ValueError("brain_identifiable must contain only true/false")
        frame["brain_identifiable"] = values.eq("true")
    return validate_manifest(frame)


def parse_false(value: str) -> bool:
    # Missing/unknown flag is not evidence of a nonderived series.
    return str(value).strip().lower() in {"false", "0", "no"}


def brain_screen(row: pd.Series, columns: list[str]) -> tuple[bool, str]:
    positive = []
    for col in columns:
        text = clean_text(row.get(col, ""))
        if SPINE.search(text):
            return False, f"spine_or_mixed:{col}"
        if BRAIN.search(text):
            positive.append(col)
    return bool(positive), "brain_keywords:" + ",".join(positive) if positive else "unidentified"


def build_manifest(raw_dir: Path, output: Path, config: Config) -> tuple[pd.DataFrame, dict]:
    paths = sorted((raw_dir / "metadata").glob("batch??_metadata.csv"))
    if not paths:
        raise ValueError("No metadata/batchXX_metadata.csv files. Run download-metadata first.")
    available_batches = [path.name.removesuffix("_metadata.csv") for path in paths]
    if config.source_batches:
        missing = set(config.source_batches) - set(available_batches)
        if missing:
            raise ValueError(f"Configured source batches have no local metadata: {sorted(missing)}")
        paths = [
            path
            for path in paths
            if path.name.removesuffix("_metadata.csv") in config.source_batches
        ]
    metadata_parts, report_parts = [], []
    for path in paths:
        batch = path.name.removesuffix("_metadata.csv")
        image_repo_path(batch, "validation")
        part = read_csv(
            path, {"patient_uid", "study_uid", "series_id", "classified_modality", "is_derived"}
        )
        part["source_batch"] = batch  # Never use splits.batch_id for image location.
        metadata_parts.append(part)
        report_path = raw_dir / "reports" / f"{batch}_reports.csv"
        report_parts.append(read_csv(report_path, {"study_uid", "impression"}))
    metadata = require_keys(
        pd.concat(metadata_parts, ignore_index=True).fillna(""),
        ["study_uid", "series_id"],
        "metadata",
    )
    if metadata.empty:
        raise ValueError("Metadata CSVs contain no series")
    if metadata.groupby("study_uid")[["source_batch", "patient_uid"]].nunique().gt(1).any().any():
        raise ValueError("Conflicting source batch or patient for a study")
    reports = require_keys(
        pd.concat(report_parts, ignore_index=True).fillna(""), ["study_uid"], "reports"
    )
    splits = validate_splits(
        read_csv(raw_dir / "splits.csv", {"patient_uid", "study_uid", "split"})
    )
    # Explicit columns prevent report/split field collisions and Cartesian joins.
    reports = reports[
        [c for c in ("study_uid", "impression", "findings", "technique") if c in reports]
    ].rename(columns={"technique": "report_technique"})
    joined = metadata.merge(
        splits[["study_uid", "patient_uid", "split"]],
        on="study_uid",
        how="left",
        validate="many_to_one",
        suffixes=("", "_split"),
    )
    if joined["split"].isna().any():
        raise ValueError("Metadata contains studies missing from official splits")
    if joined["patient_uid"].ne(joined["patient_uid_split"]).any():
        raise ValueError("Metadata/split patient identifiers disagree")
    joined = joined.drop(columns="patient_uid_split")
    joined = joined.merge(
        reports, on="study_uid", how="left", validate="many_to_one", suffixes=("_metadata", "")
    ).fillna("")
    anatomy_columns = [c for c in config.brain_columns if c in joined] + ["report_technique"]
    screening = joined.apply(lambda row: brain_screen(row, anatomy_columns), axis=1)
    joined["brain_identifiable"] = [result[0] for result in screening]
    joined["brain_screen_reason"] = [result[1] for result in screening]
    summary = {
        "seed": config.seed,
        "selection_policy": config.to_dict(),
        "columns_observed": sorted(metadata.columns),
        "anatomy_columns_examined": anatomy_columns,
        "brain_screening": (
            "Conservative English/Turkish anatomy keywords; spine veto; manual QA required"
        ),
        "brain_keyword_pattern": BRAIN.pattern,
        "spine_keyword_pattern": SPINE.pattern,
        "brain_filter_enabled": config.brain_only,
        "source_batch_policy": "metadata CSV filename",
        "available_metadata_batches": available_batches,
        "loaded_metadata_batches": [path.name.removesuffix("_metadata.csv") for path in paths],
        "allowed_source_batches": list(config.source_batches),
        "input_series": len(metadata),
        "input_reports": len(reports),
        "missing_reports_studies": int(joined.loc[joined.impression.eq(""), "study_uid"].nunique()),
        "metadata_missingness": {c: int(metadata[c].eq("").sum()) for c in metadata},
        "attrition_studies": {"all": int(joined.study_uid.nunique())},
    }
    # Local audit includes rejected candidates, but no report text; it stays under ignored data/.
    output.parent.mkdir(parents=True, exist_ok=True)
    joined[
        ["study_uid", "series_id", "source_batch", "brain_identifiable", "brain_screen_reason"]
    ].to_csv(output.parent / "brain_screening_audit.csv", index=False)
    selected = joined.loc[joined.brain_identifiable].copy() if config.brain_only else joined.copy()
    summary["attrition_studies"]["brain_identifiable_or_filter_disabled"] = int(
        selected.study_uid.nunique()
    )
    valid = selected.is_derived.map(parse_false)
    for flag in ("is_localizer", "is_subtraction"):
        if flag in selected:
            valid &= selected[flag].map(parse_false)
    priority = {s.casefold(): i for i, s in enumerate(config.sequence_priority)}
    selected["_priority"] = selected.classified_modality.str.casefold().map(priority)
    selected = selected.loc[valid & selected._priority.notna()].copy()
    summary["attrition_studies"]["valid_sequence"] = int(selected.study_uid.nunique())
    selected["impression"] = selected.impression.map(clean_text)
    selected["text_source"] = "impression"
    if config.findings_fallback and "findings" in selected:
        fallback = selected.impression.eq("")
        selected.loc[fallback, "impression"] = selected.loc[fallback, "findings"].map(clean_text)
        selected.loc[fallback, "text_source"] = "findings_fallback"
    selected = selected.loc[selected.impression.str.len() >= config.min_text_chars].copy()
    summary["attrition_studies"]["nonempty_sufficient_report"] = int(selected.study_uid.nunique())
    selected["_plane"] = selected.get("acquisition_plane", pd.Series("", index=selected.index)).map(
        lambda p: 0 if p.upper() == "AXIAL" else 1
    )
    selected["_number"] = pd.to_numeric(
        selected.get("SeriesNumber", pd.Series("", index=selected.index)), errors="coerce"
    ).fillna(float("inf"))
    selected = selected.sort_values(["study_uid", "_priority", "_plane", "_number", "series_id"])
    selected = selected.drop_duplicates("study_uid")
    selected["_sample"] = selected.study_uid.map(lambda uid: digest([config.seed, uid]))
    selected = selected.sort_values("_sample").drop_duplicates("patient_uid")
    summary["attrition_studies"]["one_per_patient_all_splits"] = len(selected)
    selected = selected.loc[selected.split.eq(config.split)].head(config.max_studies).copy()
    selected["image_repo_path"] = [
        image_repo_path(r.source_batch, r.study_uid) for r in selected.itertuples()
    ]
    provenance_path = raw_dir / "download_provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    selected["dataset_repo"] = provenance.get("repo_id", config.repo_id)
    selected["dataset_revision"] = provenance.get("resolved_revision", config.dataset_revision)
    selected["data_kind"] = provenance.get("data_kind", "real")
    keep = sorted(MANIFEST_COLUMNS) + [
        "brain_identifiable",
        "brain_screen_reason",
        "text_source",
        "dataset_repo",
        "dataset_revision",
        "data_kind",
    ]
    keep += [
        c
        for c in ("Manufacturer", "acquisition_plane", "array_shape", "array_spacing_mm")
        if c in selected
    ]
    selected = selected[keep].reset_index(drop=True)
    selected.to_csv(output, index=False)
    summary.update(
        {
            "selected_studies": len(selected),
            "unique_patients": selected.patient_uid.nunique(),
            "split": config.split,
            "dataset": provenance,
            "modalities": selected.classified_modality.value_counts().to_dict(),
            "screening_reason_counts": joined.brain_screen_reason.value_counts().to_dict(),
        }
    )
    summary["attrition_studies"]["final_selected"] = len(selected)
    write_json(output.parent / "selection_summary.json", summary)
    if selected.empty:
        raise ValueError(
            "No studies passed screening; inspect selection_summary.json and anatomy columns"
        )
    return validate_manifest(selected, config.source_batches), summary
