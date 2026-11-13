from dataclasses import replace

import pandas as pd
import pytest

from mrrate_vlm.dataset import (
    brain_screen,
    build_manifest,
    image_repo_path,
    load_manifest,
    validate_manifest,
    validate_splits,
)


def test_join_source_batch_and_determinism(fixture_root, config):
    manifest, summary = build_manifest(fixture_root / "raw", fixture_root / "manifest.csv", config)
    assert len(manifest) == manifest.patient_uid.nunique() == 6
    assert manifest.source_batch.eq("batch01").all()
    assert manifest.image_repo_path.str.startswith("mri/batch01/").all()
    assert manifest.classified_modality.eq("FLAIR").all()
    assert summary["input_series"] == 12
    assert summary["selected_studies"] == 6
    second, _ = build_manifest(fixture_root / "raw", fixture_root / "manifest.csv", config)
    pd.testing.assert_frame_equal(manifest, second)
    pd.testing.assert_frame_equal(manifest, load_manifest(fixture_root / "manifest.csv"))


def test_one_study_per_patient(fixture_root, config):
    for relative in ("metadata/batch01_metadata.csv", "splits.csv"):
        path = fixture_root / "raw" / relative
        frame = pd.read_csv(path)
        frame.loc[frame.patient_uid.eq("synthetic-patient-001"), "patient_uid"] = (
            "synthetic-patient-000"
        )
        frame.to_csv(path, index=False)
    result, _ = build_manifest(fixture_root / "raw", fixture_root / "manifest.csv", config)
    assert len(result) == 5
    assert result.patient_uid.is_unique


def test_split_leakage():
    frame = pd.DataFrame(
        {"study_uid": ["s1", "s2"], "patient_uid": ["p1", "p1"], "split": ["train", "test"]}
    )
    with pytest.raises(ValueError, match="Patient leakage"):
        validate_splits(frame)


@pytest.mark.parametrize(
    "table,field",
    [
        ("reports/batch01_reports.csv", "impression"),
        ("metadata/batch01_metadata.csv", "classified_modality"),
        ("splits.csv", "split"),
    ],
)
def test_conflicting_duplicates_fail(fixture_root, config, table, field):
    path = fixture_root / "raw" / table
    frame = pd.read_csv(path)
    duplicate = frame.iloc[[0]].copy()
    duplicate[field] = "CONFLICT"
    pd.concat([frame, duplicate]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        build_manifest(fixture_root / "raw", fixture_root / "manifest.csv", config)


def test_identical_duplicates_do_not_multiply(fixture_root, config):
    path = fixture_root / "raw" / "reports/batch01_reports.csv"
    frame = pd.read_csv(path)
    pd.concat([frame, frame]).to_csv(path, index=False)
    result, _ = build_manifest(fixture_root / "raw", fixture_root / "manifest.csv", config)
    assert len(result) == 6


def test_brain_filter_is_not_sequence_filter():
    assert brain_screen(pd.Series({"classified_modality": "FLAIR"}), ["technique"]) == (
        False,
        "unidentified",
    )
    assert brain_screen(pd.Series({"technique": "Brain MRI"}), ["technique"])[0]
    assert not brain_screen(
        pd.Series({"technique": "Brain and cervical spine MRI"}), ["technique"]
    )[0]
    assert brain_screen(pd.Series({"StudyDescription": "MRG BEYİN KRANİAL"}), ["StudyDescription"])[
        0
    ]
    assert not brain_screen(
        pd.Series({"StudyDescription": "Beyin", "ProtocolName": "Torakal"}),
        ["StudyDescription", "ProtocolName"],
    )[0]


def test_missing_and_fallback_report(fixture_root, config):
    path = fixture_root / "raw" / "reports/batch01_reports.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "impression"] = ""
    frame["findings"] = "SYNTHETIC fallback sufficient text"
    frame.to_csv(path, index=False)
    manifest, _ = build_manifest(fixture_root / "raw", fixture_root / "manifest.csv", config)
    assert len(manifest) == 5
    fallback, _ = build_manifest(
        fixture_root / "raw", fixture_root / "manifest.csv", replace(config, findings_fallback=True)
    )
    assert len(fallback) == 6
    assert fallback.text_source.eq("findings_fallback").sum() == 1


def test_patient_mismatch_and_unsafe_path(fixture_root, config):
    with pytest.raises(ValueError, match="Unsafe"):
        image_repo_path("batch01", "../other")
    manifest, _ = build_manifest(fixture_root / "raw", fixture_root / "manifest.csv", config)
    manifest.loc[0, "image_repo_path"] = "mri/batch00/wrong.zip"
    with pytest.raises(ValueError, match="source_batch"):
        validate_manifest(manifest)


def test_no_brain_candidates_still_writes_audit(fixture_root, config):
    import json

    path = fixture_root / "raw/reports/batch01_reports.csv"
    frame = pd.read_csv(path)
    frame["technique"] = "Spinal MRI"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="No studies passed"):
        build_manifest(fixture_root / "raw", fixture_root / "manifest.csv", config)
    summary = json.loads((fixture_root / "selection_summary.json").read_text())
    assert summary["selected_studies"] == 0
    assert (fixture_root / "brain_screening_audit.csv").is_file()


def test_missing_split_and_patient_disagreement_fail(fixture_root, config):
    path = fixture_root / "raw/splits.csv"
    frame = pd.read_csv(path)
    original = frame.copy()
    frame.iloc[1:].to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing from official"):
        build_manifest(fixture_root / "raw", fixture_root / "manifest.csv", config)
    original.loc[0, "patient_uid"] = "different-patient"
    original.to_csv(path, index=False)
    with pytest.raises(ValueError, match="patient identifiers disagree"):
        build_manifest(fixture_root / "raw", fixture_root / "manifest.csv", config)


def test_source_batch_filter_happens_before_reading_other_metadata(fixture_root, config):
    import shutil

    raw = fixture_root / "raw"
    shutil.copyfile(raw / "metadata/batch01_metadata.csv", raw / "metadata/batch26_metadata.csv")
    shutil.copyfile(raw / "reports/batch01_reports.csv", raw / "reports/batch26_reports.csv")
    # An unrelated malformed/large batch must not be read or selected.
    (raw / "metadata/batch01_metadata.csv").write_text("unrelated_column\nignore_me\n")
    (raw / "reports/batch01_reports.csv").unlink()
    selected, summary = build_manifest(
        raw, fixture_root / "manifest.csv", replace(config, source_batches=("batch26",))
    )
    assert selected.source_batch.eq("batch26").all()
    assert selected.image_repo_path.str.startswith("mri/batch26/").all()
    assert summary["available_metadata_batches"] == ["batch01", "batch26"]
    assert summary["loaded_metadata_batches"] == ["batch26"]
    assert len(selected) == 6


def test_manifest_rejects_other_source_batches(fixture_root, config):
    selected, _ = build_manifest(fixture_root / "raw", fixture_root / "manifest.csv", config)
    with pytest.raises(ValueError, match="outside the configured"):
        validate_manifest(selected, ("batch26",))
    with pytest.raises(ValueError, match="no local metadata"):
        build_manifest(
            fixture_root / "raw",
            fixture_root / "other.csv",
            replace(config, source_batches=("batch26",)),
        )
