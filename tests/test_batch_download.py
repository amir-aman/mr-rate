import json
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mrrate_vlm.download import download_batch, inspect_batch, parse_batches
from mrrate_vlm.utils import file_hash


def test_remote_inspection_lists_only_one_mri_batch(monkeypatch):
    import huggingface_hub as hub

    def file(path, size):
        return SimpleNamespace(path=path, size=size, lfs={"sha256": "a" * 64})

    calls = []

    def tree(path_in_repo, **kwargs):
        calls.append(path_in_repo)
        return {
            None: [
                SimpleNamespace(path="mri"),
                SimpleNamespace(path="metadata"),
                SimpleNamespace(path="reports"),
                file("splits.csv", 10),
            ],
            "mri/batch26": [file("mri/batch26/study.zip", 100)],
            "metadata": [
                file("metadata/batch00_metadata.csv", 20),
                file("metadata/batch26_metadata.csv", 20),
            ],
            "reports": [file("reports/batch26_reports.csv", 30)],
        }[path_in_repo]

    monkeypatch.setattr(
        hub,
        "HfApi",
        lambda: SimpleNamespace(
            repo_info=lambda **kwargs: SimpleNamespace(sha="commit"), list_repo_tree=tree
        ),
    )
    plan = inspect_batch("repo", "main", "batch26")
    assert plan["mri_files"] == 1
    assert plan["supporting_files"] == 4
    assert plan["total_bytes"] == 180
    assert calls == ["mri/batch26", None, "metadata", "reports"]
    assert len(parse_batches("all")) == 28


def fixture_plan(fixture_root):
    archive = fixture_root / "raw/mri/batch01/synthetic-study-000.zip"
    support = fixture_root / "raw/splits.csv"
    entries = [
        {
            "path": "splits.csv",
            "kind": "supporting",
            "size_bytes": support.stat().st_size,
            "sha256": file_hash(support),
        },
        {
            "path": "mri/batch01/synthetic-study-000.zip",
            "kind": "mri",
            "size_bytes": archive.stat().st_size,
            "sha256": file_hash(archive),
        },
    ]
    return {
        "repo_id": "test/repo",
        "requested_revision": "main",
        "resolved_revision": "sha",
        "batch": "batch01",
        "data_kind": "real",
        "files": entries,
        "mri_files": 1,
        "supporting_files": 1,
        "mri_bytes": archive.stat().st_size,
        "supporting_bytes": support.stat().st_size,
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
    }


def test_batch_download_verifies_and_persists_provenance(fixture_root, monkeypatch):
    import huggingface_hub as hub

    import mrrate_vlm.download as download

    plan = fixture_plan(fixture_root)
    monkeypatch.setattr(download, "inspect_batch", lambda *args: plan)
    output = fixture_root / "downloaded"

    def fetch(filename, local_dir, **kwargs):
        destination = output / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixture_root / "raw" / filename, destination)
        return str(destination)

    mock = Mock(side_effect=fetch)
    monkeypatch.setattr(hub, "hf_hub_download", mock)
    result = download_batch(output, "batch01", "test/repo", "main", 1, workers=1)
    assert result["status"] == "complete"
    assert mock.call_count == 2
    summary = json.loads((output / "batch01_download_summary.json").read_text())
    assert summary["completed_files"] == 2
    assert summary["completed_bytes"] == plan["total_bytes"]
    assert all(entry["status"] == "verified" for entry in summary["files"])
    stamp = output / "mri/batch01/synthetic-study-000.provenance.json"
    assert json.loads(stamp.read_text())["revision"] == "sha"


def test_batch_cap_and_inspection_never_download(fixture_root, monkeypatch):
    import huggingface_hub as hub

    import mrrate_vlm.download as download

    monkeypatch.setattr(download, "inspect_batch", lambda *args: fixture_plan(fixture_root))
    fetch = Mock(side_effect=AssertionError("No downloads allowed"))
    monkeypatch.setattr(hub, "hf_hub_download", fetch)
    download_batch(fixture_root / "out", "batch01", "test/repo", "main", 1, inspect_only=True)
    with pytest.raises(ValueError, match="exceeding cap"):
        download_batch(fixture_root / "out", "batch01", "test/repo", "main", 1e-12)
    fetch.assert_not_called()
    assert not (fixture_root / "out").exists()


def test_corrupt_download_is_not_marked_complete(fixture_root, monkeypatch):
    import huggingface_hub as hub

    import mrrate_vlm.download as download

    plan = fixture_plan(fixture_root)
    plan["files"][0]["sha256"] = "0" * 64
    monkeypatch.setattr(download, "inspect_batch", lambda *args: plan)
    output = fixture_root / "out"

    def fetch(filename, local_dir, **kwargs):
        target = output / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixture_root / "raw" / filename, target)
        return str(target)

    monkeypatch.setattr(hub, "hf_hub_download", fetch)
    with pytest.raises(ValueError, match="SHA-256"):
        download_batch(output, "batch01", "test/repo", "main", 1, workers=1)
    summary = json.loads((output / "batch01_download_summary.json").read_text())
    assert summary["status"] == "incomplete_resume_supported"
    assert summary["completed_files"] == 0
