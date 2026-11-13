from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mrrate_vlm.dataset import build_manifest
from mrrate_vlm.download import (
    AccessError,
    download_images,
    download_metadata,
    hub_call,
    parse_batches,
)


def test_metadata_dry_run_never_connects(tmp_path):
    plan = download_metadata(tmp_path, parse_batches("01,00,batch01"), "repo", "sha", True)
    assert plan["files"] == [
        "splits.csv",
        "metadata/batch00_metadata.csv",
        "reports/batch00_reports.csv",
        "metadata/batch01_metadata.csv",
        "reports/batch01_reports.csv",
    ]
    assert not plan["contains_mri"]
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError):
        parse_batches("28")


def test_image_dry_run_caps_without_network(fixture_root, config):
    manifest, _ = build_manifest(fixture_root / "raw", fixture_root / "manifest.csv", config)
    plan = download_images(manifest, fixture_root / "raw", 2, 1, "unused", "unused", True)
    assert plan["selected_studies"] == 2
    assert all(e["image_repo_path"].startswith("mri/batch01/") for e in plan["files"])


@pytest.mark.parametrize("size", [None, 6_000_000_000])
def test_size_guard_before_any_download(fixture_root, config, monkeypatch, size):
    import huggingface_hub as hub

    manifest, _ = build_manifest(fixture_root / "raw", fixture_root / "manifest.csv", config)
    manifest["dataset_repo"] = "test/repo"
    manifest["dataset_revision"] = "sha123"
    monkeypatch.setattr(
        hub, "HfApi", lambda: SimpleNamespace(repo_info=lambda **k: SimpleNamespace(sha="sha123"))
    )
    monkeypatch.setattr(hub, "get_hf_file_metadata", lambda **k: SimpleNamespace(size=size))
    fetch = Mock(side_effect=AssertionError("No file may be pulled over budget"))
    monkeypatch.setattr(hub, "hf_hub_download", fetch)
    with pytest.raises(ValueError, match="size|exceeding cap"):
        download_images(manifest, fixture_root / "raw", 2, 1, "test/repo", "sha123")
    fetch.assert_not_called()


def test_download_only_selected_studies_then_cache(fixture_root, config, monkeypatch):
    import huggingface_hub as hub

    manifest, _ = build_manifest(fixture_root / "raw", fixture_root / "manifest.csv", config)
    manifest["dataset_repo"] = "test/repo"
    manifest["dataset_revision"] = "sha123"
    monkeypatch.setattr(
        hub, "HfApi", lambda: SimpleNamespace(repo_info=lambda **k: SimpleNamespace(sha="sha123"))
    )
    # All fixture ZIPs have different compressed sizes; use URL to identify exact file.
    monkeypatch.setattr(hub, "hf_hub_url", lambda repo, filename, **k: filename)
    monkeypatch.setattr(
        hub,
        "get_hf_file_metadata",
        lambda url: SimpleNamespace(size=(fixture_root / "raw" / url).stat().st_size),
    )
    fetch = Mock(side_effect=lambda filename, **kwargs: str(fixture_root / "raw" / filename))
    monkeypatch.setattr(hub, "hf_hub_download", fetch)
    first = download_images(manifest, fixture_root / "raw", 2, 1, "test/repo", "sha123")
    assert fetch.call_count == first["selected_studies"] == 2
    assert all(
        call.kwargs["filename"] in manifest.image_repo_path.head(2).tolist()
        for call in fetch.call_args_list
    )
    second = download_images(manifest, fixture_root / "raw", 2, 1, "test/repo", "sha123")
    assert fetch.call_count == 2
    assert all(e["status"] == "validated_cache" for e in second["files"])


def test_access_errors_redact_remote_details():
    def denied(**kwargs):
        raise RuntimeError("Authorization: Bearer private-test-secret")

    with pytest.raises(AccessError) as caught:
        hub_call(denied)
    assert "private-test-secret" not in str(caught.value)
    assert "hf auth login" in str(caught.value)
