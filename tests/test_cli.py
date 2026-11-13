import json

from mrrate_vlm.cli import main
from mrrate_vlm.dataset import build_manifest


def test_metadata_and_run_dry_runs(capsys):
    assert main(["download-metadata", "--batches", "00", "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["dry_run"]
    assert main(["run", "--max-studies", "2", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"]


def test_embed_dry_run_does_not_load_model(fixture_root, config, monkeypatch, capsys):
    import mrrate_vlm.embed as embed

    def forbidden(*args, **kwargs):
        raise AssertionError("Model may not be loaded in dry-run")

    monkeypatch.setattr(embed, "MedSiglipEncoder", forbidden)
    manifest_path = fixture_root / "manifest.csv"
    build_manifest(fixture_root / "raw", manifest_path, config)
    assert (
        main(["embed", "--manifest", str(manifest_path), "--source-batches", "01", "--dry-run"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["n"] == 6


def test_cli_invalid_config_and_missing_manifest(tmp_path, capsys):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("unknown_key: 12")
    assert main(["--config", str(config_path), "run", "--dry-run"]) == 2
    assert "Unknown configuration" in capsys.readouterr().err
    assert main(["embed", "--manifest", str(tmp_path / "missing.csv"), "--dry-run"]) == 2
    assert "Error:" in capsys.readouterr().err


def test_download_dry_run_accepts_local_fixture(fixture_root, config, capsys):
    manifest_path = fixture_root / "manifest.csv"
    build_manifest(fixture_root / "raw", manifest_path, config)
    assert (
        main(
            [
                "download-images",
                "--manifest",
                str(manifest_path),
                "--source-batches",
                "01",
                "--max-studies",
                "2",
                "--dry-run",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["selected_studies"] == 2


def test_batch_download_dry_run_stays_offline(capsys):
    assert main(["download-batch", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mri_pattern"] == "mri/batch26/*.zip"
    assert plan["max_total_gb"] == 10


def test_qc_cli_runs_without_model(fixture_root, config, capsys):
    manifest_path = fixture_root / "manifest.csv"
    build_manifest(fixture_root / "raw", manifest_path, config)
    qc_dir = fixture_root / "qc-preview"
    assert (
        main(
            [
                "qc",
                "--data-dir",
                str(fixture_root),
                "--manifest",
                str(manifest_path),
                "--source-batches",
                "01",
                "--qc-dir",
                str(qc_dir),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["n"] == 6
    assert result["model_loaded"] is False
    assert len(list(qc_dir.glob("*.png"))) == 6
    assert (
        json.loads((qc_dir / "qc_summary.json").read_text())["studies"][0]["data_kind"]
        == "synthetic"
    )
