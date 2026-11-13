import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from mrrate_vlm.config import Config, load_config


@pytest.mark.parametrize(
    "changes",
    [
        {"seed": -1},
        {"max_total_gb": float("nan")},
        {"max_total_gb": float("inf")},
        {"max_studies": 0},
        {"split": "anything"},
        {"brain_only": "false"},
        {"n_slices": 4},
        {"percentiles": [99, 1]},
        {"batch_size": True},
    ],
)
def test_invalid_config_fails(changes):
    with pytest.raises(ValueError):
        replace(Config(), **changes).validate()


def test_example_config_loads():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config/default.yaml")
    assert config.max_studies == 30
    assert config.brain_only
    assert config.source_batches == ("batch26",)


def test_source_batch_config_is_validated():
    assert Config(source_batches=[26, "batch01"]).validate().source_batches == (
        "batch01",
        "batch26",
    )
    for batches in ("batch26", ["batch28"], ["../batch26"]):
        with pytest.raises(ValueError, match="source_batches"):
            Config(source_batches=batches).validate()


def test_gitignore_covers_private_artifacts(tmp_path):
    git = shutil.which("git")
    if not git:
        pytest.skip("Git is required for its own ignore-rule matcher")
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / ".gitignore", tmp_path / ".gitignore")
    subprocess.run([git, "init", "--quiet", str(tmp_path)], check=True, capture_output=True)
    sensitive = [
        "data/manifest.csv",
        "data/raw/reports/batch00_reports.csv",
        "data/qc/private.png",
        "data/raw/mri/batch01/study.zip",
        "data/embeddings/private.npz",
        "cache/model.bin",
        "results/retrieval_examples.csv",
        ".env",
        "private_token.txt",
        "volume.nii.gz",
        "checkpoint.safetensors",
        "report.csv",
        "MR_RATE_Codex_Handoff/01_CHAT_HISTORY_RECONSTRUCTED.md",
    ]
    result = subprocess.run(
        [git, "-C", str(tmp_path), "check-ignore", "--stdin", "-z"],
        input=("\0".join(sensitive) + "\0.env.example\0src/mrrate_vlm/cli.py\0").encode(),
        capture_output=True,
        check=True,
    )
    assert set(result.stdout.decode().strip("\0").split("\0")) == set(sensitive)
