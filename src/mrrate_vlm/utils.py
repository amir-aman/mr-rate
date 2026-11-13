"""Small deterministic, privacy-conscious I/O helpers."""

import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, ensure_ascii=True).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def private_id(value: str) -> str:
    return digest(value)[:16]


def safe_component(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(value)) or value in {".", ".."}:
        raise ValueError("Unsafe identifier/path component")
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def environment() -> dict:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in (
        "mrrate-vlm",
        "numpy",
        "pandas",
        "nibabel",
        "torch",
        "transformers",
        "huggingface-hub",
        "Pillow",
        "matplotlib",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def code_hash() -> str:
    return digest({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
