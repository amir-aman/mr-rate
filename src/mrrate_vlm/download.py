"""Fetch explicit individual files only, never a repository snapshot."""

import json
import math
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import image_repo_path, validate_manifest
from .utils import file_hash, write_json


class AccessError(RuntimeError):
    pass


def inspect_batch(repo_id: str, revision: str, batch: str) -> dict:
    """List one explicit MRI batch and every supporting folder without file downloads."""
    from huggingface_hub import HfApi

    image_repo_path(batch, "check")
    api = HfApi()
    resolved = hub_call(api.repo_info, repo_id=repo_id, repo_type="dataset", revision=revision).sha

    def listing(path: str | None, recursive: bool = False) -> list:
        return hub_call(
            lambda **kwargs: list(api.list_repo_tree(**kwargs)),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=path,
            recursive=recursive,
            revision=resolved,
        )

    images = [item for item in listing(f"mri/{batch}", True) if hasattr(item, "size")]
    supporting = []
    for item in listing(None):
        if hasattr(item, "size"):
            supporting.append(item)
        elif item.path != "mri":
            supporting.extend(f for f in listing(item.path, True) if hasattr(f, "size"))
    if not images:
        raise ValueError("Selected MRI batch contains no files")
    entries = []
    for kind, items in (("supporting", supporting), ("mri", images)):
        for item in sorted(items, key=lambda f: f.path):
            path = Path(item.path)
            if path.is_absolute() or ".." in path.parts or "\\" in item.path or ":" in item.path:
                raise ValueError("Unsafe remote dataset path")
            if kind == "mri" and (
                len(path.parts) != 3 or path.parts[:2] != ("mri", batch) or path.suffix != ".zip"
            ):
                raise ValueError("MRI batch contains an unexpected file path")
            if kind == "supporting" and path.parts[0] == "mri":
                raise ValueError("Supporting files must not include another MRI batch")
            if not isinstance(item.size, int) or item.size < 0:
                raise ValueError("Unknown remote file size; cannot enforce download cap")
            lfs = getattr(item, "lfs", None)
            sha256 = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            entries.append(
                {"path": item.path, "kind": kind, "size_bytes": item.size, "sha256": sha256}
            )
    return {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": resolved,
        "batch": batch,
        "data_kind": "real",
        "files": entries,
        "mri_files": len(images),
        "supporting_files": len(supporting),
        "mri_bytes": sum(f.size for f in images),
        "supporting_bytes": sum(f.size for f in supporting),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
    }


def download_batch(
    raw_dir: Path,
    batch: str,
    repo_id: str,
    revision: str,
    max_total_gb: float,
    workers: int = 4,
    inspect_only: bool = False,
) -> dict:
    """Explicit whole-batch opt-in, with all auxiliary files, size cap and resume state."""
    from huggingface_hub import hf_hub_download

    if not math.isfinite(max_total_gb) or max_total_gb <= 0:
        raise ValueError("max_total_gb must be positive and finite")
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    plan = inspect_batch(repo_id, revision, batch)
    brief = {key: value for key, value in plan.items() if key != "files"}
    print(json.dumps({"download_plan": brief}, indent=2), flush=True)
    if inspect_only:
        return brief
    if plan["total_bytes"] > max_total_gb * 1e9:
        raise ValueError(
            f"Selected batch plus supporting files total {plan['total_bytes'] / 1e9:.3f} "
            f"GB, exceeding cap {max_total_gb:.3f} GB. No files downloaded."
        )
    raw_dir.mkdir(parents=True, exist_ok=True)
    remaining = sum(
        entry["size_bytes"] for entry in plan["files"] if not (raw_dir / entry["path"]).exists()
    )
    if remaining > shutil.disk_usage(raw_dir).free:
        raise ValueError("Insufficient free disk space for selected downloads")
    provenance_file = raw_dir / "download_provenance.json"
    if provenance_file.exists():
        previous = json.loads(provenance_file.read_text(encoding="utf-8"))
        if (
            previous.get("repo_id") != repo_id
            or previous.get("resolved_revision") != plan["resolved_revision"]
        ):
            raise ValueError(
                "Data directory belongs to another dataset/revision; use a new data_dir"
            )
    summary_path = raw_dir / f"{batch}_download_summary.json"
    state = {**plan, "status": "downloading", "completed_files": 0, "completed_bytes": 0}
    write_json(summary_path, state)
    write_json(
        provenance_file,
        {
            **brief,
            "status": "downloading",
            "files": [entry["path"] for entry in plan["files"] if entry["kind"] == "supporting"],
        },
    )

    def fetch(entry: dict) -> dict:
        destination = raw_dir / entry["path"]
        if not destination.resolve().is_relative_to(raw_dir.resolve()):
            raise ValueError("Download destination escapes the data directory")
        fetched = Path(
            hub_call(
                hf_hub_download,
                repo_id=repo_id,
                repo_type="dataset",
                filename=entry["path"],
                revision=plan["resolved_revision"],
                local_dir=str(raw_dir),
            )
        )
        if fetched.stat().st_size != entry["size_bytes"]:
            raise ValueError("Downloaded dataset file has an unexpected size")
        if entry["sha256"] and file_hash(fetched) != entry["sha256"]:
            raise ValueError("Downloaded dataset file failed its SHA-256 check")
        if entry["kind"] == "mri":
            if not valid_zip(fetched):
                raise ValueError("Downloaded study ZIP failed CRC validation")
            write_json(
                fetched.with_suffix(".provenance.json"),
                {
                    "repo_id": repo_id,
                    "revision": plan["resolved_revision"],
                    "size_bytes": entry["size_bytes"],
                },
            )
        return {**entry, "status": "verified"}

    completed = {}
    # Supporting files first: metadata inspection can proceed before the MRI stage finishes.
    for kind in ("supporting", "mri"):
        entries = [entry for entry in plan["files"] if entry["kind"] == kind]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(fetch, entry) for entry in entries]
            try:
                for count, future in enumerate(as_completed(futures), 1):
                    entry = future.result()
                    completed[entry["path"]] = entry
                    state["completed_files"] = len(completed)
                    state["completed_bytes"] += entry["size_bytes"]
                    state["files"] = [completed.get(f["path"], f) for f in plan["files"]]
                    write_json(summary_path, state)
                    if count == 1 or count % 10 == 0 or count == len(entries):
                        print(
                            f"{kind}: {count}/{len(entries)} verified; "
                            f"total {state['completed_bytes'] / 1e9:.3f} GB",
                            flush=True,
                        )
            except Exception:
                for future in futures:
                    future.cancel()
                state["status"] = "incomplete_resume_supported"
                write_json(summary_path, state)
                raise
    state["status"] = "complete"
    write_json(summary_path, state)
    write_json(
        provenance_file,
        {
            **brief,
            "status": "complete",
            "files": [entry["path"] for entry in plan["files"] if entry["kind"] == "supporting"],
        },
    )
    return {**brief, "status": "complete", "local_dir": str(raw_dir)}


def hub_call(function: Any, **kwargs) -> Any:
    try:
        return function(**kwargs)
    except Exception as error:
        # Never echo remote request details, authorization headers or report identifiers.
        raise AccessError(
            f"Hugging Face request failed ({type(error).__name__}). Check network, repository "
            "revision and gated access. Accept the dataset/model terms with your own account "
            "and run `hf auth login` or set HF_TOKEN privately. No token is printed."
        ) from None


def parse_batches(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return [f"batch{number:02d}" for number in range(28)]
    batches = []
    for item in value.split(","):
        item = item.strip().removeprefix("batch")
        if not item.isdigit() or not 0 <= int(item) <= 27:
            raise ValueError("batches must be comma-separated values from 00 through 27")
        batches.append(f"batch{int(item):02d}")
    return sorted(set(batches))


def download_metadata(
    raw_dir: Path, batches: list[str], repo_id: str, revision: str, dry_run: bool = False
) -> dict:
    files = ["splits.csv"]
    for batch in batches:
        image_repo_path(batch, "check")
        files.extend([f"metadata/{batch}_metadata.csv", f"reports/{batch}_reports.csv"])
    plan = {
        "repo_id": repo_id,
        "requested_revision": revision,
        "files": files,
        "contains_mri": False,
        "dry_run": dry_run,
    }
    if dry_run:
        return plan
    from huggingface_hub import HfApi, hf_hub_download

    info = hub_call(HfApi().repo_info, repo_id=repo_id, repo_type="dataset", revision=revision)
    resolved = info.sha
    provenance_file = raw_dir / "download_provenance.json"
    if provenance_file.exists():
        existing = json.loads(provenance_file.read_text(encoding="utf-8"))
        if existing.get("resolved_revision") != resolved or existing.get("repo_id") != repo_id:
            raise ValueError(
                "Metadata directory contains another dataset revision; use a new data_dir"
            )
    for filename in files:
        hub_call(
            hf_hub_download,
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            revision=resolved,
            local_dir=str(raw_dir),
        )
    plan.update({"resolved_revision": resolved, "data_kind": "real"})
    write_json(provenance_file, plan)
    return plan


def valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.infolist()) and archive.testzip() is None
    except (OSError, zipfile.BadZipFile, RuntimeError, EOFError):
        return False


def image_plan(manifest: pd.DataFrame, max_studies: int, raw_dir: Path) -> list[dict]:
    validate_manifest(manifest)
    if max_studies < 1:
        raise ValueError("max_studies must be positive")
    return [
        {
            "image_repo_path": row.image_repo_path,
            "destination": str(raw_dir / row.image_repo_path),
            "cached": (raw_dir / row.image_repo_path).is_file(),
        }
        for row in manifest.head(max_studies).itertuples()
    ]


def download_images(
    manifest: pd.DataFrame,
    raw_dir: Path,
    max_studies: int,
    max_total_gb: float,
    repo_id: str,
    revision: str,
    dry_run: bool = False,
) -> dict:
    plan = image_plan(manifest, max_studies, raw_dir)
    result = {
        "dry_run": dry_run,
        "selected_studies": len(plan),
        "files": plan,
        "max_total_gb": max_total_gb,
        "size_unit": "decimal GB (10^9 bytes)",
    }
    if dry_run:
        result["note"] = "Offline plan; sizes unknown until authorized HEAD requests. No downloads."
        return result
    if not math.isfinite(max_total_gb) or max_total_gb <= 0:
        raise ValueError("max_total_gb must be positive and finite")
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url

    selected = manifest.head(max_studies)
    for col, expected in (("dataset_repo", repo_id), ("dataset_revision", revision)):
        if col in selected and set(selected[col]) != {expected}:
            raise ValueError(f"Manifest {col} differs from requested download source")
    info = hub_call(HfApi().repo_info, repo_id=repo_id, repo_type="dataset", revision=revision)
    resolved = info.sha
    # HEAD all selected files before pulling any bytes; unknown sizes fail closed.
    total = 0
    for entry in plan:
        url = hf_hub_url(
            repo_id, filename=entry["image_repo_path"], repo_type="dataset", revision=resolved
        )
        metadata = hub_call(get_hf_file_metadata, url=url)
        if metadata.size is None:
            raise ValueError("Server did not return a file size; cannot enforce max_total_gb")
        entry["size_bytes"] = metadata.size
        total += metadata.size
    if total > max_total_gb * 1_000_000_000:
        raise ValueError(
            f"Selected ZIPs total {total / 1e9:.3f} GB, exceeding cap {max_total_gb:.3f} GB. "
            "Reduce --max-studies or explicitly raise --max-total-gb. No images downloaded."
        )
    result.update({"total_bytes": total, "resolved_revision": resolved})
    for entry in plan:
        path = raw_dir / entry["image_repo_path"]
        stamp = path.with_suffix(".provenance.json")
        source = {"repo_id": repo_id, "revision": resolved, "size_bytes": entry["size_bytes"]}
        cached_source = json.loads(stamp.read_text()) if stamp.exists() else {}
        if cached_source == source and path.is_file() and valid_zip(path):
            entry["status"] = "validated_cache"
            continue
        # hf_hub_download's own content-addressed cache handles reuse/resume safely.
        fetched = Path(
            hub_call(
                hf_hub_download,
                repo_id=repo_id,
                repo_type="dataset",
                filename=entry["image_repo_path"],
                revision=resolved,
                local_dir=str(raw_dir),
            )
        )
        if fetched.stat().st_size != entry["size_bytes"] or not valid_zip(fetched):
            raise ValueError("Downloaded study ZIP is corrupt or has an unexpected size")
        write_json(stamp, source)
        entry["status"] = "downloaded"
    write_json(raw_dir / "image_download_summary.json", result)
    return result
