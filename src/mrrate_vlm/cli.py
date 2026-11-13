import argparse
import json
import sys
import zipfile
from pathlib import Path

from .config import load_config
from .dataset import build_manifest, load_manifest, validate_manifest
from .download import AccessError, download_batch, download_images, download_metadata, parse_batches


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Research-only 2.5D MRI-report retrieval")
    root.add_argument(
        "--config", type=Path, help="YAML config; paths relative to working directory"
    )
    commands = root.add_subparsers(dest="command", required=True)
    for name in (
        "download-batch",
        "download-metadata",
        "build-manifest",
        "download-images",
        "embed",
        "evaluate",
        "qc",
        "run",
    ):
        command = commands.add_parser(name)
        command.add_argument("--data-dir")
        if name in {"build-manifest", "download-images", "embed", "evaluate", "qc", "run"}:
            command.add_argument("--source-batches", help="Allowed MRI source batches; default 26")
        if name in {"build-manifest", "download-images", "embed", "evaluate", "qc", "run"}:
            command.add_argument("--manifest", type=Path)
            command.add_argument("--max-studies", type=int)
            command.add_argument("--split", choices=["train", "validation", "test"])
            command.add_argument("--seed", type=int)
        if name in {"download-metadata", "run"}:
            command.add_argument(
                "--batches", help="Metadata batches; 'all' downloads all batch CSVs"
            )
        if name == "download-batch":
            command.add_argument("--batch", default="26")
            command.add_argument("--workers", type=int, default=4)
            command.add_argument(
                "--inspect-remote",
                action="store_true",
                help="List remote sizes without downloading any files",
            )
        if name in {"download-metadata", "build-manifest"}:
            command.add_argument(
                "--metadata-only",
                action="store_true",
                help="These commands always operate on metadata only",
            )
        if name != "evaluate":
            command.add_argument(
                "--dry-run", action="store_true", help="No network or model access"
            )
        if name in {"build-manifest", "run"}:
            command.add_argument(
                "--include-unidentified-anatomy",
                action="store_true",
                default=None,
                help="Disable conservative brain screening; cannot claim brain-only",
            )
        if name in {"download-batch", "download-images", "run"}:
            command.add_argument("--max-total-gb", type=float)
        if name in {"embed", "run"}:
            command.add_argument("--model", dest="model_id")
            command.add_argument("--device")
            command.add_argument("--batch-size", type=int)
        if name in {"embed", "evaluate"}:
            command.add_argument("--embeddings-dir", type=Path)
        if name == "qc":
            command.add_argument("--qc-dir", type=Path)
        if name in {"evaluate", "run"}:
            command.add_argument("--results-dir")
    demo = commands.add_parser(
        "demo-synthetic", help="Completely offline fake-encoder engineering demo"
    )
    demo.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    demo.add_argument("--n-studies", type=int, default=12)
    demo.add_argument("--seed", type=int)
    return root


def source(manifest, config):
    repo, revision = config.repo_id, config.dataset_revision
    if "dataset_repo" in manifest:
        if manifest.dataset_repo.nunique() != 1 or manifest.dataset_revision.nunique() != 1:
            raise ValueError("Manifest must have one dataset source and revision")
        repo, revision = str(manifest.dataset_repo.iloc[0]), str(manifest.dataset_revision.iloc[0])
    if repo == "SYNTHETIC":
        raise ValueError(
            "Use demo-synthetic for offline fixtures; never fetch synthetic IDs from HF"
        )
    return repo, revision


def dispatch(args: argparse.Namespace) -> dict:
    overrides = {
        key: getattr(args, key, None)
        for key in (
            "data_dir",
            "max_studies",
            "max_total_gb",
            "split",
            "seed",
            "model_id",
            "device",
            "batch_size",
            "results_dir",
        )
    }
    if getattr(args, "include_unidentified_anatomy", None):
        overrides["brain_only"] = False
    if getattr(args, "source_batches", None):
        overrides["source_batches"] = parse_batches(args.source_batches)
    config = load_config(args.config, **overrides)
    data_dir = Path(config.data_dir)
    raw_dir = data_dir / "raw"
    dry_run = getattr(args, "dry_run", False)
    manifest_path = getattr(args, "manifest", None) or data_dir / "manifest.csv"
    embeddings_dir = getattr(args, "embeddings_dir", None) or data_dir / "embeddings"
    batches = getattr(args, "batches", None) or ",".join(config.source_batches) or "26"
    if args.command == "demo-synthetic":
        from .synthetic import run_demo

        return run_demo(args.output_dir, args.n_studies, config)
    if args.command == "download-metadata":
        return download_metadata(
            raw_dir, parse_batches(batches), config.repo_id, config.dataset_revision, dry_run
        )
    if args.command == "download-batch":
        selected_batches = parse_batches(args.batch)
        if len(selected_batches) != 1:
            raise ValueError("download-batch requires exactly one explicit batch")
        batch = selected_batches[0]
        if dry_run:
            return {
                "dry_run": True,
                "mri_pattern": f"mri/{batch}/*.zip",
                "supporting": "All non-MRI folders and root information files",
                "max_total_gb": config.max_total_gb,
                "note": "Offline plan; --inspect-remote checks sizes without file downloads",
            }
        return download_batch(
            raw_dir,
            batch,
            config.repo_id,
            config.dataset_revision,
            config.max_total_gb,
            args.workers,
            args.inspect_remote,
        )
    if args.command == "build-manifest":
        _, summary = build_manifest(raw_dir, manifest_path, config)
        return {
            "manifest": str(manifest_path),
            "selected_studies": summary["selected_studies"],
            "split": config.split,
            "network_access": False,
            "summary": str(manifest_path.parent / "selection_summary.json"),
        }
    if args.command == "run":
        if dry_run:
            return {
                "dry_run": True,
                "metadata": download_metadata(
                    raw_dir,
                    parse_batches(batches),
                    config.repo_id,
                    config.dataset_revision,
                    True,
                ),
                "selection": {
                    "split": config.split,
                    "max_studies": config.max_studies,
                    "brain_only": config.brain_only,
                    "source_batches": list(config.source_batches),
                },
                "max_total_gb": config.max_total_gb,
                "model": config.model_id,
                "note": "No network, model or MRI access",
            }
        download_metadata(raw_dir, parse_batches(batches), config.repo_id, config.dataset_revision)
        manifest, _ = build_manifest(raw_dir, manifest_path, config)
        repo, revision = source(manifest, config)
        download_images(manifest, raw_dir, config.max_studies, config.max_total_gb, repo, revision)
        from .embed import embed_manifest
        from .evaluate import evaluate_bundle

        embedding = embed_manifest(manifest, raw_dir, embeddings_dir, config)
        stats = evaluate_bundle(manifest, embeddings_dir, Path(config.results_dir), config)
        return {
            "embedding": embedding,
            "metrics": stats["image_to_report"],
            "results": config.results_dir,
        }
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest, config.source_batches)
    selected = manifest.loc[manifest.split.eq(config.split)].head(config.max_studies).copy()
    if selected.empty:
        raise ValueError("Manifest has no studies in requested split")
    if args.command == "download-images":
        repo, revision = (
            (config.repo_id, config.dataset_revision) if dry_run else source(selected, config)
        )
        return download_images(
            selected, raw_dir, config.max_studies, config.max_total_gb, repo, revision, dry_run
        )
    if args.command == "qc":
        qc_dir = (
            args.qc_dir
            or data_dir / "qc" / f"{'-'.join(selected.source_batch.unique())}-{config.split}"
        )
        if dry_run:
            return {
                "dry_run": True,
                "n": len(selected),
                "qc_dir": str(qc_dir),
                "model_loaded": False,
            }
        from .imaging import prepare_manifest_qc

        return prepare_manifest_qc(selected, raw_dir, qc_dir, config)
    if args.command == "embed":
        if dry_run:
            return {
                "dry_run": True,
                "n": len(selected),
                "model": config.model_id,
                "split": config.split,
                "note": "No model import or network access",
            }
        from .embed import embed_manifest

        return embed_manifest(selected, raw_dir, embeddings_dir, config)
    if args.command == "evaluate":
        from .evaluate import evaluate_bundle

        stats = evaluate_bundle(manifest, embeddings_dir, Path(config.results_dir), config)
        return {
            "label": stats["label"],
            "split": stats["split"],
            "metrics": stats["image_to_report"],
            "results": config.results_dir,
        }
    raise ValueError("Unknown command")


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        result = dispatch(arguments)
    except (ValueError, OSError, RuntimeError, AccessError, zipfile.BadZipFile) as error:
        # Application errors contain counts/policies, never raw report text or tokens.
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0
