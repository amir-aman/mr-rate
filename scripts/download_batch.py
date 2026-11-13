"""Interactive, session-only authentication for an explicit capped batch download."""

import argparse
import getpass
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default="batch26")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-total-gb", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = getpass.getpass(
            "Hugging Face token (hidden; session only): "
        ).strip()
    from mrrate_vlm.download import download_batch

    try:
        result = download_batch(
            args.data_dir / "raw",
            args.batch,
            "Forithmus/MR-RATE",
            args.revision,
            args.max_total_gb,
            args.workers,
        )
    except Exception as error:
        print(f"Download stopped ({type(error).__name__}): {error}", flush=True)
        return 2
    print(
        f"Complete: {result['mri_files']} MRI ZIPs, {result['supporting_files']} supporting files",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
