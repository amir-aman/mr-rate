"""Deterministic native-space NIfTI preprocessing, with bounded ZIP extraction."""

import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw

from .utils import safe_component


def load_volume(path: Path, max_voxels: int = 100_000_000) -> tuple[np.ndarray, dict]:
    try:
        image = nib.load(str(path))
    except nib.filebasedimages.ImageFileError:
        raise ValueError("Selected file is not a readable NIfTI image") from None
    if len(image.shape) != 3 or min(image.shape) < 2:
        raise ValueError("Expected a nondegenerate 3D NIfTI volume")
    if int(np.prod(image.shape, dtype=np.int64)) > max_voxels:
        raise ValueError("Volume exceeds voxel safety limit")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-10:
        raise ValueError("NIfTI affine must be finite and invertible")
    original_orientation = "".join(nib.aff2axcodes(image.affine))
    image = nib.as_closest_canonical(image)
    spacing = np.asarray(image.header.get_zooms()[:3])
    if not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Voxel spacing must be positive and finite")
    volume = image.get_fdata(dtype=np.float32)
    nonfinite = int((~np.isfinite(volume)).sum())
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    qc = {
        "shape": list(volume.shape),
        "spacing_mm": spacing.tolist(),
        "original_orientation": original_orientation,
        "canonical_orientation": "".join(nib.aff2axcodes(image.affine)),
        "nonfinite_replaced": nonfinite,
        "oblique_degrees": np.rad2deg(nib.affines.obliquity(image.affine)).tolist(),
        "orientation_note": "RAS reorientation only; oblique volumes are not resampled",
    }
    return volume, qc


def load_zip_volume(
    path: Path, study_uid: str, series_id: str, max_member_bytes: int = 1_000_000_000
) -> tuple[np.ndarray, dict]:
    study_uid, series_id = safe_component(study_uid), safe_component(series_id)
    member = f"{study_uid}/img/{study_uid}_{series_id}.nii.gz"
    with zipfile.ZipFile(path) as archive:
        names = []
        for item in archive.infolist():
            parts = PurePosixPath(item.filename)
            if (
                parts.is_absolute()
                or ".." in parts.parts
                or "\\" in item.filename
                or ":" in item.filename
                or stat.S_ISLNK(item.external_attr >> 16)
            ):
                raise ValueError("Unsafe path or symlink in study ZIP")
            names.append(item.filename)
        if names.count(member) != 1:
            raise ValueError("Selected series NIfTI is missing or duplicated in study ZIP")
        info = archive.getinfo(member)
        if info.file_size > max_member_bytes:
            raise ValueError("Selected NIfTI exceeds archive member size limit")
        # Never extractall: copy one exact member into a generated temporary filename.
        with tempfile.TemporaryDirectory(prefix="mrrate-") as temporary:
            target = Path(temporary) / "selected.nii.gz"
            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            return load_volume(target)


def axial_slices(
    volume: np.ndarray, fractions: tuple = (0.2, 0.5, 0.8), percentiles: tuple = (1.0, 99.0)
) -> tuple[list[Image.Image], dict]:
    if volume.ndim != 3:
        raise ValueError("Slice extraction requires a 3D array")
    if not fractions or any(not 0 <= f <= 1 for f in fractions):
        raise ValueError("Slice fractions must lie in [0, 1]")
    if not 0 <= percentiles[0] < percentiles[1] <= 100:
        raise ValueError("Invalid window percentiles")
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    foreground = np.abs(volume) > 1e-8
    counts = foreground.sum(axis=(0, 1))
    minimum = max(4, int(np.prod(volume.shape[:2]) * 0.005), int(counts.max() * 0.1))
    eligible = np.flatnonzero(counts >= minimum)
    if len(eligible) < len(fractions):
        raise ValueError("Insufficient nonblank axial slices for the requested slice policy")
    lo, hi = np.percentile(volume[foreground], percentiles)
    if hi <= lo:
        lo = min(0.0, float(lo))
        if hi <= lo:
            raise ValueError("Foreground has no usable intensity range")
    indices = []
    for fraction in fractions:
        target = eligible[0] + fraction * (eligible[-1] - eligible[0])
        candidates = sorted(
            (int(i) for i in eligible if i not in indices), key=lambda i: (abs(i - target), i)
        )
        indices.append(candidates[0])
    images = []
    for index in indices:
        pixels = np.clip((volume[:, :, index] - lo) / (hi - lo), 0, 1)
        # Canonical RAS: display anterior at top, patient's right at screen right.
        pixels = np.flipud(pixels.T)
        grayscale = np.rint(pixels * 255).astype(np.uint8)
        images.append(Image.fromarray(np.repeat(grayscale[..., None], 3, axis=-1)))
    return images, {
        "slice_indices": indices,
        "slice_fractions": list(fractions),
        "window": [float(lo), float(hi)],
        "window_percentiles": list(percentiles),
        "eligible_axial_slices": len(eligible),
        "foreground": "finite nonzero voxels",
    }


def save_montage(images: list[Image.Image], path: Path, title: str) -> None:
    width, height = 240, 260
    canvas = Image.new("RGB", (width * len(images), height + 35), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="white")
    for index, image in enumerate(images):
        scale = min(width / image.width, (height - 15) / image.height)
        resized = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.BILINEAR,
        )
        canvas.paste(resized, (index * width + (width - resized.width) // 2, 35))
        draw.text((index * width + 8, height + 10), f"Slice {index + 1}", fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def orthogonal_views(volume: np.ndarray) -> list[Image.Image]:
    result = []
    for axis in range(3):
        pixels = np.take(volume, volume.shape[axis] // 2, axis=axis)
        finite = pixels[np.isfinite(pixels)]
        lo, hi = np.percentile(finite, (1, 99)) if finite.size else (0, 1)
        scaled = np.clip((np.nan_to_num(pixels) - lo) / max(float(hi - lo), 1e-8), 0, 1)
        result.append(Image.fromarray(np.uint8(np.flipud(scaled.T) * 255)).convert("RGB"))
    return result


def prepare_manifest_qc(manifest, raw_dir: Path, output_dir: Path, config) -> dict:
    """Prepare local-only slice QC for the selected manifest without importing a model."""
    from .dataset import validate_manifest
    from .utils import private_id, write_json

    validate_manifest(manifest, config.source_batches)
    selected = manifest.loc[manifest.split.eq(config.split)].head(config.max_studies)
    if selected.empty:
        raise ValueError("No studies in the requested QC split")
    missing = sum(not (raw_dir / row.image_repo_path).is_file() for row in selected.itertuples())
    if missing:
        raise ValueError(
            f"{missing} selected MRI ZIP(s) are not downloaded yet; QC cohort unchanged"
        )
    records = []
    for row in selected.itertuples():
        volume, volume_qc = load_zip_volume(
            raw_dir / row.image_repo_path, row.study_uid, row.series_id
        )
        images, slice_qc = axial_slices(volume, config.slice_fractions, config.percentiles)
        identifier = private_id(row.study_uid)
        kind = getattr(row, "data_kind", "real")
        label = "SYNTHETIC - " if kind == "synthetic" else "Local research QC - "
        save_montage(images, output_dir / f"{identifier}.png", label + identifier)
        records.append(
            {"study_hash": identifier, "data_kind": kind, "volume": volume_qc, "slices": slice_qc}
        )
    summary = {
        "n": len(records),
        "split": config.split,
        "source_batches": sorted(selected.source_batch.unique()),
        "model_loaded": False,
        "studies": records,
    }
    write_json(output_dir / "qc_summary.json", summary)
    return {"n": len(records), "qc_dir": str(output_dir), "model_loaded": False}
