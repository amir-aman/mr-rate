import zipfile

import nibabel as nib
import numpy as np
import pytest

from mrrate_vlm.imaging import axial_slices, load_volume, load_zip_volume, orthogonal_views


def test_canonical_nonfinite_and_rgb(tmp_path):
    data = np.ones((12, 13, 14), dtype=np.float32)
    data[:, :, :] *= np.arange(14) + 1
    data[0, 0, :3] = [np.nan, np.inf, -np.inf]
    path = tmp_path / "volume.nii.gz"
    nib.save(nib.Nifti1Image(data, np.diag([-1.0, 2.0, 3.0, 1.0])), path)
    volume, qc = load_volume(path)
    assert np.isfinite(volume).all()
    assert qc["canonical_orientation"] == "RAS"
    assert qc["nonfinite_replaced"] == 3
    images, slices = axial_slices(volume)
    assert len(images) == len(set(slices["slice_indices"])) == 3
    for image in images:
        pixels = np.asarray(image)
        assert pixels.shape == (13, 12, 3)
        assert pixels.dtype == np.uint8
        np.testing.assert_array_equal(pixels[..., 0], pixels[..., 2])
    assert len(orthogonal_views(volume)) == 3


def test_zip_reads_exact_series(fixture_root):
    path = fixture_root / "raw/mri/batch01/synthetic-study-000.zip"
    data, qc = load_zip_volume(path, "synthetic-study-000", "flair-raw-axi")
    assert data.shape == (24, 26, 20)
    assert qc["original_orientation"] == "LAS"
    with pytest.raises(ValueError, match="missing"):
        load_zip_volume(path, "synthetic-study-000", "t2w-raw-axi")


@pytest.mark.parametrize("name", ["../escape.nii.gz", "/absolute", "C:/absolute", "x\\..\\evil"])
def test_zip_slip_rejected(tmp_path, name):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name, "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        load_zip_volume(path, "s", "flair")
    assert not (tmp_path / "escape.nii.gz").exists()


def test_invalid_and_blank_volumes(tmp_path):
    with pytest.raises(ValueError, match="nonblank"):
        axial_slices(np.zeros((12, 12, 12)))
    path = tmp_path / "4d.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3, 2)), np.eye(4)), path)
    with pytest.raises(ValueError, match="3D"):
        load_volume(path)
