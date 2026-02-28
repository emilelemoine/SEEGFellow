# tests/test_brain_mask.py
import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow")
)

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from SEEGFellowLib.brain_mask import (
    BrainMaskStrategy,
    ScipyBrainMask,
    SynthStripBrainMask,
    get_available_strategies,
)


def _sphere_volume(radius: int = 10, size: int = 30) -> np.ndarray:
    """Return a 3-D float32 volume with a bright sphere at the centre."""
    volume = np.zeros((size, size, size), dtype=np.float32)
    center = np.array([size // 2, size // 2, size // 2])
    coords = np.indices((size, size, size)).transpose(1, 2, 3, 0)
    volume[np.linalg.norm(coords - center, axis=-1) < radius] = 1000.0
    return volume


class TestScipyBrainMask:
    def test_is_available_returns_true(self):
        assert ScipyBrainMask().is_available() is True

    def test_compute_returns_binary_uint8(self):
        volume = _sphere_volume()
        affine = np.eye(4)
        mask = ScipyBrainMask().compute(volume, affine)

        assert mask.shape == volume.shape
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})
        assert np.any(mask)

    def test_compute_respects_affine_spacing(self):
        """With 2 mm voxels the erosion should use fewer iterations than 1 mm."""
        volume = _sphere_volume(radius=12, size=40)
        affine_1mm = np.eye(4)
        affine_2mm = np.diag([2.0, 2.0, 2.0, 1.0])

        mask_1mm = ScipyBrainMask().compute(volume, affine_1mm)
        mask_2mm = ScipyBrainMask().compute(volume, affine_2mm)

        # 2 mm voxels → fewer erosion iterations → larger brain mask
        assert mask_2mm.sum() >= mask_1mm.sum()

    def test_compute_raises_on_empty_volume(self):
        with pytest.raises(ValueError, match="volume is empty"):
            ScipyBrainMask().compute(np.zeros((0, 0, 0), dtype=np.float32), np.eye(4))

    def test_compute_raises_on_empty_mask(self):
        """All-zero volume produces no foreground voxels → brain mask is empty."""
        volume = np.zeros((20, 20, 20), dtype=np.float32)
        with pytest.raises(RuntimeError, match="brain mask is empty"):
            ScipyBrainMask().compute(volume, np.eye(4))

    def test_implements_protocol(self):
        assert isinstance(ScipyBrainMask(), BrainMaskStrategy)


class TestSynthStripBrainMask:
    def test_is_available_false_without_freesurfer(self, monkeypatch):
        monkeypatch.delenv("FREESURFER_HOME", raising=False)
        monkeypatch.setattr("shutil.which", lambda *a, **kw: None)
        assert SynthStripBrainMask().is_available() is False

    def test_is_available_true_when_on_path(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/usr/bin/mri_synthstrip")
        assert SynthStripBrainMask().is_available() is True

    def test_is_available_true_via_freesurfer_home(self, monkeypatch, tmp_path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        exe = bin_dir / "mri_synthstrip"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
        monkeypatch.setenv("FREESURFER_HOME", str(tmp_path))
        monkeypatch.setattr("shutil.which", lambda *a, **kw: None)
        assert SynthStripBrainMask().is_available() is True

    def test_compute_raises_when_unavailable(self, monkeypatch):
        monkeypatch.delenv("FREESURFER_HOME", raising=False)
        monkeypatch.setattr("shutil.which", lambda *a, **kw: None)
        strategy = SynthStripBrainMask()
        with pytest.raises(RuntimeError, match="mri_synthstrip not found"):
            strategy.compute(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4))

    def test_compute_with_mocked_subprocess(self, monkeypatch, tmp_path):
        """compute() should return the mask NIfTI written by the subprocess."""
        import nibabel as nib

        volume = _sphere_volume()
        affine = np.eye(4)

        # Pre-build the expected mask that the fake subprocess will "produce"
        expected_mask = np.zeros(volume.shape, dtype=np.uint8)
        expected_mask[12:18, 12:18, 12:18] = 1

        def fake_run(cmd, *, capture_output, text, env=None):
            # cmd[-1] is the mask output path; write a NIfTI there
            mask_path = cmd[cmd.index("-m") + 1]
            nib.save(
                nib.Nifti1Image(expected_mask.astype(np.float32), affine), mask_path
            )
            result = MagicMock()
            result.returncode = 0
            return result

        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/usr/bin/mri_synthstrip")
        monkeypatch.setattr("subprocess.run", fake_run)

        mask = SynthStripBrainMask().compute(volume, affine)

        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})
        np.testing.assert_array_equal(mask, expected_mask)

    def test_compute_raises_on_subprocess_failure(self, monkeypatch):
        def fake_run(cmd, *, capture_output, text, env=None):
            result = MagicMock()
            result.returncode = 1
            result.stderr = "some error"
            return result

        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/usr/bin/mri_synthstrip")
        monkeypatch.setattr("subprocess.run", fake_run)

        with pytest.raises(RuntimeError, match="mri_synthstrip failed"):
            SynthStripBrainMask().compute(
                np.zeros((5, 5, 5), dtype=np.float32), np.eye(4)
            )

    def test_compute_raises_on_empty_mask_output(self, monkeypatch):
        """compute() raises RuntimeError when mri_synthstrip writes an all-zero mask."""
        import nibabel as nib

        volume = _sphere_volume()
        affine = np.eye(4)
        empty_mask = np.zeros(volume.shape, dtype=np.uint8)

        def fake_run(cmd, *, capture_output, text, env=None):
            mask_path = cmd[cmd.index("-m") + 1]
            nib.save(nib.Nifti1Image(empty_mask.astype(np.float32), affine), mask_path)
            result = MagicMock()
            result.returncode = 0
            return result

        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/usr/bin/mri_synthstrip")
        monkeypatch.setattr("subprocess.run", fake_run)

        with pytest.raises(RuntimeError, match="brain mask is empty"):
            SynthStripBrainMask().compute(volume, affine)

    def test_compute_transposes_volume_axes_for_nifti_io(self, monkeypatch):
        """Volume must be transposed (K,J,I)→(I,J,K) when writing to NIfTI.

        Slicer's arrayFromVolume returns arrays in (K,J,I) order, but NIfTI
        convention (used by SynthStrip) expects (I,J,K).  Passing an un-
        transposed array misaligns the data relative to the affine, producing
        distorted masks (e.g. blobs in the posterior fossa).
        """
        import nibabel as nib

        # Non-square shape so any axis swap is unambiguous.
        # In Slicer convention: K=10, J=20, I=30
        volume = np.zeros((10, 20, 30), dtype=np.float32)
        volume[5, 10, 15] = 1000.0  # marker at (K=5, J=10, I=15)
        affine = np.eye(4)

        # Capture the shape of the array saved as the input NIfTI.
        real_save = nib.save
        saved_input_shapes: list[tuple] = []

        def capturing_save(img, path):
            if "input" in path:
                saved_input_shapes.append(np.asarray(img.dataobj).shape)
            real_save(img, path)

        def fake_run(cmd, *, capture_output, text, env=None):
            # SynthStrip outputs a mask in the same NIfTI (I,J,K) space.
            # Physical voxel at (K=5,J=10,I=15) → NIfTI index (I=15,J=10,K=5).
            mask_path = cmd[cmd.index("-m") + 1]
            mask_nifti = np.zeros((30, 20, 10), dtype=np.float32)
            mask_nifti[15, 10, 5] = 1.0
            real_save(nib.Nifti1Image(mask_nifti, affine), mask_path)
            r = MagicMock()
            r.returncode = 0
            return r

        monkeypatch.setattr(nib, "save", capturing_save)
        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/usr/bin/mri_synthstrip")
        monkeypatch.setattr("subprocess.run", fake_run)

        mask = SynthStripBrainMask().compute(volume, affine)

        # Input NIfTI must be volume.T = (I=30, J=20, K=10)
        assert saved_input_shapes == [(30, 20, 10)], (
            f"Expected input NIfTI shape (30,20,10), got {saved_input_shapes}. "
            "Volume was not transposed before saving (NIfTI needs (I,J,K) order)."
        )

        # Returned mask must be back in Slicer's (K,J,I) = (10,20,30) shape.
        assert mask.shape == (10, 20, 30)

        # The physical voxel (K=5,J=10,I=15) must be marked in the output.
        assert mask[5, 10, 15] == 1, (
            "Mask content was not correctly transposed back from NIfTI (I,J,K) "
            "to Slicer (K,J,I) order."
        )

    def test_implements_protocol(self):
        assert isinstance(SynthStripBrainMask(), BrainMaskStrategy)


class TestGetAvailableStrategies:
    def test_returns_list(self):
        strategies = get_available_strategies()
        assert isinstance(strategies, list)
        assert len(strategies) >= 1

    def test_scipy_always_present(self):
        strategies = get_available_strategies()
        names = [s.name for s in strategies]
        assert "Scipy (morphological)" in names

    def test_available_strategies_come_first(self, monkeypatch):
        """Available strategies should precede unavailable ones."""
        monkeypatch.delenv("FREESURFER_HOME", raising=False)
        monkeypatch.setattr("shutil.which", lambda *a, **kw: None)

        strategies = get_available_strategies()
        availability = [s.is_available() for s in strategies]

        # Once we hit the first False, all subsequent must also be False
        seen_unavailable = False
        for avail in availability:
            if not avail:
                seen_unavailable = True
            if seen_unavailable:
                assert not avail, "Available strategy found after an unavailable one"

    def test_synthstrip_before_scipy_when_both_available(self, monkeypatch):
        """SynthStrip should appear before Scipy when it is available."""
        monkeypatch.setattr(
            "SEEGFellowLib.brain_mask.SynthStripBrainMask.is_available",
            lambda self: True,
        )
        strategies = get_available_strategies()
        names = [s.name for s in strategies]
        assert names.index("SynthStrip (FreeSurfer)") < names.index(
            "Scipy (morphological)"
        )

    def test_all_implement_protocol(self):
        for strategy in get_available_strategies():
            assert isinstance(strategy, BrainMaskStrategy)
