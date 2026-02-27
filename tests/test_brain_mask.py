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

    def test_implements_protocol(self):
        assert isinstance(ScipyBrainMask(), BrainMaskStrategy)


class TestSynthStripBrainMask:
    def test_is_available_false_without_freesurfer(self, monkeypatch):
        monkeypatch.delenv("FREESURFER_HOME", raising=False)
        monkeypatch.setattr("shutil.which", lambda _: None)
        assert SynthStripBrainMask().is_available() is False

    def test_is_available_true_when_on_path(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/mri_synthstrip")
        assert SynthStripBrainMask().is_available() is True

    def test_is_available_true_via_freesurfer_home(self, monkeypatch, tmp_path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        exe = bin_dir / "mri_synthstrip"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
        monkeypatch.setenv("FREESURFER_HOME", str(tmp_path))
        monkeypatch.setattr("shutil.which", lambda _: None)
        assert SynthStripBrainMask().is_available() is True

    def test_compute_raises_when_unavailable(self, monkeypatch):
        monkeypatch.delenv("FREESURFER_HOME", raising=False)
        monkeypatch.setattr("shutil.which", lambda _: None)
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

        def fake_run(cmd, *, capture_output, text):
            # cmd[-1] is the mask output path; write a NIfTI there
            mask_path = cmd[cmd.index("-m") + 1]
            nib.save(
                nib.Nifti1Image(expected_mask.astype(np.float32), affine), mask_path
            )
            result = MagicMock()
            result.returncode = 0
            return result

        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/mri_synthstrip")
        monkeypatch.setattr("subprocess.run", fake_run)

        mask = SynthStripBrainMask().compute(volume, affine)

        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})
        np.testing.assert_array_equal(mask, expected_mask)

    def test_compute_raises_on_subprocess_failure(self, monkeypatch):
        def fake_run(cmd, *, capture_output, text):
            result = MagicMock()
            result.returncode = 1
            result.stderr = "some error"
            return result

        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/mri_synthstrip")
        monkeypatch.setattr("subprocess.run", fake_run)

        with pytest.raises(RuntimeError, match="mri_synthstrip failed"):
            SynthStripBrainMask().compute(
                np.zeros((5, 5, 5), dtype=np.float32), np.eye(4)
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
        assert "scipy" in names

    def test_available_strategies_come_first(self, monkeypatch):
        """Available strategies should precede unavailable ones."""
        monkeypatch.delenv("FREESURFER_HOME", raising=False)
        monkeypatch.setattr("shutil.which", lambda _: None)

        strategies = get_available_strategies()
        availability = [s.is_available() for s in strategies]

        # Once we hit the first False, all subsequent must also be False
        seen_unavailable = False
        for avail in availability:
            if not avail:
                seen_unavailable = True
            if seen_unavailable:
                assert not avail, "Available strategy found after an unavailable one"

    def test_all_implement_protocol(self):
        for strategy in get_available_strategies():
            assert isinstance(strategy, BrainMaskStrategy)
