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
    SynthSegBrainMask,
    get_available_strategies,
)


def _sphere_volume(radius: int = 10, size: int = 30) -> np.ndarray:
    """Return a 3-D float32 volume with a bright sphere at the centre."""
    volume = np.zeros((size, size, size), dtype=np.float32)
    center = np.array([size // 2, size // 2, size // 2])
    coords = np.indices((size, size, size)).transpose(1, 2, 3, 0)
    volume[np.linalg.norm(coords - center, axis=-1) < radius] = 1000.0
    return volume


class TestSynthSegBrainMask:
    def test_is_available_false_without_freesurfer(self, monkeypatch):
        monkeypatch.delenv("FREESURFER_HOME", raising=False)
        monkeypatch.setattr("shutil.which", lambda *a, **kw: None)
        assert SynthSegBrainMask().is_available() is False

    def test_is_available_true_when_on_path(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/usr/bin/mri_synthseg")
        assert SynthSegBrainMask().is_available() is True

    def test_is_available_true_via_freesurfer_home(self, monkeypatch, tmp_path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        exe = bin_dir / "mri_synthseg"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
        monkeypatch.setenv("FREESURFER_HOME", str(tmp_path))
        monkeypatch.setattr("shutil.which", lambda *a, **kw: None)
        assert SynthSegBrainMask().is_available() is True

    def test_compute_raises_when_unavailable(self, monkeypatch):
        monkeypatch.delenv("FREESURFER_HOME", raising=False)
        monkeypatch.setattr("shutil.which", lambda *a, **kw: None)
        strategy = SynthSegBrainMask()
        with pytest.raises(RuntimeError, match="mri_synthseg not found"):
            strategy.compute(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4))

    def test_compute_returns_binary_mask_from_parcellation(self, monkeypatch):
        """compute() should binarize the SynthSeg parcellation output."""
        import nibabel as nib

        volume = _sphere_volume()
        affine = np.eye(4)

        # SynthSeg outputs integer labels (not binary); build a fake parcellation
        # in NIfTI (I,J,K) space.  Volume is (K=30,J=30,I=30) → NIfTI (30,30,30).
        parcellation_nifti = np.zeros((30, 30, 30), dtype=np.int32)
        parcellation_nifti[12:18, 12:18, 12:18] = 17  # Left Hippocampus

        def fake_run(cmd, *, capture_output, text, env=None):
            out_path = cmd[cmd.index("--o") + 1]
            nib.save(nib.Nifti1Image(parcellation_nifti, affine), out_path)
            result = MagicMock()
            result.returncode = 0
            return result

        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/usr/bin/mri_synthseg")
        monkeypatch.setattr("subprocess.run", fake_run)

        strategy = SynthSegBrainMask()
        mask = strategy.compute(volume, affine)

        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})
        # The mask should be nonzero where parcellation was nonzero
        expected = (parcellation_nifti.T > 0).astype(np.uint8)  # NIfTI→Slicer transpose
        np.testing.assert_array_equal(mask, expected)

    def test_parcellation_stored_after_compute(self, monkeypatch):
        """After compute(), the full parcellation is available as self.parcellation."""
        import nibabel as nib

        volume = _sphere_volume()
        affine = np.eye(4)

        parcellation_nifti = np.zeros((30, 30, 30), dtype=np.int32)
        parcellation_nifti[14:16, 14:16, 14:16] = 53  # Right Hippocampus

        def fake_run(cmd, *, capture_output, text, env=None):
            out_path = cmd[cmd.index("--o") + 1]
            nib.save(nib.Nifti1Image(parcellation_nifti, affine), out_path)
            result = MagicMock()
            result.returncode = 0
            return result

        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/usr/bin/mri_synthseg")
        monkeypatch.setattr("subprocess.run", fake_run)

        strategy = SynthSegBrainMask()
        strategy.compute(volume, affine)

        assert strategy.parcellation is not None
        assert strategy.parcellation_affine is not None
        # Parcellation is transposed to Slicer (K,J,I) convention
        assert strategy.parcellation.shape == volume.shape
        # The label value should be preserved
        assert strategy.parcellation[15, 15, 15] == 53  # K=15,J=15,I=15 (symmetric)

    def test_compute_passes_robust_and_threads(self, monkeypatch):
        """Constructor args robust and threads should appear in the subprocess command."""
        import nibabel as nib

        volume = _sphere_volume()
        affine = np.eye(4)
        parcellation_nifti = np.zeros((30, 30, 30), dtype=np.int32)
        parcellation_nifti[14:16, 14:16, 14:16] = 10

        captured_cmds: list[list[str]] = []

        def fake_run(cmd, *, capture_output, text, env=None):
            captured_cmds.append(list(cmd))
            out_path = cmd[cmd.index("--o") + 1]
            nib.save(nib.Nifti1Image(parcellation_nifti, affine), out_path)
            result = MagicMock()
            result.returncode = 0
            return result

        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/usr/bin/mri_synthseg")
        monkeypatch.setattr("subprocess.run", fake_run)

        # Test robust mode (default)
        strategy = SynthSegBrainMask(robust=True, threads=4)
        strategy.compute(volume, affine)
        cmd = captured_cmds[-1]
        assert "--robust" in cmd
        assert "--threads" in cmd
        assert cmd[cmd.index("--threads") + 1] == "4"
        assert "--parc" in cmd
        assert "--cpu" in cmd

        # Test fast mode
        captured_cmds.clear()
        strategy = SynthSegBrainMask(robust=False, threads=2)
        strategy.compute(volume, affine)
        cmd = captured_cmds[-1]
        assert "--fast" in cmd
        assert "--robust" not in cmd
        assert cmd[cmd.index("--threads") + 1] == "2"

    def test_compute_raises_on_subprocess_failure(self, monkeypatch):
        def fake_run(cmd, *, capture_output, text, env=None):
            result = MagicMock()
            result.returncode = 1
            result.stderr = "some error"
            return result

        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/usr/bin/mri_synthseg")
        monkeypatch.setattr("subprocess.run", fake_run)

        with pytest.raises(RuntimeError, match="mri_synthseg failed"):
            SynthSegBrainMask().compute(
                np.zeros((5, 5, 5), dtype=np.float32), np.eye(4)
            )

    def test_compute_raises_on_empty_parcellation(self, monkeypatch):
        """compute() raises RuntimeError when SynthSeg produces all zeros."""
        import nibabel as nib

        volume = _sphere_volume()
        affine = np.eye(4)
        empty_parc = np.zeros((30, 30, 30), dtype=np.int32)

        def fake_run(cmd, *, capture_output, text, env=None):
            out_path = cmd[cmd.index("--o") + 1]
            nib.save(nib.Nifti1Image(empty_parc, affine), out_path)
            result = MagicMock()
            result.returncode = 0
            return result

        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/usr/bin/mri_synthseg")
        monkeypatch.setattr("subprocess.run", fake_run)

        with pytest.raises(RuntimeError, match="brain mask is empty"):
            SynthSegBrainMask().compute(volume, affine)

    def test_implements_protocol(self):
        assert isinstance(SynthSegBrainMask(), BrainMaskStrategy)


class TestGetAvailableStrategies:
    def test_returns_list(self):
        strategies = get_available_strategies()
        assert isinstance(strategies, list)
        assert len(strategies) == 1

    def test_synthseg_is_the_only_strategy(self):
        strategies = get_available_strategies()
        assert strategies[0].name == "SynthSeg (FreeSurfer)"

    def test_all_implement_protocol(self):
        for strategy in get_available_strategies():
            assert isinstance(strategy, BrainMaskStrategy)
