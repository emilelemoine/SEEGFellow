# SEEGFellow/SEEGFellow/SEEGFellowLib/brain_mask.py
"""Brain mask strategies for T1-weighted MRI brain extraction.

Provides a Protocol (BrainMaskStrategy) and two implementations:
- ScipyBrainMask: morphological approach, always available
- SynthStripBrainMask: FreeSurfer mri_synthstrip, requires FreeSurfer install
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Protocol, runtime_checkable

import numpy as np
from scipy import ndimage


@runtime_checkable
class BrainMaskStrategy(Protocol):
    """Protocol for brain mask extraction strategies.

    Example::

        strategy = ScipyBrainMask()
        if strategy.is_available():
            mask = strategy.compute(volume, affine)
    """

    name: str

    def compute(self, volume: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """Compute a binary brain mask from a T1-weighted MRI volume.

        Args:
            volume: 3-D numpy array (arbitrary intensity scale).
            affine: 4x4 voxel-to-world (IJK-to-RAS) transformation matrix.

        Returns:
            Binary uint8 mask (1 = brain, 0 = outside).
        """
        ...

    def is_available(self) -> bool:
        """Return True if this strategy can run on the current system."""
        ...


class ScipyBrainMask:
    """Brain extraction using scipy morphological operations.

    Always available (no external dependencies beyond scipy).

    Algorithm:
    1. Threshold at 5% of max intensity
    2. Morphological closing + hole filling
    3. Keep largest connected component
    4. Erode ~5 mm to strip the skull

    Example::

        strategy = ScipyBrainMask()
        mask = strategy.compute(volume, affine)
    """

    name = "scipy"

    def is_available(self) -> bool:
        return True

    def compute(self, volume: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """Compute brain mask via morphological operations.

        Args:
            volume: 3-D numpy array (arbitrary intensity scale).
            affine: 4x4 voxel-to-world (IJK-to-RAS) transformation matrix.

        Returns:
            Binary uint8 mask (1 = brain, 0 = outside).

        Example::

            mask = ScipyBrainMask().compute(t1_array, np.eye(4))
        """
        voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        min_voxel_mm = float(np.clip(voxel_sizes.min(), 0.1, None))

        foreground = volume > volume.max() * 0.05
        filled = ndimage.binary_fill_holes(
            ndimage.binary_closing(foreground, iterations=2)
        )

        labeled, n = ndimage.label(filled)
        if n == 0:
            return foreground.astype(np.uint8)
        sizes = ndimage.sum(filled, labeled, range(1, n + 1))
        head = labeled == (int(np.argmax(sizes)) + 1)

        erosion_voxels = max(1, int(round(5.0 / min_voxel_mm)))
        brain = ndimage.binary_erosion(head, iterations=erosion_voxels)

        return brain.astype(np.uint8)


class SynthStripBrainMask:
    """Brain extraction using FreeSurfer's mri_synthstrip.

    Requires mri_synthstrip on PATH or under $FREESURFER_HOME/bin/.
    Produces higher-quality masks than the scipy approach.

    Example::

        strategy = SynthStripBrainMask()
        if strategy.is_available():
            mask = strategy.compute(volume, affine)
    """

    name = "synthstrip"

    def is_available(self) -> bool:
        """Return True if mri_synthstrip is found on PATH or in FREESURFER_HOME."""
        if shutil.which("mri_synthstrip") is not None:
            return True
        freesurfer_home = os.environ.get("FREESURFER_HOME", "")
        if freesurfer_home:
            candidate = os.path.join(freesurfer_home, "bin", "mri_synthstrip")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return True
        return False

    def compute(self, volume: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """Compute brain mask using mri_synthstrip.

        Writes the volume to a temporary NIfTI file, runs mri_synthstrip,
        and reads back the resulting mask. Cleans up temp files on exit.

        Args:
            volume: 3-D numpy array (arbitrary intensity scale).
            affine: 4x4 voxel-to-world (IJK-to-RAS) transformation matrix.

        Returns:
            Binary uint8 mask (1 = brain, 0 = outside).

        Raises:
            RuntimeError: If mri_synthstrip is not available or fails.

        Example::

            mask = SynthStripBrainMask().compute(t1_array, affine)
        """
        import nibabel as nib  # noqa: PLC0415 – kept lazy for clarity

        if not self.is_available():
            raise RuntimeError(
                "mri_synthstrip not found. Install FreeSurfer or add mri_synthstrip to PATH."
            )

        executable = shutil.which("mri_synthstrip")
        if executable is None:
            freesurfer_home = os.environ.get("FREESURFER_HOME", "")
            executable = os.path.join(freesurfer_home, "bin", "mri_synthstrip")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.nii.gz")
            mask_path = os.path.join(tmpdir, "mask.nii.gz")
            brain_path = os.path.join(tmpdir, "brain.nii.gz")

            nib.save(nib.Nifti1Image(volume, affine), input_path)

            result = subprocess.run(
                [executable, "-i", input_path, "-o", brain_path, "-m", mask_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"mri_synthstrip failed (exit {result.returncode}):\n{result.stderr}"
                )

            mask_img = nib.load(mask_path)
            mask = np.asarray(mask_img.dataobj)

        return (mask > 0).astype(np.uint8)


def get_available_strategies() -> list[BrainMaskStrategy]:
    """Return all brain mask strategies, with available ones first.

    Example::

        strategies = get_available_strategies()
        mask = strategies[0].compute(volume, affine)
    """
    all_strategies: list[BrainMaskStrategy] = [ScipyBrainMask(), SynthStripBrainMask()]
    available = [s for s in all_strategies if s.is_available()]
    unavailable = [s for s in all_strategies if not s.is_available()]
    return available + unavailable
