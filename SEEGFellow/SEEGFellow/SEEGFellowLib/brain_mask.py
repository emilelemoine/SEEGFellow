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

    name = "Scipy (morphological)"

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
        if volume.size == 0:
            raise ValueError("volume is empty")

        voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        min_voxel_mm = float(np.clip(voxel_sizes.min(), 0.1, None))

        foreground = volume > volume.max() * 0.05
        filled = ndimage.binary_fill_holes(
            ndimage.binary_closing(foreground, iterations=2)
        )

        labeled, n = ndimage.label(filled)
        if n == 0:
            raise RuntimeError("brain mask is empty")
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

    name = "SynthStrip (FreeSurfer)"

    def _resolve_command(self, executable: str) -> list[str]:
        """Return the argv prefix needed to run *executable* safely.

        Scripts with a ``#!/usr/bin/env python`` shebang fail on macOS when
        ``python`` is not installed (only ``python3`` is).  If the shebang
        requests any Python variant we find a concrete system Python and
        prepend it so the OS shebang resolver is bypassed entirely.

        Returns ``[executable]`` unchanged if the shebang is not Python.
        """
        try:
            with open(executable, "rb") as fh:
                first = fh.readline().decode("utf-8", errors="replace").strip()
        except OSError:
            return [executable]

        if not first.startswith("#!") or "python" not in first:
            return [executable]

        # Build a system-only search path (same set as _build_subprocess_env).
        freesurfer_home = os.environ.get("FREESURFER_HOME", "")
        parts: list[str] = []
        if freesurfer_home:
            parts.append(os.path.join(freesurfer_home, "bin"))
        parts += [
            "/usr/local/bin",
            "/opt/homebrew/bin",
            "/usr/bin",
            "/bin",
            "/opt/local/bin",
        ]
        search = ":".join(parts)

        for name in ("python3", "python"):
            found = shutil.which(name, path=search)
            if found:
                return [found, executable]

        return [executable]

    def _build_subprocess_env(self) -> dict[str, str]:
        """Return an env dict where system paths precede Slicer-internal Python paths.

        Slicer prepends its bundled Python to PATH.  When mri_synthstrip's shebang
        (``#!/usr/bin/env python``) resolves against that Python it gets
        "Permission denied" (exit 126) because the framework binary can't be
        invoked directly.  We fix this by moving well-known system directories to
        the front so that the OS-level python/python3 is found first.
        """
        env = os.environ.copy()

        # Prepend FREESURFER_HOME/bin and common system paths.
        freesurfer_home = env.get("FREESURFER_HOME", "")
        priority: list[str] = []
        if freesurfer_home:
            priority.append(os.path.join(freesurfer_home, "bin"))
        priority += [
            "/usr/local/bin",
            "/opt/homebrew/bin",
            "/usr/bin",
            "/bin",
            "/opt/local/bin",
        ]

        # Keep existing PATH entries that are NOT Slicer-internal Python dirs,
        # de-duplicating against what we already put in priority.
        existing = env.get("PATH", "").split(":")
        seen = set(priority)
        tail = [
            p
            for p in existing
            if p not in seen and ("Slicer" not in p and "slicer" not in p)
        ]

        env["PATH"] = ":".join(priority + tail)

        # Remove Slicer-injected Python env vars so the subprocess uses its
        # own stdlib instead of Slicer's bundled one (which causes SRE mismatch).
        for var in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONNOUSERSITE"):
            env.pop(var, None)

        return env

    def _find_executable(self) -> str | None:
        """Return the path to mri_synthstrip, or None if not found."""
        exe = shutil.which("mri_synthstrip")
        if exe is not None:
            return exe
        freesurfer_home = os.environ.get("FREESURFER_HOME", "")
        if freesurfer_home:
            candidate = os.path.join(freesurfer_home, "bin", "mri_synthstrip")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        # GUI apps (e.g. Slicer) may have a stripped PATH that excludes
        # user-managed directories, so also search common locations.
        fallback = shutil.which(
            "mri_synthstrip",
            path="/usr/local/bin:/opt/homebrew/bin:/opt/local/bin",
        )
        return fallback

    def is_available(self) -> bool:
        """Return True if mri_synthstrip is found on PATH or in FREESURFER_HOME."""
        return self._find_executable() is not None

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

        executable = self._find_executable()
        if executable is None:
            raise RuntimeError(
                "mri_synthstrip not found. Install FreeSurfer or add mri_synthstrip to PATH."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.nii.gz")
            mask_path = os.path.join(tmpdir, "mask.nii.gz")
            brain_path = os.path.join(tmpdir, "brain.nii.gz")

            # Slicer arrays are (K,J,I); NIfTI expects (I,J,K) → transpose.
            nib.save(nib.Nifti1Image(volume.T, affine), input_path)

            result = subprocess.run(
                self._resolve_command(executable)
                + ["-i", input_path, "-o", brain_path, "-m", mask_path],
                capture_output=True,
                text=True,
                env=self._build_subprocess_env(),
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"mri_synthstrip failed (exit {result.returncode}):\n{result.stderr}"
                )

            mask_img = nib.load(mask_path)
            # Transpose back from NIfTI (I,J,K) to Slicer's (K,J,I).
            mask = np.asarray(mask_img.dataobj).T

        result = (mask > 0).astype(np.uint8)
        if result.sum() == 0:
            raise RuntimeError("brain mask is empty")
        return result


def get_available_strategies() -> list[BrainMaskStrategy]:
    """Return all brain mask strategies, with available ones first.

    Example::

        strategies = get_available_strategies()
        mask = strategies[0].compute(volume, affine)
    """
    all_strategies: list[BrainMaskStrategy] = [SynthStripBrainMask(), ScipyBrainMask()]
    available = [s for s in all_strategies if s.is_available()]
    unavailable = [s for s in all_strategies if not s.is_available()]
    return available + unavailable
