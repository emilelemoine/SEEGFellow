# SEEGFellow/SEEGFellow/SEEGFellowLib/brain_mask.py
"""Brain mask via FreeSurfer SynthSeg volumetric parcellation.

Provides a single strategy (SynthSegBrainMask) that runs mri_synthseg --parc
to produce a labeled parcellation. The binary brain mask is derived by
binarizing the parcellation (label > 0 = brain). The full parcellation is
stored for downstream contact labeling.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BrainMaskStrategy(Protocol):
    """Protocol for brain mask extraction strategies.

    Example::

        strategy = SynthSegBrainMask()
        if strategy.is_available():
            mask = strategy.compute(volume, affine)
    """

    name: str

    def compute(
        self,
        volume: np.ndarray,
        affine: np.ndarray,
        output_dir: str | None = None,
    ) -> np.ndarray:
        """Compute a binary brain mask from a T1-weighted MRI volume.

        Args:
            volume: 3-D numpy array (arbitrary intensity scale).
            affine: 4x4 voxel-to-world (IJK-to-RAS) transformation matrix.
            output_dir: If set, save and reuse cached results in this directory.

        Returns:
            Binary uint8 mask (1 = brain, 0 = outside).
        """
        ...

    def is_available(self) -> bool:
        """Return True if this strategy can run on the current system."""
        ...


class SynthSegBrainMask:
    """Brain extraction + parcellation using FreeSurfer's mri_synthseg.

    Runs ``mri_synthseg --parc`` to produce a volumetric parcellation
    (~100 DKT regions). The binary brain mask is derived by binarizing
    the parcellation (label > 0). The full parcellation is stored as
    ``self.parcellation`` for downstream contact labeling.

    Args:
        robust: Use ``--robust`` mode (default True). Set False for ``--fast``.
        threads: Number of CPU threads for SynthSeg (default 1).

    Example::

        strategy = SynthSegBrainMask(robust=True, threads=2)
        if strategy.is_available():
            mask = strategy.compute(volume, affine)
            parc = strategy.parcellation  # int32 parcellation array
    """

    name = "SynthSeg (FreeSurfer)"

    def __init__(self, robust: bool = True, threads: int = 1) -> None:
        self.robust = robust
        self.threads = threads
        self.parcellation: np.ndarray | None = None
        self.parcellation_affine: np.ndarray | None = None

    def _build_subprocess_env(self) -> dict[str, str]:
        """Return an env dict where system paths precede Slicer-internal paths."""
        env = os.environ.copy()

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

        existing = env.get("PATH", "").split(":")
        seen = set(priority)
        tail = [
            p
            for p in existing
            if p not in seen and ("Slicer" not in p and "slicer" not in p)
        ]

        env["PATH"] = ":".join(priority + tail)

        for var in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONNOUSERSITE"):
            env.pop(var, None)

        return env

    def _find_executable(self) -> str | None:
        """Return the path to mri_synthseg, or None if not found."""
        exe = shutil.which("mri_synthseg")
        if exe is not None:
            return exe
        freesurfer_home = os.environ.get("FREESURFER_HOME", "")
        if freesurfer_home:
            candidate = os.path.join(freesurfer_home, "bin", "mri_synthseg")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        fallback = shutil.which(
            "mri_synthseg",
            path="/usr/local/bin:/opt/homebrew/bin:/opt/local/bin",
        )
        return fallback

    def _resolve_command(self, executable: str) -> list[str]:
        """Return the argv prefix needed to run *executable* safely.

        Scripts with a ``#!/usr/bin/env python`` shebang fail on macOS when
        ``python`` is not installed (only ``python3`` is). If the shebang
        requests any Python variant we find a concrete system Python and
        prepend it so the OS shebang resolver is bypassed entirely.
        """
        try:
            with open(executable, "rb") as fh:
                first = fh.readline().decode("utf-8", errors="replace").strip()
        except OSError:
            return [executable]

        if not first.startswith("#!") or "python" not in first:
            return [executable]

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

    def is_available(self) -> bool:
        """Return True if mri_synthseg is found on PATH or in FREESURFER_HOME."""
        return self._find_executable() is not None

    def compute(
        self,
        volume: np.ndarray,
        affine: np.ndarray,
        output_dir: str | None = None,
    ) -> np.ndarray:
        """Run SynthSeg parcellation and return binarized brain mask.

        After calling this method, ``self.parcellation`` contains the full
        int32 parcellation array (Slicer K,J,I axis order) and
        ``self.parcellation_affine`` contains its 4x4 voxel-to-world matrix.

        Args:
            volume: 3-D numpy array (arbitrary intensity scale).
            affine: 4x4 voxel-to-world (IJK-to-RAS) transformation matrix.
            output_dir: If set, save the SynthSeg output here and reuse
                cached results on subsequent runs.

        Returns:
            Binary uint8 mask (1 = brain, 0 = outside).

        Raises:
            RuntimeError: If mri_synthseg is not available or fails.

        Example::

            strategy = SynthSegBrainMask()
            mask = strategy.compute(volume, affine, output_dir="/tmp/seeg")
        """
        import nibabel as nib

        # Check for cached result in output_dir
        cached_path = None
        if output_dir:
            cached_path = os.path.join(output_dir, "synthseg_parc.nii.gz")
            if os.path.isfile(cached_path):
                print(f"[SEEGFellow] Loading cached SynthSeg result: {cached_path}")
                seg_img = nib.load(cached_path)
                seg_affine = np.array(seg_img.affine)
                parcellation = np.asarray(seg_img.dataobj, dtype=np.int32).T
                self.parcellation = parcellation
                self.parcellation_affine = seg_affine
                mask = (parcellation > 0).astype(np.uint8)
                if mask.sum() == 0:
                    raise RuntimeError("cached brain mask is empty")
                return mask

        executable = self._find_executable()
        if executable is None:
            raise RuntimeError(
                "mri_synthseg not found. Install FreeSurfer or add mri_synthseg to PATH."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.nii.gz")
            output_path = os.path.join(tmpdir, "seg.nii.gz")

            # Slicer arrays are (K,J,I); NIfTI expects (I,J,K) → transpose.
            nib.save(nib.Nifti1Image(volume.T, affine), input_path)

            cmd = self._resolve_command(executable) + [
                "--i",
                input_path,
                "--o",
                output_path,
                "--parc",
                "--cpu",
                "--threads",
                str(self.threads),
            ]
            if self.robust:
                cmd.append("--robust")
            else:
                cmd.append("--fast")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=self._build_subprocess_env(),
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"mri_synthseg failed (exit {result.returncode}):\n{result.stderr}"
                )

            seg_img = nib.load(output_path)
            seg_affine = np.array(seg_img.affine)
            # Transpose from NIfTI (I,J,K) to Slicer's (K,J,I).
            parcellation = np.asarray(seg_img.dataobj, dtype=np.int32).T

            # Save to output_dir for future reuse
            if cached_path and output_dir:
                import shutil as _shutil

                os.makedirs(output_dir, exist_ok=True)
                _shutil.copy2(output_path, cached_path)
                print(f"[SEEGFellow] Saved SynthSeg result to: {cached_path}")

        self.parcellation = parcellation
        self.parcellation_affine = seg_affine

        mask = (parcellation > 0).astype(np.uint8)
        if mask.sum() == 0:
            raise RuntimeError("brain mask is empty")
        return mask


def get_available_strategies() -> list[BrainMaskStrategy]:
    """Return all brain mask strategies.

    Example::

        strategies = get_available_strategies()
        mask = strategies[0].compute(volume, affine)
    """
    return [SynthSegBrainMask()]
