# SynthSeg Integration & Contact Labeling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace SynthStrip/Scipy brain mask with SynthSeg volumetric parcellation and add contact-to-region anatomical labeling.

**Architecture:** SynthSeg runs once on the T1 (via `mri_synthseg --parc` subprocess). Its output provides both the brain mask (binarized parcellation) and a region lookup table for contact labeling. A new `contact_labeler.py` module maps contact RAS coordinates to SynthSeg voxel labels.

**Tech Stack:** FreeSurfer CLI (`mri_synthseg`), nibabel, numpy, Qt widgets (QTableWidget, QComboBox, QSpinBox)

---

## Branch 1: `feat/synthseg-brain-mask`

Replace `SynthStripBrainMask` and `ScipyBrainMask` with `SynthSegBrainMask`.

### Task 1.1: Add `region` field to Contact dataclass

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_model.py:8-18`
- Test: `tests/test_electrode_model.py`

**Step 1: Write the failing test**

Add to `tests/test_electrode_model.py`:

```python
def test_contact_region_defaults_to_empty():
    c = Contact(index=1, position_ras=(0.0, 0.0, 0.0))
    assert c.region == ""


def test_contact_region_can_be_set():
    c = Contact(index=1, position_ras=(0.0, 0.0, 0.0), region="Left Hippocampus")
    assert c.region == "Left Hippocampus"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_electrode_model.py::test_contact_region_defaults_to_empty -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'region'`

**Step 3: Write minimal implementation**

In `SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_model.py`, add `region` field to `Contact`:

```python
@dataclass
class Contact:
    """A single electrode contact.

    Example::

        c = Contact(index=1, position_ras=(10.0, 20.0, 30.0), label="A1")
    """

    index: int  # 1-based (1 = deepest/mesial)
    position_ras: tuple[float, float, float]  # (R, A, S) in mm
    label: str = ""
    region: str = ""
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_electrode_model.py -v`
Expected: All pass

**Step 5: Commit**

```
git add SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_model.py tests/test_electrode_model.py
git commit -m "feat: add region field to Contact dataclass"
```

---

### Task 1.2: Implement `SynthSegBrainMask` class

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/brain_mask.py`
- Test: `tests/test_brain_mask.py`

**Step 1: Write the failing tests**

Replace the existing `TestSynthStripBrainMask` and `TestScipyBrainMask` classes and add `TestSynthSegBrainMask`. Update imports at the top of `tests/test_brain_mask.py`:

```python
from SEEGFellowLib.brain_mask import (
    BrainMaskStrategy,
    SynthSegBrainMask,
    get_available_strategies,
)
```

Remove `TestScipyBrainMask` and `TestSynthStripBrainMask` classes entirely (those classes will be deleted from the source). Replace with:

```python
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
```

Update `TestGetAvailableStrategies`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_brain_mask.py -v`
Expected: FAIL — `ImportError: cannot import name 'SynthSegBrainMask'`

**Step 3: Write minimal implementation**

Replace the contents of `SEEGFellow/SEEGFellow/SEEGFellowLib/brain_mask.py`. Remove `ScipyBrainMask` and `SynthStripBrainMask` entirely. Add `SynthSegBrainMask`:

```python
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

    def compute(self, volume: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """Run SynthSeg parcellation and return binarized brain mask.

        After calling this method, ``self.parcellation`` contains the full
        int32 parcellation array (Slicer K,J,I axis order) and
        ``self.parcellation_affine`` contains its 4×4 voxel-to-world matrix.

        Args:
            volume: 3-D numpy array (arbitrary intensity scale).
            affine: 4x4 voxel-to-world (IJK-to-RAS) transformation matrix.

        Returns:
            Binary uint8 mask (1 = brain, 0 = outside).

        Raises:
            RuntimeError: If mri_synthseg is not available or fails.
        """
        import nibabel as nib

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
                "--i", input_path,
                "--o", output_path,
                "--parc",
                "--cpu",
                "--threads", str(self.threads),
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
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_brain_mask.py -v`
Expected: All pass

**Step 5: Update `__init__.py`**

In `SEEGFellow/SEEGFellow/SEEGFellowLib/__init__.py`, replace imports:

```python
from SEEGFellowLib.brain_mask import (
    BrainMaskStrategy,
    SynthSegBrainMask,
    get_available_strategies,
)
```

**Step 6: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All pass

**Step 7: Commit**

```
git add SEEGFellow/SEEGFellow/SEEGFellowLib/brain_mask.py SEEGFellow/SEEGFellow/SEEGFellowLib/__init__.py tests/test_brain_mask.py
git commit -m "feat: replace SynthStrip/Scipy with SynthSegBrainMask

SynthSeg runs mri_synthseg --parc to produce a volumetric parcellation.
Brain mask is derived by binarizing the parcellation. The full
parcellation is stored for downstream contact labeling."
```

---

### Task 1.3: Update Logic and Widget for SynthSeg

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellow.py`
- Modify: `SEEGFellow/SEEGFellow/Resources/UI/SEEGFellow.ui`

**Step 1: Update the .ui file**

Replace the Step 3a intracranial mask section with the new Brain Segmentation panel. Add a FreeSurfer path browse widget, mode toggle (robust/fast), and thread count spinbox. Rename button from "Compute Intracranial Mask" to "Compute Segmentation".

In `SEEGFellow.ui`, replace the `<!-- Step 3a: Intracranial Mask -->` block:

```xml
   <!-- Step 3a: Brain Segmentation (SynthSeg) -->
   <item>
    <widget class="ctkCollapsibleButton" name="intracranialMaskCollapsibleButton">
     <property name="text"><string>Step 3a: Brain Segmentation</string></property>
     <property name="collapsed"><bool>true</bool></property>
     <layout class="QVBoxLayout" name="intracranialMaskLayout">
      <item>
       <layout class="QFormLayout" name="synthSegFormLayout">
        <item row="0" column="0"><widget class="QLabel"><property name="text"><string>FreeSurfer:</string></property></widget></item>
        <item row="0" column="1"><widget class="QLabel" name="freesurferStatusLabel"><property name="text"><string>Checking...</string></property></widget></item>
        <item row="1" column="0"><widget class="QLabel" name="freesurferPathLabel"><property name="text"><string>FreeSurfer path:</string></property><property name="visible"><bool>false</bool></property></widget></item>
        <item row="1" column="1"><widget class="ctkPathLineEdit" name="freesurferPathLineEdit"><property name="visible"><bool>false</bool></property><property name="filters"><enum>ctkPathLineEdit::Dirs</enum></property></widget></item>
        <item row="2" column="0"><widget class="QLabel"><property name="text"><string>Mode:</string></property></widget></item>
        <item row="2" column="1">
         <widget class="QComboBox" name="synthSegModeComboBox">
          <item><property name="text"><string>Robust (recommended)</string></property></item>
          <item><property name="text"><string>Fast</string></property></item>
         </widget>
        </item>
        <item row="3" column="0"><widget class="QLabel"><property name="text"><string>CPU threads:</string></property></widget></item>
        <item row="3" column="1"><widget class="QSpinBox" name="synthSegThreadsSpinBox"><property name="minimum"><number>1</number></property><property name="maximum"><number>16</number></property><property name="value"><number>1</number></property></widget></item>
       </layout>
      </item>
      <item><widget class="QPushButton" name="computeHeadMaskButton"><property name="text"><string>Compute Segmentation</string></property></widget></item>
      <item><widget class="QPushButton" name="editHeadMaskButton"><property name="text"><string>Edit in Segment Editor</string></property></widget></item>
     </layout>
    </widget>
   </item>
```

**Step 2: Update the Widget class**

In `SEEGFellow.py`, replace `_setup_brain_mask_combo` with `_setup_synthseg_ui`:

```python
    def _setup_synthseg_ui(self):
        """Check FreeSurfer availability and configure status label."""
        from SEEGFellowLib.brain_mask import SynthSegBrainMask

        strategy = SynthSegBrainMask()
        if strategy.is_available():
            self.ui.freesurferStatusLabel.setText("Found")
            self.ui.freesurferPathLabel.visible = False
            self.ui.freesurferPathLineEdit.visible = False
        else:
            self.ui.freesurferStatusLabel.setText("Not found — set path below")
            self.ui.freesurferPathLabel.visible = True
            self.ui.freesurferPathLineEdit.visible = True
            # Check Slicer settings for a saved path
            saved_path = slicer.app.settings().value("SEEGFellow/FreeSurferHome", "")
            if saved_path:
                self.ui.freesurferPathLineEdit.currentPath = saved_path
```

Update `setup()`: replace `self._setup_brain_mask_combo()` with `self._setup_synthseg_ui()`.

Update `_on_compute_head_mask_clicked`:

```python
    def _on_compute_head_mask_clicked(self):
        self._ensure_session_restored()

        # Resolve FreeSurfer path if set via the UI browse widget
        fs_path = self.ui.freesurferPathLineEdit.currentPath
        if fs_path:
            import os
            os.environ["FREESURFER_HOME"] = fs_path
            slicer.app.settings().setValue("SEEGFellow/FreeSurferHome", fs_path)

        from SEEGFellowLib.brain_mask import SynthSegBrainMask

        robust = self.ui.synthSegModeComboBox.currentIndex == 0
        threads = self.ui.synthSegThreadsSpinBox.value
        strategy = SynthSegBrainMask(robust=robust, threads=threads)

        if not strategy.is_available():
            slicer.util.errorDisplay(
                "FreeSurfer not found. Set FREESURFER_HOME or browse to the install directory."
            )
            return

        try:
            slicer.util.showStatusMessage("Running SynthSeg brain segmentation...")
            self.logic.run_intracranial_mask(strategy=strategy)
            slicer.util.showStatusMessage("Brain segmentation complete.")
            self.ui.metalThresholdCollapsibleButton.collapsed = False
            # Update status label
            self.ui.freesurferStatusLabel.setText("Found")
            self.ui.freesurferPathLabel.visible = False
            self.ui.freesurferPathLineEdit.visible = False
        except Exception as e:
            slicer.util.errorDisplay(f"SynthSeg failed: {e}")
```

**Step 3: Update `run_intracranial_mask` in Logic**

Modify `SEEGFellowLogic.run_intracranial_mask` to store parcellation data when the strategy is `SynthSegBrainMask`:

Add `self._parcellation = None` and `self._parcellation_affine = None` to `__init__`.

After `brain_mask_t1 = strategy.compute(t1_array, affine)`, add:

```python
        # Store parcellation for downstream contact labeling
        from SEEGFellowLib.brain_mask import SynthSegBrainMask
        if isinstance(strategy, SynthSegBrainMask) and strategy.parcellation is not None:
            self._parcellation = strategy.parcellation
            self._parcellation_affine = strategy.parcellation_affine
```

Also update the default strategy:

```python
        if strategy is None:
            from SEEGFellowLib.brain_mask import SynthSegBrainMask
            strategy = SynthSegBrainMask()
```

**Step 4: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All pass

**Step 5: Commit**

```
git add SEEGFellow/SEEGFellow/SEEGFellow.py SEEGFellow/SEEGFellow/Resources/UI/SEEGFellow.ui
git commit -m "feat: update UI and Logic for SynthSeg brain segmentation

Replace brain mask strategy combo box with SynthSeg mode/threads
controls and FreeSurfer path browser. Store parcellation for
contact labeling."
```

---

## Branch 2: `feat/contact-labeling`

### Task 2.1: Implement `contact_labeler.py` with LUT and labeling function

**Files:**
- Create: `SEEGFellow/SEEGFellow/SEEGFellowLib/contact_labeler.py`
- Create: `tests/test_contact_labeler.py`

**Step 1: Write the failing tests**

Create `tests/test_contact_labeler.py`:

```python
import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow")
)

import numpy as np
import pytest

from SEEGFellowLib.contact_labeler import SYNTHSEG_LUT, label_contacts


class TestSynthSegLUT:
    def test_contains_hippocampus(self):
        assert SYNTHSEG_LUT[17] == "Left Hippocampus"
        assert SYNTHSEG_LUT[53] == "Right Hippocampus"

    def test_contains_cortical_parcels(self):
        assert SYNTHSEG_LUT[1028] == "Left Superior Frontal"
        assert SYNTHSEG_LUT[2028] == "Right Superior Frontal"

    def test_contains_white_matter(self):
        assert SYNTHSEG_LUT[2] == "Left Cerebral WM"
        assert SYNTHSEG_LUT[41] == "Right Cerebral WM"

    def test_background_not_in_lut(self):
        assert 0 not in SYNTHSEG_LUT


class TestLabelContacts:
    def _make_parcellation(self, shape=(30, 30, 30)):
        """Return a zeroed parcellation array and identity affine."""
        return np.zeros(shape, dtype=np.int32), np.eye(4)

    def test_single_contact_in_hippocampus(self):
        parc, affine = self._make_parcellation()
        # Place hippocampus label at voxel (15, 15, 15)
        # Slicer convention: array indexed as [K, J, I]
        # With identity affine, RAS = (I, J, K) → so RAS (15,15,15) maps to IJK (15,15,15) → KJI (15,15,15)
        parc[15, 15, 15] = 17  # Left Hippocampus
        contacts_ras = np.array([[15.0, 15.0, 15.0]])
        labels = label_contacts(contacts_ras, parc, affine)
        assert labels == ["Left Hippocampus"]

    def test_contact_in_white_matter_with_nearby_cortex(self):
        """WM contact should report 'WM near <nearest cortical label>'."""
        parc, affine = self._make_parcellation()
        parc[15, 15, 15] = 2  # Left Cerebral WM
        parc[15, 15, 17] = 1028  # Left Superior Frontal, 2 voxels away
        contacts_ras = np.array([[15.0, 15.0, 15.0]])
        labels = label_contacts(contacts_ras, parc, affine)
        assert labels == ["WM near Left Superior Frontal"]

    def test_contact_in_white_matter_no_nearby_cortex(self):
        """WM contact with no cortex nearby should report just 'Left Cerebral WM'."""
        parc, affine = self._make_parcellation()
        parc[15, 15, 15] = 2  # Left Cerebral WM
        contacts_ras = np.array([[15.0, 15.0, 15.0]])
        labels = label_contacts(contacts_ras, parc, affine, search_radius_mm=1.0)
        assert labels == ["Left Cerebral WM"]

    def test_contact_outside_brain(self):
        parc, affine = self._make_parcellation()
        contacts_ras = np.array([[15.0, 15.0, 15.0]])
        labels = label_contacts(contacts_ras, parc, affine)
        assert labels == ["Outside brain"]

    def test_contact_outside_volume_bounds(self):
        parc, affine = self._make_parcellation()
        contacts_ras = np.array([[100.0, 100.0, 100.0]])
        labels = label_contacts(contacts_ras, parc, affine)
        assert labels == ["Outside brain"]

    def test_multiple_contacts(self):
        parc, affine = self._make_parcellation()
        parc[10, 10, 10] = 17  # Left Hippocampus
        parc[20, 20, 20] = 53  # Right Hippocampus
        contacts_ras = np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
        labels = label_contacts(contacts_ras, parc, affine)
        assert labels == ["Left Hippocampus", "Right Hippocampus"]

    def test_non_identity_affine(self):
        """With 2mm voxels, RAS (20,20,20) maps to voxel (10,10,10)."""
        parc, _ = self._make_parcellation()
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        parc[10, 10, 10] = 18  # Left Amygdala
        contacts_ras = np.array([[20.0, 20.0, 20.0]])
        labels = label_contacts(contacts_ras, parc, affine)
        assert labels == ["Left Amygdala"]
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_contact_labeler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'SEEGFellowLib.contact_labeler'`

**Step 3: Write minimal implementation**

Create `SEEGFellow/SEEGFellow/SEEGFellowLib/contact_labeler.py`:

```python
# SEEGFellow/SEEGFellow/SEEGFellowLib/contact_labeler.py
"""Map electrode contact positions to anatomical regions via SynthSeg parcellation.

Uses the FreeSurfer SynthSeg label lookup table (DKT atlas) to convert integer
parcellation labels into human-readable region names.
"""

from __future__ import annotations

import numpy as np

# FreeSurfer SynthSeg label → human-readable name.
# Base labels (subcortical + WM + ventricles):
_BASE_LABELS: dict[int, str] = {
    2: "Left Cerebral WM",
    3: "Left Cerebral Cortex",
    4: "Left Lateral Ventricle",
    5: "Left Inf Lat Ventricle",
    7: "Left Cerebellum WM",
    8: "Left Cerebellum Cortex",
    10: "Left Thalamus",
    11: "Left Caudate",
    12: "Left Putamen",
    13: "Left Pallidum",
    14: "3rd Ventricle",
    15: "4th Ventricle",
    16: "Brain Stem",
    17: "Left Hippocampus",
    18: "Left Amygdala",
    24: "CSF",
    26: "Left Accumbens",
    28: "Left Ventral DC",
    41: "Right Cerebral WM",
    42: "Right Cerebral Cortex",
    43: "Right Lateral Ventricle",
    44: "Right Inf Lat Ventricle",
    46: "Right Cerebellum WM",
    47: "Right Cerebellum Cortex",
    49: "Right Thalamus",
    50: "Right Caudate",
    51: "Right Putamen",
    52: "Right Pallidum",
    53: "Right Hippocampus",
    54: "Right Amygdala",
    58: "Right Accumbens",
    60: "Right Ventral DC",
}

# DKT cortical parcellation names (same for left 1000-series and right 2000-series).
_DKT_NAMES: dict[int, str] = {
    2: "Caudal Anterior Cingulate",
    3: "Caudal Middle Frontal",
    5: "Cuneus",
    6: "Entorhinal",
    7: "Fusiform",
    8: "Inferior Parietal",
    9: "Inferior Temporal",
    10: "Isthmus Cingulate",
    11: "Lateral Occipital",
    12: "Lateral Orbitofrontal",
    13: "Lingual",
    14: "Medial Orbitofrontal",
    15: "Middle Temporal",
    16: "Parahippocampal",
    17: "Paracentral",
    18: "Pars Opercularis",
    19: "Pars Orbitalis",
    20: "Pars Triangularis",
    21: "Pericalcarine",
    22: "Postcentral",
    23: "Posterior Cingulate",
    24: "Precentral",
    25: "Precuneus",
    26: "Rostral Anterior Cingulate",
    27: "Rostral Middle Frontal",
    28: "Superior Frontal",
    29: "Superior Parietal",
    30: "Superior Temporal",
    31: "Supramarginal",
    34: "Transverse Temporal",
    35: "Insula",
}

# Build the combined LUT.
SYNTHSEG_LUT: dict[int, str] = dict(_BASE_LABELS)
for _offset, _name in _DKT_NAMES.items():
    SYNTHSEG_LUT[1000 + _offset] = f"Left {_name}"
    SYNTHSEG_LUT[2000 + _offset] = f"Right {_name}"

# Labels that represent cerebral white matter (trigger nearest-cortex search).
_WM_LABELS = {2, 41}

# Labels in the 1000–1035 and 2000–2035 range are cortical parcels.
_CORTICAL_LABEL_RANGES = (range(1000, 1036), range(2000, 2036))


def _is_cortical_parcel(label: int) -> bool:
    return any(label in r for r in _CORTICAL_LABEL_RANGES)


def label_contacts(
    contacts_ras: np.ndarray,
    parcellation: np.ndarray,
    parcellation_affine: np.ndarray,
    search_radius_mm: float = 3.0,
) -> list[str]:
    """Map contact RAS positions to anatomical region names.

    Args:
        contacts_ras: (N, 3) array of contact positions in RAS world space.
        parcellation: 3-D int32 array of SynthSeg labels (Slicer K,J,I order).
        parcellation_affine: 4×4 voxel-to-world matrix of the parcellation
            volume (NIfTI convention: maps I,J,K → R,A,S).
        search_radius_mm: Radius for nearest-cortical-label search when a
            contact lands in white matter.

    Returns:
        List of region name strings, one per contact.

    Example::

        labels = label_contacts(
            np.array([[10.0, 20.0, 30.0]]),
            parcellation_array,
            parcellation_affine,
        )
    """
    inv_affine = np.linalg.inv(parcellation_affine)
    voxel_sizes = np.sqrt((parcellation_affine[:3, :3] ** 2).sum(axis=0))
    search_radius_vox = search_radius_mm / float(voxel_sizes.min())

    results: list[str] = []
    for ras in contacts_ras:
        # RAS → IJK (NIfTI convention)
        ijk_h = inv_affine @ np.append(ras, 1.0)
        ijk = np.round(ijk_h[:3]).astype(int)

        # Parcellation is stored in Slicer (K,J,I) order, so index as [K,J,I]
        k, j, i = ijk[2], ijk[1], ijk[0]

        # Bounds check
        if (
            k < 0 or k >= parcellation.shape[0]
            or j < 0 or j >= parcellation.shape[1]
            or i < 0 or i >= parcellation.shape[2]
        ):
            results.append("Outside brain")
            continue

        label = int(parcellation[k, j, i])

        if label == 0:
            results.append("Outside brain")
        elif label in _WM_LABELS:
            nearest = _nearest_cortical_label(parcellation, k, j, i, search_radius_vox)
            if nearest is not None:
                results.append(f"WM near {SYNTHSEG_LUT.get(nearest, f'Label {nearest}')}")
            else:
                results.append(SYNTHSEG_LUT.get(label, f"Label {label}"))
        else:
            results.append(SYNTHSEG_LUT.get(label, f"Label {label}"))

    return results


def _nearest_cortical_label(
    parcellation: np.ndarray,
    k: int,
    j: int,
    i: int,
    radius_vox: float,
) -> int | None:
    """Search a sphere around (k,j,i) for the nearest cortical parcel label.

    Returns the label integer, or None if no cortical parcel is found within radius.
    """
    r = int(np.ceil(radius_vox))
    k_lo, k_hi = max(0, k - r), min(parcellation.shape[0], k + r + 1)
    j_lo, j_hi = max(0, j - r), min(parcellation.shape[1], j + r + 1)
    i_lo, i_hi = max(0, i - r), min(parcellation.shape[2], i + r + 1)

    patch = parcellation[k_lo:k_hi, j_lo:j_hi, i_lo:i_hi]
    coords = np.argwhere(patch)  # (N, 3) in patch-local coordinates

    best_label: int | None = None
    best_dist = float("inf")

    center = np.array([k - k_lo, j - j_lo, i - i_lo], dtype=float)
    for idx in range(len(coords)):
        lbl = int(patch[coords[idx, 0], coords[idx, 1], coords[idx, 2]])
        if not _is_cortical_parcel(lbl):
            continue
        dist = float(np.linalg.norm(coords[idx] - center))
        if dist <= radius_vox and dist < best_dist:
            best_dist = dist
            best_label = lbl

    return best_label
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_contact_labeler.py -v`
Expected: All pass

**Step 5: Update `__init__.py`**

Add to `SEEGFellow/SEEGFellow/SEEGFellowLib/__init__.py`:

```python
from SEEGFellowLib.contact_labeler import SYNTHSEG_LUT, label_contacts
```

**Step 6: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All pass

**Step 7: Commit**

```
git add SEEGFellow/SEEGFellow/SEEGFellowLib/contact_labeler.py SEEGFellow/SEEGFellow/SEEGFellowLib/__init__.py tests/test_contact_labeler.py
git commit -m "feat: add contact_labeler module with SynthSeg LUT and labeling

Maps contact RAS positions to anatomical region names using SynthSeg
parcellation labels. Includes WM fallback (nearest cortical label
search within a configurable radius)."
```

---

### Task 2.2: Add contact labeling step to Logic and UI

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellow.py`
- Modify: `SEEGFellow/SEEGFellow/Resources/UI/SEEGFellow.ui`

**Step 1: Add the UI panel**

In `SEEGFellow.ui`, add a new collapsible button **after** the Rename Electrodes section and **before** the Results & Export section:

```xml
   <!-- Step 5: Label Contacts -->
   <item>
    <widget class="ctkCollapsibleButton" name="labelContactsCollapsibleButton">
     <property name="text"><string>Step 5: Label Contacts</string></property>
     <property name="collapsed"><bool>true</bool></property>
     <layout class="QVBoxLayout" name="labelContactsLayout">
      <item><widget class="QPushButton" name="labelContactsButton"><property name="text"><string>Label All Contacts</string></property></widget></item>
      <item><widget class="QTableWidget" name="anatomyTable"/></item>
     </layout>
    </widget>
   </item>
```

**Step 2: Add Logic method `run_contact_labeling`**

In `SEEGFellowLogic`, add:

```python
    def run_contact_labeling(self) -> None:
        """Label each contact with its anatomical region from the SynthSeg parcellation.

        Requires parcellation (from run_intracranial_mask with SynthSeg) and
        detected electrodes.

        Example::

            logic.run_contact_labeling()
            for e in logic.electrodes:
                for c in e.contacts:
                    print(c.label, c.region)
        """
        import numpy as np
        from SEEGFellowLib.contact_labeler import label_contacts

        if self._parcellation is None:
            raise RuntimeError(
                "No parcellation available. Run brain segmentation (SynthSeg) first."
            )
        if not self.electrodes:
            raise RuntimeError("No electrodes detected. Run electrode detection first.")

        for electrode in self.electrodes:
            contacts_ras = np.array([c.position_ras for c in electrode.contacts])
            regions = label_contacts(
                contacts_ras, self._parcellation, self._parcellation_affine
            )
            for contact, region in zip(electrode.contacts, regions):
                contact.region = region
```

**Step 3: Wire up the Widget**

In `SEEGFellowWidget.setup()`, add connection:

```python
        # Step 5: Label Contacts
        self.ui.labelContactsButton.clicked.connect(self._on_label_contacts_clicked)
```

Add handler and anatomy table methods:

```python
    def _on_label_contacts_clicked(self):
        self._ensure_session_restored()
        try:
            slicer.util.showStatusMessage("Labeling contacts...")
            self.logic.run_contact_labeling()
            self._populate_anatomy_table()
            slicer.util.showStatusMessage("Contact labeling complete.")
            self.ui.exportCollapsibleButton.collapsed = False
        except Exception as e:
            slicer.util.errorDisplay(f"Contact labeling failed: {e}")

    def _populate_anatomy_table(self):
        from qt import QTableWidgetItem

        electrodes = self.logic.electrodes
        if not electrodes:
            return

        max_contacts = max(e.num_contacts for e in electrodes)
        sorted_electrodes = sorted(electrodes, key=lambda e: e.name)

        table = self.ui.anatomyTable
        table.setRowCount(len(sorted_electrodes))
        table.setColumnCount(max_contacts)
        table.setHorizontalHeaderLabels([str(i + 1) for i in range(max_contacts)])
        table.setVerticalHeaderLabels([e.name for e in sorted_electrodes])

        for row, electrode in enumerate(sorted_electrodes):
            for col, contact in enumerate(electrode.contacts):
                table.setItem(row, col, QTableWidgetItem(contact.region))

        table.resizeColumnsToContents()
```

**Step 4: Update `_on_apply_names_clicked` to uncollapse Step 5**

Change `self.ui.exportCollapsibleButton.collapsed = False` to `self.ui.labelContactsCollapsibleButton.collapsed = False` in `_on_apply_names_clicked`.

**Step 5: Update CSV export to include Region column**

In `SEEGFellowLogic.export_csv`:

```python
    def export_csv(self, path: str) -> None:
        """Export all contact positions and regions to a CSV file.

        Example::

            logic.export_csv("/output/contacts.csv")
        """
        import csv

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Electrode", "Contact", "R", "A", "S", "Region"])
            for electrode in self.electrodes:
                for contact in electrode.contacts:
                    r, a, s = contact.position_ras
                    writer.writerow(
                        [electrode.name, contact.label, r, a, s, contact.region]
                    )
```

Update `_populate_contact_table` to add the Region column:

```python
    def _populate_contact_table(self):
        from qt import QTableWidgetItem

        self.ui.contactTable.setColumnCount(6)
        self.ui.contactTable.setHorizontalHeaderLabels(
            ["Electrode", "Contact", "R", "A", "S", "Region"]
        )
        self.ui.contactTable.horizontalHeader().setStretchLastSection(True)

        rows = []
        for electrode in self.logic.electrodes:
            for contact in electrode.contacts:
                r, a, s = contact.position_ras
                rows.append(
                    (electrode.name, contact.label, f"{r:.2f}", f"{a:.2f}", f"{s:.2f}", contact.region)
                )

        self.ui.contactTable.setRowCount(len(rows))
        for row_idx, row_data in enumerate(rows):
            for col_idx, value in enumerate(row_data):
                self.ui.contactTable.setItem(row_idx, col_idx, QTableWidgetItem(value))
```

**Step 6: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All pass

**Step 7: Commit**

```
git add SEEGFellow/SEEGFellow/SEEGFellow.py SEEGFellow/SEEGFellow/Resources/UI/SEEGFellow.ui
git commit -m "feat: add contact labeling step with anatomy table and CSV region column

New Step 5 runs contact labeling via SynthSeg parcellation. Anatomy
table shows electrode × contact grid with region names. CSV export
includes Region column."
```

---

### Task 2.3: Session restore for parcellation

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellow.py` (Logic class)

**Step 1: Store parcellation as a scene node**

In `run_intracranial_mask`, after storing `self._parcellation`, also save it as a label map node so it persists with the Slicer scene:

```python
        # Save parcellation as a label map node for scene persistence
        if self._parcellation is not None:
            parc_node = slicer.util.getFirstNodeByClassByName(
                "vtkMRMLLabelMapVolumeNode", "_SEEGFellow_SynthSeg_Parcellation"
            )
            if parc_node is None:
                parc_node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLLabelMapVolumeNode", "_SEEGFellow_SynthSeg_Parcellation"
                )
            # Set geometry from parcellation affine
            import vtk
            mat = vtk.vtkMatrix4x4()
            for r_idx in range(4):
                for c_idx in range(4):
                    mat.SetElement(r_idx, c_idx, float(self._parcellation_affine[r_idx, c_idx]))
            parc_node.SetIJKToRASMatrix(mat)
            # Store the parcellation (transpose from Slicer K,J,I back to NIfTI I,J,K
            # for the label map node, then Slicer handles it internally)
            updateVolumeFromArray(parc_node, self._parcellation)
            parc_node.SetHideFromEditors(True)
```

**Step 2: Restore parcellation in `try_restore_from_scene`**

After the metal mask restore block, add:

```python
        # Restore parcellation if saved
        parc_node = slicer.util.getFirstNodeByClassByName(
            "vtkMRMLLabelMapVolumeNode", "_SEEGFellow_SynthSeg_Parcellation"
        )
        if parc_node is not None:
            import vtk
            self._parcellation = np.array(
                slicer.util.arrayFromVolume(parc_node), dtype=np.int32
            )
            mat = vtk.vtkMatrix4x4()
            parc_node.GetIJKToRASMatrix(mat)
            self._parcellation_affine = np.array(
                [[mat.GetElement(r, c) for c in range(4)] for r in range(4)]
            )
```

**Step 3: Update session restore UI in Widget**

In `_try_restore_session`, extend to check for parcellation and uncollapse accordingly:

```python
        has_parcellation = self.logic._parcellation is not None
        has_electrodes = len(self.logic.electrodes) > 0
```

**Step 4: Commit**

```
git add SEEGFellow/SEEGFellow/SEEGFellow.py
git commit -m "feat: persist and restore SynthSeg parcellation across sessions

Save parcellation as a hidden label map node so it survives scene
save/load. Restore in try_restore_from_scene for contact labeling
without re-running SynthSeg."
```

---

## Summary

| Branch | Tasks | What it does |
|--------|-------|-------------|
| `feat/synthseg-brain-mask` | 1.1–1.3 | Replace SynthStrip/Scipy with SynthSeg, add `region` field to Contact, update UI |
| `feat/contact-labeling` | 2.1–2.3 | Add contact_labeler module, labeling UI step, anatomy table, CSV region column, session restore |
