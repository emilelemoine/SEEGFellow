# Modular Brain Mask — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the monolithic `compute_brain_mask()` with a strategy pattern supporting scipy (fast/rough) and SynthStrip CLI (high-quality) brain extraction, selectable via a UI dropdown.

**Architecture:** A new `brain_mask.py` module defines a `BrainMaskStrategy` protocol with two implementations. The UI populates a combo box from available strategies. `run_intracranial_mask()` delegates to the selected strategy. DeepBet is removed entirely.

**Tech Stack:** Python 3.9+, scipy, nibabel, FreeSurfer `mri_synthstrip` CLI, 3D Slicer Qt UI

---

### Task 1: Create `brain_mask.py` with strategy protocol and scipy implementation

Move the existing scipy brain mask logic out of `metal_segmenter.py` into a new dedicated module with the strategy protocol.

**Files:**
- Create: `SEEGFellow/SEEGFellow/SEEGFellowLib/brain_mask.py`
- Test: `tests/test_brain_mask.py`

**Step 1: Write failing tests for `ScipyBrainMask`**

Create `tests/test_brain_mask.py`:

```python
# tests/test_brain_mask.py
import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow")
)

import pytest
import numpy as np
from SEEGFellowLib.brain_mask import ScipyBrainMask


class TestScipyBrainMask:
    def test_name(self):
        strategy = ScipyBrainMask()
        assert strategy.name == "Scipy (morphological)"

    def test_is_available(self):
        strategy = ScipyBrainMask()
        assert strategy.is_available() is True

    def test_returns_binary_mask(self):
        """Should return a non-empty binary uint8 mask for a head-like volume."""
        volume = np.zeros((30, 30, 30), dtype=np.float32)
        center = np.array([15, 15, 15])
        coords = np.indices((30, 30, 30)).transpose(1, 2, 3, 0)
        volume[np.linalg.norm(coords - center, axis=-1) < 10] = 1000.0
        affine = np.eye(4)

        strategy = ScipyBrainMask()
        result = strategy.compute(volume, affine)

        assert result.shape == (30, 30, 30)
        assert result.dtype == np.uint8
        assert np.any(result)

    def test_empty_volume_raises(self):
        """Should raise RuntimeError if the volume produces no foreground."""
        volume = np.zeros((20, 20, 20), dtype=np.float32)
        affine = np.eye(4)

        strategy = ScipyBrainMask()
        with pytest.raises(RuntimeError, match="empty"):
            strategy.compute(volume, affine)
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_brain_mask.py -v`
Expected: FAIL with `ModuleNotFoundError` (brain_mask doesn't exist yet)

**Step 3: Implement `brain_mask.py`**

Create `SEEGFellow/SEEGFellow/SEEGFellowLib/brain_mask.py`:

```python
# SEEGFellow/SEEGFellow/SEEGFellowLib/brain_mask.py
"""Brain mask strategies for intracranial segmentation.

Each strategy implements the same interface: compute(volume, affine) -> mask.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy import ndimage


class BrainMaskStrategy(Protocol):
    """Interface for brain mask extraction algorithms."""

    name: str

    def compute(self, volume: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """Return a binary uint8 mask (1 = brain, 0 = outside).

        Raises RuntimeError if the resulting mask is empty.
        """
        ...

    def is_available(self) -> bool:
        """Return True if this strategy can run on the current system."""
        ...


class ScipyBrainMask:
    """Brain extraction using morphological operations.

    Quality is lower than CNN-based methods but sufficient as an initial mask
    that the user can refine in the Segment Editor.

    Example::

        strategy = ScipyBrainMask()
        mask = strategy.compute(t1_array, affine)
    """

    name: str = "Scipy (morphological)"

    def is_available(self) -> bool:
        return True

    def compute(self, volume: np.ndarray, affine: np.ndarray) -> np.ndarray:
        voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        min_voxel_mm = float(np.clip(voxel_sizes.min(), 0.1, None))

        foreground = volume > volume.max() * 0.05
        filled = ndimage.binary_fill_holes(
            ndimage.binary_closing(foreground, iterations=2)
        )

        labeled, n = ndimage.label(filled)
        if n == 0:
            raise RuntimeError(
                "Brain mask is empty – brain extraction produced no output."
            )
        sizes = ndimage.sum(filled, labeled, range(1, n + 1))
        head = labeled == (int(np.argmax(sizes)) + 1)

        erosion_voxels = max(1, int(round(5.0 / min_voxel_mm)))
        brain = ndimage.binary_erosion(head, iterations=erosion_voxels)

        mask = brain.astype(np.uint8)
        if not np.any(mask):
            raise RuntimeError(
                "Brain mask is empty – brain extraction produced no output."
            )
        return mask


def get_available_strategies() -> list[BrainMaskStrategy]:
    """Return all brain mask strategies, available ones first.

    Example::

        strategies = get_available_strategies()
        names = [s.name for s in strategies]
    """
    all_strategies: list[BrainMaskStrategy] = [ScipyBrainMask()]
    # SynthStripBrainMask will be added in Task 2
    return [s for s in all_strategies if s.is_available()] + [
        s for s in all_strategies if not s.is_available()
    ]
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_brain_mask.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/brain_mask.py tests/test_brain_mask.py
git commit -m "feat: add brain mask strategy protocol with scipy implementation"
```

---

### Task 2: Add SynthStrip CLI strategy

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/brain_mask.py`
- Test: `tests/test_brain_mask.py`

**Step 1: Write failing tests for `SynthStripBrainMask`**

Append to `tests/test_brain_mask.py`:

```python
from unittest.mock import patch, MagicMock
from SEEGFellowLib.brain_mask import SynthStripBrainMask, get_available_strategies


class TestSynthStripBrainMask:
    def test_name(self):
        strategy = SynthStripBrainMask()
        assert strategy.name == "SynthStrip (FreeSurfer)"

    def test_is_available_when_on_path(self, tmp_path):
        """Should return True when mri_synthstrip is found."""
        with patch("shutil.which", return_value="/usr/local/bin/mri_synthstrip"):
            assert SynthStripBrainMask().is_available() is True

    def test_is_available_when_missing(self):
        """Should return False when mri_synthstrip is not found."""
        with patch("shutil.which", return_value=None):
            with patch.dict(os.environ, {}, clear=True):
                assert SynthStripBrainMask().is_available() is False

    def test_is_available_via_freesurfer_home(self, tmp_path):
        """Should find mri_synthstrip under FREESURFER_HOME/bin/."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        exe = bin_dir / "mri_synthstrip"
        exe.touch()
        exe.chmod(0o755)
        with patch("shutil.which", return_value=None):
            with patch.dict(os.environ, {"FREESURFER_HOME": str(tmp_path)}):
                assert SynthStripBrainMask().is_available() is True

    def test_compute_calls_subprocess(self, tmp_path):
        """Should write NIfTI, call mri_synthstrip, read back mask."""
        volume = np.ones((10, 10, 10), dtype=np.float32)
        affine = np.eye(4)

        fake_mask = np.ones((10, 10, 10), dtype=np.uint8)

        def fake_run(cmd, **kwargs):
            import nibabel as nib
            # Write a fake mask to the output path (the -m argument)
            mask_path = cmd[cmd.index("-m") + 1]
            nib.save(nib.Nifti1Image(fake_mask, affine), mask_path)
            return MagicMock(returncode=0)

        with patch("shutil.which", return_value="/usr/bin/mri_synthstrip"):
            with patch("subprocess.run", side_effect=fake_run):
                strategy = SynthStripBrainMask()
                result = strategy.compute(volume, affine)

        assert result.shape == (10, 10, 10)
        assert result.dtype == np.uint8
        assert np.all(result == 1)

    def test_compute_raises_on_empty_mask(self, tmp_path):
        """Should raise RuntimeError if SynthStrip produces an empty mask."""
        volume = np.ones((10, 10, 10), dtype=np.float32)
        affine = np.eye(4)

        def fake_run(cmd, **kwargs):
            import nibabel as nib
            mask_path = cmd[cmd.index("-m") + 1]
            nib.save(
                nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.uint8), affine),
                mask_path,
            )
            return MagicMock(returncode=0)

        with patch("shutil.which", return_value="/usr/bin/mri_synthstrip"):
            with patch("subprocess.run", side_effect=fake_run):
                strategy = SynthStripBrainMask()
                with pytest.raises(RuntimeError, match="empty"):
                    strategy.compute(volume, affine)

    def test_compute_raises_on_subprocess_failure(self):
        """Should raise RuntimeError if mri_synthstrip exits non-zero."""
        volume = np.ones((10, 10, 10), dtype=np.float32)
        affine = np.eye(4)

        with patch("shutil.which", return_value="/usr/bin/mri_synthstrip"):
            with patch(
                "subprocess.run",
                return_value=MagicMock(returncode=1, stderr="segfault"),
            ):
                strategy = SynthStripBrainMask()
                with pytest.raises(RuntimeError, match="mri_synthstrip failed"):
                    strategy.compute(volume, affine)


class TestGetAvailableStrategies:
    def test_scipy_always_present(self):
        strategies = get_available_strategies()
        names = [s.name for s in strategies]
        assert "Scipy (morphological)" in names

    def test_available_strategies_come_first(self):
        strategies = get_available_strategies()
        available = [s.is_available() for s in strategies]
        # All True values should come before all False values
        assert available == sorted(available, reverse=True)
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_brain_mask.py -v`
Expected: FAIL with `ImportError` (SynthStripBrainMask doesn't exist)

**Step 3: Implement `SynthStripBrainMask`**

Add to `SEEGFellow/SEEGFellow/SEEGFellowLib/brain_mask.py`:

```python
import os
import shutil
import subprocess
import tempfile


class SynthStripBrainMask:
    """Brain extraction using FreeSurfer's SynthStrip CNN via CLI.

    Requires ``mri_synthstrip`` on PATH or under ``$FREESURFER_HOME/bin/``.

    Example::

        strategy = SynthStripBrainMask()
        if strategy.is_available():
            mask = strategy.compute(t1_array, affine)
    """

    name: str = "SynthStrip (FreeSurfer)"

    def _find_executable(self) -> str | None:
        exe = shutil.which("mri_synthstrip")
        if exe is not None:
            return exe
        fs_home = os.environ.get("FREESURFER_HOME")
        if fs_home:
            candidate = os.path.join(fs_home, "bin", "mri_synthstrip")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        return None

    def is_available(self) -> bool:
        return self._find_executable() is not None

    def compute(self, volume: np.ndarray, affine: np.ndarray) -> np.ndarray:
        import nibabel as nib

        exe = self._find_executable()
        if exe is None:
            raise RuntimeError(
                "mri_synthstrip not found. Install FreeSurfer or add it to PATH."
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "input.nii.gz")
            mask_path = os.path.join(tmp_dir, "mask.nii.gz")

            nib.save(nib.Nifti1Image(volume, affine), input_path)

            result = subprocess.run(
                [exe, "-i", input_path, "-m", mask_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"mri_synthstrip failed (exit {result.returncode}): "
                    f"{result.stderr}"
                )

            mask_img = nib.load(mask_path)
            mask = np.asarray(mask_img.dataobj, dtype=np.uint8)

        mask = (mask > 0).astype(np.uint8)
        if not np.any(mask):
            raise RuntimeError(
                "Brain mask is empty – SynthStrip produced no output."
            )
        return mask
```

Also update `get_available_strategies()` to include `SynthStripBrainMask`:

```python
def get_available_strategies() -> list[BrainMaskStrategy]:
    all_strategies: list[BrainMaskStrategy] = [
        SynthStripBrainMask(),
        ScipyBrainMask(),
    ]
    return [s for s in all_strategies if s.is_available()] + [
        s for s in all_strategies if not s.is_available()
    ]
```

Note: `SynthStripBrainMask` is listed first so it's the default when available (higher quality).

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_brain_mask.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/brain_mask.py tests/test_brain_mask.py
git commit -m "feat: add SynthStrip CLI brain mask strategy"
```

---

### Task 3: Remove brain mask code from `metal_segmenter.py` and update imports

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/metal_segmenter.py` — remove `compute_brain_mask`, `_compute_brain_mask_scipy`
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/__init__.py` — export from `brain_mask` instead
- Modify: `tests/test_metal_segmenter.py` — remove `TestComputeBrainMask` class and its import
- Modify: `SEEGFellow/SEEGFellow/SEEGFellow.py` — update import in `run_intracranial_mask()`

**Step 1: Remove `_compute_brain_mask_scipy` and `compute_brain_mask` from `metal_segmenter.py`**

Delete lines 24–87 of `metal_segmenter.py` (the two functions).

**Step 2: Update `__init__.py`**

Replace:
```python
from SEEGFellowLib.metal_segmenter import (
    compute_head_mask,
    threshold_volume,
    detect_contact_centers,
)
```

With:
```python
from SEEGFellowLib.metal_segmenter import (
    compute_head_mask,
    threshold_volume,
    detect_contact_centers,
)
from SEEGFellowLib.brain_mask import (
    BrainMaskStrategy,
    ScipyBrainMask,
    SynthStripBrainMask,
    get_available_strategies,
)
```

**Step 3: Update `SEEGFellow.py` import in `run_intracranial_mask()`**

In `SEEGFellow.py` line 558, change:
```python
from SEEGFellowLib.metal_segmenter import compute_brain_mask
```
to:
```python
from SEEGFellowLib.brain_mask import ScipyBrainMask
```

And on line 574, change:
```python
brain_mask_t1 = compute_brain_mask(t1_array, affine)
```
to:
```python
strategy = ScipyBrainMask()  # will be replaced by UI selection in Task 4
brain_mask_t1 = strategy.compute(t1_array, affine)
```

**Step 4: Remove brain mask tests from `test_metal_segmenter.py`**

Delete the `TestComputeBrainMask` class (lines 103–124) and remove the import `from SEEGFellowLib.metal_segmenter import compute_brain_mask` (line 14).

**Step 5: Run all tests**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests PASS (brain mask tests now in `test_brain_mask.py`, metal segmenter tests unchanged minus the removed class)

**Step 6: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/metal_segmenter.py \
       SEEGFellow/SEEGFellow/SEEGFellowLib/__init__.py \
       SEEGFellow/SEEGFellow/SEEGFellow.py \
       tests/test_metal_segmenter.py
git commit -m "refactor: move brain mask to dedicated module, remove deepbet"
```

---

### Task 4: Add method combo box to the UI and wire it up

**Files:**
- Modify: `SEEGFellow/SEEGFellow/Resources/UI/SEEGFellow.ui` — add combo box
- Modify: `SEEGFellow/SEEGFellow/SEEGFellow.py` — populate combo box, pass strategy to logic

**Step 1: Add combo box to `.ui` file**

In `SEEGFellow.ui`, inside the `intracranialMaskLayout` (between the opening `<layout>` tag and the `computeHeadMaskButton` item), add:

```xml
<item>
 <layout class="QHBoxLayout">
  <item><widget class="QLabel"><property name="text"><string>Method:</string></property></widget></item>
  <item><widget class="QComboBox" name="brainMaskMethodComboBox"/></item>
 </layout>
</item>
```

**Step 2: Populate combo box in `SEEGFellowWidget.setup()`**

After the line connecting `computeHeadMaskButton`, add:

```python
from SEEGFellowLib.brain_mask import get_available_strategies
self._brain_mask_strategies = get_available_strategies()
for strategy in self._brain_mask_strategies:
    suffix = "" if strategy.is_available() else " (unavailable)"
    self.ui.brainMaskMethodComboBox.addItem(strategy.name + suffix)
# Disable unavailable entries
for i, strategy in enumerate(self._brain_mask_strategies):
    if not strategy.is_available():
        model = self.ui.brainMaskMethodComboBox.model()
        item = model.item(i)
        item.setEnabled(False)
```

**Step 3: Pass selected strategy to logic**

In `_on_compute_head_mask_clicked()`, change:

```python
self.logic.run_intracranial_mask()
```

to:

```python
idx = self.ui.brainMaskMethodComboBox.currentIndex()
strategy = self._brain_mask_strategies[idx]
self.logic.run_intracranial_mask(strategy=strategy)
```

**Step 4: Update `run_intracranial_mask()` to accept a strategy parameter**

Change the method signature from:
```python
def run_intracranial_mask(self) -> None:
```
to:
```python
def run_intracranial_mask(self, strategy: BrainMaskStrategy | None = None) -> None:
```

And replace the body lines:
```python
from SEEGFellowLib.brain_mask import ScipyBrainMask
...
strategy = ScipyBrainMask()
brain_mask_t1 = strategy.compute(t1_array, affine)
```

with:
```python
from SEEGFellowLib.brain_mask import ScipyBrainMask

if strategy is None:
    strategy = ScipyBrainMask()
brain_mask_t1 = strategy.compute(t1_array, affine)
```

Also update the `BrainMaskStrategy` import. Add at line 557 (lazy import section):
```python
from SEEGFellowLib.brain_mask import BrainMaskStrategy, ScipyBrainMask
```

**Step 5: Run all tests**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add SEEGFellow/SEEGFellow/Resources/UI/SEEGFellow.ui \
       SEEGFellow/SEEGFellow/SEEGFellow.py
git commit -m "feat: add brain mask method dropdown to UI"
```

---

### Task 5: Clean up dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update `pyproject.toml`**

Replace:
```toml
dependencies = [
    "deepbet",
    "nibabel",
]
```

with:
```toml
dependencies = [
    "nibabel",
]
```

**Step 2: Sync the dev environment**

Run: `uv sync`

**Step 3: Run all tests one final time**

Run: `.venv/bin/pytest tests/ -v`
Expected: All 30+ tests PASS

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: remove deepbet dependency, keep nibabel for SynthStrip I/O"
```
