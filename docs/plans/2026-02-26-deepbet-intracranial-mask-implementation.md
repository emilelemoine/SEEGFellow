# deepbet Intracranial Mask Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the broken hand-rolled brain mask with deepbet CNN skull stripping.

**Architecture:** `compute_brain_mask` saves the T1 array as a temp NIfTI, calls `deepbet.run_bet()`, loads the resulting mask, and returns it as a numpy array. The Slicer-side caller passes the IJK-to-RAS affine matrix.

**Tech Stack:** deepbet, nibabel, numpy, pytest

---

### Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add deepbet and nibabel to pyproject.toml**

```toml
[project]
name = "seegfellow-dev"
version = "0.0.0"
requires-python = ">=3.9"
dependencies = [
    "deepbet",
    "nibabel",
]
```

**Step 2: Install dependencies**

Run: `uv sync`

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add deepbet and nibabel for brain extraction"
```

---

### Task 2: Write failing test for compute_brain_mask

**Files:**
- Modify: `tests/test_metal_segmenter.py`

**Step 1: Write the failing test**

Add to `tests/test_metal_segmenter.py`:

```python
from unittest.mock import patch, MagicMock
from SEEGFellowLib.metal_segmenter import compute_brain_mask


class TestComputeBrainMask:
    def test_returns_binary_mask_from_deepbet(self, tmp_path):
        """compute_brain_mask should call deepbet and return a binary mask."""
        volume = np.random.rand(10, 20, 30).astype(np.float32)
        affine = np.eye(4)

        # Mock deepbet.run_bet to write a mask file filled with ones
        def fake_run_bet(input_paths, brain_paths, mask_paths, tiv_paths,
                         threshold, n_dilate, no_gpu):
            import nibabel as nib
            mask = np.ones((10, 20, 30), dtype=np.uint8)
            nib.save(nib.Nifti1Image(mask, affine), mask_paths[0])

        with patch("SEEGFellowLib.metal_segmenter.run_bet", fake_run_bet):
            result = compute_brain_mask(volume, affine)

        assert result.shape == (10, 20, 30)
        assert result.dtype == np.uint8
        assert np.all(result == 1)

    def test_empty_mask_raises(self, tmp_path):
        """compute_brain_mask should raise if deepbet produces an empty mask."""
        volume = np.random.rand(10, 20, 30).astype(np.float32)
        affine = np.eye(4)

        def fake_run_bet(input_paths, brain_paths, mask_paths, tiv_paths,
                         threshold, n_dilate, no_gpu):
            import nibabel as nib
            mask = np.zeros((10, 20, 30), dtype=np.uint8)
            nib.save(nib.Nifti1Image(mask, affine), mask_paths[0])

        with patch("SEEGFellowLib.metal_segmenter.run_bet", fake_run_bet):
            import pytest as pt
            with pt.raises(RuntimeError, match="empty"):
                compute_brain_mask(volume, affine)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_metal_segmenter.py::TestComputeBrainMask -v`
Expected: FAIL (signature mismatch or import error since `run_bet` not imported yet)

**Step 3: Commit**

```bash
git add tests/test_metal_segmenter.py
git commit -m "test: add failing tests for deepbet-based compute_brain_mask"
```

---

### Task 3: Implement compute_brain_mask with deepbet

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/metal_segmenter.py`

**Step 1: Replace compute_brain_mask and remove unused helpers**

Replace the entire `compute_brain_mask` function, and remove `_otsu_threshold`,
`_largest_connected_component`, and `_spherical_structuring_element`:

```python
def compute_brain_mask(
    volume: np.ndarray,
    affine: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Create a binary mask of brain parenchyma from a T1-weighted MRI.

    Uses deepbet (CNN-based skull stripping) for robust brain extraction.
    The volume is saved to a temporary NIfTI file, processed by deepbet,
    and the resulting mask is loaded back as a numpy array.

    Args:
        volume: T1 MRI array (3-D numpy, arbitrary intensity scale).
        affine: 4x4 voxel-to-world (IJK-to-RAS) transformation matrix.
        threshold: deepbet segmentation threshold (0-1, default 0.5).

    Returns:
        Binary uint8 mask (1 = brain parenchyma).

    Raises:
        RuntimeError: If the resulting brain mask is empty.

    Example::

        brain = compute_brain_mask(t1_array, affine)
    """
    import tempfile
    import nibabel as nib
    from deepbet import run_bet

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, "t1.nii.gz")
        brain_path = os.path.join(tmp_dir, "brain.nii.gz")
        mask_path = os.path.join(tmp_dir, "mask.nii.gz")
        tiv_path = os.path.join(tmp_dir, "tiv.csv")

        nib.save(nib.Nifti1Image(volume, affine), input_path)

        run_bet(
            [input_path], [brain_path], [mask_path], [tiv_path],
            threshold=threshold, n_dilate=0, no_gpu=True,
        )

        mask_img = nib.load(mask_path)
        mask = np.asarray(mask_img.dataobj, dtype=np.uint8)

    if not np.any(mask):
        raise RuntimeError("Brain mask is empty – deepbet produced no output.")

    return (mask > 0).astype(np.uint8)
```

Also add `import os` near the top of the file (after the existing imports).

Remove these functions entirely:
- `_otsu_threshold` (lines 24-55)
- `_largest_connected_component` (lines 58-65)
- `_spherical_structuring_element` (lines 68-96)

The `scipy` import can also be removed from the top-level imports since
`compute_brain_mask` no longer uses it. BUT keep it if `compute_head_mask`
or other functions still use `ndimage` — check before removing. (Yes,
`compute_head_mask` uses `ndimage`, so keep `from scipy import ndimage`.)

**Step 2: Run tests**

Run: `.venv/bin/pytest tests/test_metal_segmenter.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/metal_segmenter.py
git commit -m "feat: replace hand-rolled brain mask with deepbet CNN extraction"
```

---

### Task 4: Update run_intracranial_mask in SEEGFellow.py

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellow.py:545-579`

**Step 1: Update the caller to pass affine instead of spacing**

In `run_intracranial_mask`, replace lines 564-576 with:

```python
        # --- Compute mask in MRI voxel space ---
        t1_array = arrayFromVolume(self._t1_node)

        # Extract the 4x4 IJK-to-RAS affine for NIfTI export
        ijkToRAS = vtk.vtkMatrix4x4()
        self._t1_node.GetIJKToRASMatrix(ijkToRAS)
        affine = np.array([
            [ijkToRAS.GetElement(r, c) for c in range(4)]
            for r in range(4)
        ])

        brain_mask_t1 = compute_brain_mask(t1_array, affine)
```

Also update the error message on line 579:
```python
        if not np.any(brain_mask_t1):
            raise RuntimeError("Brain mask is empty – deepbet failed on this MRI.")
```

Remove the `spacing_ijk` / `voxel_size_kji` lines (568-569) since they're no longer needed.

**Step 2: Verify no syntax errors**

This runs inside Slicer so we can't run it standalone, but verify no
obvious issues by checking the file parses:

Run: `.venv/bin/python -c "import ast; ast.parse(open('SEEGFellow/SEEGFellow/SEEGFellow.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellow.py
git commit -m "refactor: pass affine to compute_brain_mask in run_intracranial_mask"
```

---

### Task 5: Run full test suite and verify

**Step 1: Run all tests**

Run: `.venv/bin/pytest tests/ -v`
Expected: ALL PASS (30 existing + 2 new = 32 tests)

**Step 2: Final commit if any fixups needed**

Only if test failures required changes.
