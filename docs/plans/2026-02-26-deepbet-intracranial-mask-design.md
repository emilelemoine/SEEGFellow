# Replace Intracranial Mask with deepbet

**Date:** 2026-02-26

## Problem

The current `compute_brain_mask` in `metal_segmenter.py` uses a hand-rolled
Otsu + distance-transform erosion/dilation approach that fails to reliably
isolate brain parenchyma. The mask often selects regions outside the skull.

## Solution

Replace the custom algorithm with [deepbet](https://github.com/wwu-mmll/deepbet),
a CNN-based brain extraction tool that achieves 99% Dice on unseen T1 datasets
and runs in ~2 seconds on CPU.

## Design

### `compute_brain_mask` (metal_segmenter.py)

New implementation:

1. Write the T1 array + affine to a temporary `.nii.gz` via nibabel.
2. Call `deepbet.run_bet()` with `no_gpu=True` to produce a brain mask NIfTI.
3. Load the mask back as a numpy array.
4. Clean up temp files.
5. Return `np.ndarray` (binary mask, same as today).

**Signature change:** add `affine: np.ndarray` parameter (4x4 voxel-to-world
matrix) required for valid NIfTI output. Remove `voxel_size_mm`, `erosion_mm`,
`dilation_mm` parameters.

**Removed functions:** `_otsu_threshold`, `_largest_connected_component`,
`_spherical_structuring_element` — all unused after this change.

### `run_intracranial_mask` (SEEGFellow.py)

Extract the IJK-to-RAS affine from `self._t1_node` and pass it to
`compute_brain_mask` instead of spacing.

### Dependencies

Add to `pyproject.toml`:
- `deepbet` — brain extraction
- `nibabel` — NIfTI I/O

### Testing

Add a test for `compute_brain_mask` that mocks `deepbet.run_bet` to avoid
requiring the model weights in CI.

## Constraints

- Input: pre-implant T1-weighted MRI (no electrode artifacts).
- CPU-only inference (`no_gpu=True`).
- Must work standalone (outside 3D Slicer) for testing.
