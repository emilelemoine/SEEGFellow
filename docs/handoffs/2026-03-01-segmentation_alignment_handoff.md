# Handoff: Segmentation Alignment Fix + SynthSeg Output Directory

## Goal

Fix two problems observed after step 3a (Brain Segmentation):

1. The intracranial mask segment (and metal mask segment) appeared as a small blob
   in the wrong anatomical position — clearly outside the brain in the slice views —
   even though the SynthSeg parcellation overlay was correctly aligned.
2. SynthSeg can take 10–30 minutes; if Slicer crashes the result is lost and
   must be recomputed. Add a persistent output directory so the NIfTI file is
   saved and reused on subsequent runs.

---

## Root Cause: `hardenTransform` Does Not Update `GetIJKToRASMatrix()`

### The broken pattern (was used in both `run_intracranial_mask` and `run_metal_threshold`)

```python
# Save CT geometry, harden, use as reference, restore
ct_ijk_to_ras_orig = vtk.vtkMatrix4x4()
self._ct_node.GetIJKToRASMatrix(ct_ijk_to_ras_orig)
slicer.vtkSlicerTransformLogic.hardenTransform(self._ct_node)
# ... CLI or updateSegmentBinaryLabelmapFromArray using self._ct_node as ref ...
self._ct_node.SetIJKToRASMatrix(ct_ijk_to_ras_orig)
self._ct_node.SetAndObserveTransformNodeID(ct_transform_id)
```

The intent was: harden bakes the registration transform into the CT's local
`IJKToRAS`, so `GetIJKToRASMatrix()` returns the world-space (T1/RAS) mapping.
Both `resamplescalarvectordwivolume` CLI and `updateSegmentBinaryLabelmapFromArray`
read only the **local** `IJKToRAS` (they do not follow parent transforms).

**The problem**: `hardenTransform` on a Slicer volume node apparently does **not**
update `GetIJKToRASMatrix()` to include the transform — it only removes the parent
transform pointer. So after hardening, `GetIJKToRASMatrix()` still returns the CT's
native pre-registration matrix. Using this as a reference placed the brain mask and
metal mask segments in CT native coordinate space (before registration), which is
spatially offset from T1/RAS world space. Hence the misaligned blob.

This explains why:
- The parcellation was correct (stored directly in T1/RAS via its own NIfTI affine).
- The brain mask and metal mask were both wrong (both used the same broken pattern).
- The metal mask appeared "inside" the brain mask blob — they shared the same
  misaligned coordinate frame.

### The fix (both methods)

Compute the world-space IJKToRAS **manually**, without touching the CT node at all:

```python
ct_world_matrix = vtk.vtkMatrix4x4()
self._ct_node.GetMatrixTransformToWorld(ct_world_matrix)  # full chain to world
ct_local_ijk_to_ras = vtk.vtkMatrix4x4()
self._ct_node.GetIJKToRASMatrix(ct_local_ijk_to_ras)
ct_world_ijk_to_ras = vtk.vtkMatrix4x4()
vtk.vtkMatrix4x4.Multiply4x4(ct_world_matrix, ct_local_ijk_to_ras, ct_world_ijk_to_ras)

ct_world_ref = slicer.mrmlScene.AddNewNodeByClass(
    "vtkMRMLLabelMapVolumeNode", "_SEEGFellow_CT_WorldRef"
)
ct_world_ref.SetIJKToRASMatrix(ct_world_ijk_to_ras)
# use ct_world_ref as reference for CLI and/or updateSegmentBinaryLabelmapFromArray
slicer.mrmlScene.RemoveNode(ct_world_ref)
```

This is the same pattern already used successfully in
`ElectrodeDetector._get_ijk_to_ras_matrix()` for correct contact placement.

---

## Changes Made

### `SEEGFellow.py` — `run_intracranial_mask()`

- Removed `hardenTransform` / restore of `self._ct_node`.
- Added manual world-space IJKToRAS computation → `ct_world_ref` temp node.
- `ct_world_ref` is now used as the reference for:
  - `resamplescalarvectordwivolume` CLI (so the resampled output is in CT world-space voxel grid).
  - `SetReferenceImageGeometryParameterFromVolumeNode` on the segmentation node.
  - `updateSegmentBinaryLabelmapFromArray` for the Brain segment.
- `ct_world_ref` is removed after segment creation.
- Accepted `output_dir: str | None = None` parameter (see below).

### `SEEGFellow.py` — `run_metal_threshold()`

- Removed `hardenTransform` / restore of `self._ct_node`.
- Same `ct_world_ref` pattern for `updateSegmentBinaryLabelmapFromArray`.

### `SEEGFellowLib/brain_mask.py` — `SynthSegBrainMask.compute()`

- Added `output_dir: str | None = None` parameter.
- If `output_dir` is set and `output_dir/synthseg_parc.nii.gz` already exists,
  loads it directly (skips the SynthSeg subprocess entirely).
- After a fresh SynthSeg run, copies the NIfTI output to
  `output_dir/synthseg_parc.nii.gz` for future reuse.

### `SEEGFellow.ui` — Step 3a form

- Added row 4: `Output directory` label + `ctkPathLineEdit` (dirs-only)
  named `synthSegOutputDirLineEdit`.

### `SEEGFellow.py` — `_on_compute_head_mask_clicked()`

- Reads `synthSegOutputDirLineEdit.currentPath`, passes it as `output_dir`
  to `run_intracranial_mask()`.
- Uses `inspect.signature` to check whether the loaded `compute()` supports
  `output_dir` before calling it (guards against Slicer loading a stale
  `brain_mask.pyc` in a live session where only `SEEGFellow.py` was reloaded).

---

## Current State

- 72/72 tests passing.
- Brain mask and metal mask segments should now appear aligned with the
  brain in the CT slice views.
- Output directory feature: set a directory in Step 3a, and SynthSeg results
  are saved to `<dir>/synthseg_parc.nii.gz` and reloaded on future runs.

---

## What Has NOT Been Tested Yet

The alignment fix has not been re-run in Slicer after this session.
The next agent/session should:

1. Reload the SEEGFellow module in Slicer (or restart Slicer to ensure
   fresh `.pyc` files are loaded from the updated sources).
2. Run steps 1–3b and verify the Brain segment covers the full brain in the
   slice views and the 3D view.
3. Verify the Metal segment is inside the brain (not offset to a corner).
4. Test the output directory: set a directory, run segmentation, confirm
   `synthseg_parc.nii.gz` is created. Re-run and confirm it loads instantly.

---

## Key Files

| File | What changed |
|------|-------------|
| `SEEGFellow/SEEGFellow.py` | `run_intracranial_mask`: removed hardenTransform, use `ct_world_ref`; `run_metal_threshold`: same; `_on_compute_head_mask_clicked`: output_dir support |
| `SEEGFellowLib/brain_mask.py` | `SynthSegBrainMask.compute()`: added `output_dir` caching |
| `Resources/UI/SEEGFellow.ui` | Added output directory widget to Step 3a |

## Coordinate System Notes (important for future debugging)

- All contact positions and segmentation segments should be in **T1/RAS world space**.
- The CT has a parent transform (`CT_to_T1_Registration`). Its local `IJKToRAS`
  maps to CT native space; the parent transform then maps to T1/RAS.
- **`GetMatrixTransformToWorld`** × **`GetIJKToRASMatrix`** = world-space IJKToRAS. Use this, never `hardenTransform`.
- The parcellation (`_SEEGFellow_SynthSeg_Parcellation`) has its own NIfTI affine
  stored directly as local IJKToRAS (no parent transform) → it is inherently in
  T1/RAS world space.
