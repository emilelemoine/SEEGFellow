# Session Restore from Saved Slicer Scene

## Problem

When developing/debugging SEEGFellow, restarting Slicer requires manually re-running steps 1–3 (load volumes, rough align, register) before reaching the step under development. Slicer's scene save/load preserves MRML nodes correctly, but SEEGFellowLogic reinitializes with `None` references.

## Solution

Auto-reconnect Logic to existing scene nodes on module load using node-name conventions.

## Design

### New method: `SEEGFellowLogic.try_restore_from_scene() -> bool`

Scans the MRML scene for known nodes:

1. **Registration transform**: find `vtkMRMLLinearTransformNode` named `"CT_to_T1_Registration"` → `_registration_transform_node`
2. **CT volume**: find the `vtkMRMLScalarVolumeNode` whose parent transform is the registration transform → `_ct_node`
3. **T1 volume**: among remaining scalar volumes, pick the first one → `_t1_node`
4. **Segmentation**: find `vtkMRMLSegmentationNode` named `"SEEGFellow Segmentation"` → `_segmentation_node`
5. **Masks**: if segmentation has a `"Brain"` segment, recover `_head_mask` from its binary labelmap. Same for `"Metal"` → `_metal_mask`.

Returns `True` if at least CT + T1 + registration transform are found.

If there's ambiguity (0 or 2+ CT candidates), skip auto-restore.

### Widget integration

`SEEGFellowWidget.setup()` calls `logic.try_restore_from_scene()` after creating the logic. On success, uncollapse the furthest-reached panel:

- CT + T1 + transform → step 4a (intracranial mask)
- Above + Brain segment → step 4b (metal threshold)
- Above + Metal segment → step 4c (contact detection)

Status message: "Restored session from scene."

### Out of scope

- `electrodes` list (re-run detection from step 4c)
- `_rough_transform_node` (not needed post-registration)
- Seed/direction markup nodes (manual fallback state)
