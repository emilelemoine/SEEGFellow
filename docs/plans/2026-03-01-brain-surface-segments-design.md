# Design: Left/Right Brain Surface Segments

**Date:** 2026-03-01

## Goal

Add "Left Hemisphere" and "Right Hemisphere" segments to the existing `SEEGFellow Segmentation` node.
These are for 3D visualization only — the outer surface of the parcellation-derived hemispheric volumes
shows the cortical folding (gyri/sulci), unlike the current "Brain" segment which is a coarse CT-space
intracranial mask.

## Labels

SynthSeg DKT parcellation labels used per hemisphere:

- **Left:** base subcortical (2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28) + DKT cortical (1000–1035)
- **Right:** base subcortical (41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60) + DKT cortical (2000–2035)

Midline labels (14, 15, 16 Ventricles/Brain Stem, 24 CSF) are excluded from both hemispheres.

## Colors

- Left: `(0.6, 0.65, 0.75)` muted blue-grey
- Right: `(0.75, 0.65, 0.65)` muted rose-grey
- Opacity: 50% (so electrodes are visible through the brain in 3D)

## Implementation

### Where

`SEEGFellow.py` → `SEEGFellowLogic.run_intracranial_mask()`, at the end (after Brain segment is created).

A private helper `_add_hemisphere_segments()` is called from `run_intracranial_mask` when a
`SynthSegBrainMask` strategy was used and `self._parcellation` is set.

### Helper logic

1. Build `left_mask` / `right_mask` by OR-ing all relevant label voxels from `self._parcellation`.
2. For each hemisphere:
   a. Create a temporary `vtkMRMLLabelMapVolumeNode` with the parcellation affine (`self._parcellation_affine`).
   b. Inherit and harden the T1 parent transform (same pattern as Brain mask).
   c. Resample to CT geometry via `resamplescalarvectordwivolume` CLI (NearestNeighbor).
   d. Remove any pre-existing segment with that name, add new segment with color + 50% opacity.
3. Clean up temp nodes in `finally`.

### Session restore

`restore_from_scene()` already restores Brain/Metal by name. Add analogous restore for
"Left Hemisphere" and "Right Hemisphere" (no mask stored in logic state — these are viz-only).

## Out of Scope

- No new UI elements.
- No changes to CSV export or contact labeling.
- The existing "Brain" segment is unchanged.
