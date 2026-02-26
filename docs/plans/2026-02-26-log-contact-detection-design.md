# LoG-Based Contact Center Detection — Design Document

## Overview

Redesign the segmentation-to-detection pipeline for SEEG electrode contact localization. Replace the monolithic threshold + shape-heuristic approach with a discrete, interactive, editable pipeline using Laplacian of Gaussian (LoG) blob detection to find contact centers.

## Motivation

The previous approach (commits `be7836d`, `88ebe40`, `31e42a9`) folded segmentation into detection as a single opaque step. Problems:
- No way to see where things go wrong (intracranial mask? threshold? shape filter?)
- No interactivity (threshold slider didn't update anything visible)
- Shape heuristics (elongation ratio, component size) are fragile and hard to tune
- No way to manually correct intermediate results

## Pipeline

| Step | What | Output | Editable? |
|------|------|--------|-----------|
| 4a — Intracranial Mask | Auto skull-strip (threshold at -200 HU + fill holes + largest component) | Binary segmentation of intracranial space | Yes, via Segment Editor |
| 4b — Metal Threshold | Threshold CT within intracranial mask | Binary segmentation of candidate metal voxels | Yes, via Segment Editor (live threshold slider, paint, erase) |
| 4c — Contact Centers | LoG blob detection at contact scale on CT, filtered by metal mask | Fiducial points at each contact center | Yes, drag/delete/add fiducials |
| 5 — Electrode Grouping | Cluster collinear centers, fit lines, verify spacing | Named electrodes with ordered contacts | Existing Step 5 logic |

Each step produces a visible result. Each can be hand-corrected before proceeding.

## LoG Blob Detection

Inspired by astronomy (star detection) and microscopy (nuclei detection). SEEG contacts are point sources of high intensity at a known scale — the exact problem LoG solves.

**Algorithm:**
1. Take the CT array (masked to intracranial space)
2. Apply `scipy.ndimage.gaussian_laplace` at scale sigma (~1.0–1.5 mm, tunable)
3. Find local minima of the LoG response that are also within the metal mask from step 4b
4. Each minimum = one contact center in IJK space, convert to RAS

**Why this works:**
- Blooming artifact doesn't matter — the center of the bloom IS the contact center
- Naturally rejects wrong-sized structures (bone, dental) — they don't match the scale
- Adjacent contacts that merge in the binary mask still have separate LoG peaks
- One parameter (sigma) replaces min_component_size, min_elongation_size, min_elongation_ratio
- ~15 lines of numpy/scipy. Well-understood math (Lindeberg scale-space, 40+ years of theory)

**Tunable parameter:** sigma slider (default ~1.2 mm, range 0.5–3.0 mm). Larger sigma for electrodes with more blooming.

## Segment Editor Integration

Steps 4a and 4b each produce a segmentation node displayed as an overlay.

**Accept path:** Click "Accept" — the auto result feeds into the next step.

**Edit path:** Click "Edit in Segment Editor" — switches to Slicer's Segment Editor with the segmentation pre-selected. User gets paint, erase, islands, scissors, grow from seeds, etc. Return to SEEGFellow and click "Accept" to proceed with the edited mask.

Implementation is ~3 lines:
```python
slicer.util.selectModule("SegmentEditor")
editor_widget.setSegmentationNode(segmentation_node)
editor_widget.setSourceVolumeNode(ct_node)
```

Step 4a: segmentation node with one segment ("Intracranial"), auto-populated by `compute_head_mask`.
Step 4b: adds a second segment ("Metal"), auto-populated by thresholding within the intracranial mask. Threshold slider updates the segment in real-time.

## Code Changes

### Revert
Undo commits `be7836d`, `88ebe40`, `31e42a9`.

### `metal_segmenter.py`
- Keep `threshold_volume` and `compute_head_mask`
- Remove `cleanup_metal_mask` (replaced by LoG + Segment Editor)
- Add `detect_contact_centers(ct_array, metal_mask, sigma=1.2)` — LoG blob detection, returns (N, 3) IJK coordinates

### `electrode_detector.py`
- `detect_electrodes` now takes (N, 3) contact center RAS coordinates (not raw metal voxels)
- Each center IS a contact — clustering groups them into electrodes, line fitting validates collinearity, spacing analysis checks regularity
- Keep: `cluster_into_electrodes`, `merge_collinear_clusters`, `fit_electrode_axis`, `analyze_spacing`, `orient_deepest_first`
- Remove: `detect_contacts_along_axis` (LoG replaces this), `extract_metal_coords` (no longer needed)

### `SEEGFellow.py` + `SEEGFellow.ui`
- Restore Step 4 as "Segmentation" with 3 sub-panels (4a, 4b, 4c)
- Step 5 becomes electrode grouping/naming
- Add "Edit in Segment Editor" buttons for 4a and 4b
- Add sigma slider for 4c
- Threshold slider in 4b updates the metal segment in real-time

### Tests
- Keep existing tests for surviving functions
- Add tests for `detect_contact_centers` (synthetic blobs at known positions, verify centers found)
- Update `detect_electrodes` tests to pass center coordinates instead of raw voxels
