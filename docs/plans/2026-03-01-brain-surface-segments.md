# Brain Surface Segments Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automatically add "Left Hemisphere" and "Right Hemisphere" segments to the SEEGFellow Segmentation node after SynthSeg completes, for cortical surface 3D visualization.

**Architecture:** A private helper `_add_hemisphere_segments()` is added to `SEEGFellowLogic` and called at the end of `run_intracranial_mask`. It builds binary masks by OR-ing relevant DKT/subcortical labels from `self._parcellation`, then resamples each mask to CT space (same pattern as the Brain segment) and adds them to the existing segmentation node with colour and 50% 3D opacity. Two module-level frozensets define the label memberships.

**Tech Stack:** Python, numpy, 3D Slicer MRML/CLI API (vtkMRMLSegmentationNode, resamplescalarvectordwivolume), pytest

---

### Task 1: Add hemisphere label constants and unit tests

These are pure-Python constants — the only part of this feature that can be unit-tested without Slicer.

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellow.py` (add two module-level frozensets after the imports)
- Create: `tests/test_hemisphere_labels.py`

**Step 1: Write the failing test**

```python
# tests/test_hemisphere_labels.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow"))

# Import will fail until we add the constants
from SEEGFellow import _LEFT_HEMISPHERE_LABELS, _RIGHT_HEMISPHERE_LABELS


class TestHemisphereLabels:
    def test_no_overlap(self):
        assert _LEFT_HEMISPHERE_LABELS.isdisjoint(_RIGHT_HEMISPHERE_LABELS)

    def test_left_contains_expected_subcortical(self):
        # Left Cerebral Cortex=3, Left WM=2, Left Hippocampus=17, Left Thalamus=10
        assert {2, 3, 10, 17}.issubset(_LEFT_HEMISPHERE_LABELS)

    def test_right_contains_expected_subcortical(self):
        # Right Cerebral Cortex=42, Right WM=41, Right Hippocampus=53, Right Thalamus=49
        assert {41, 42, 49, 53}.issubset(_RIGHT_HEMISPHERE_LABELS)

    def test_left_cortical_parcels(self):
        # A few DKT 1000-series parcels must be present
        assert {1002, 1024, 1035}.issubset(_LEFT_HEMISPHERE_LABELS)

    def test_right_cortical_parcels(self):
        # Corresponding 2000-series
        assert {2002, 2024, 2035}.issubset(_RIGHT_HEMISPHERE_LABELS)

    def test_midline_labels_excluded(self):
        # 3rd Ventricle=14, 4th Ventricle=15, Brain Stem=16, CSF=24
        midline = {14, 15, 16, 24}
        assert not midline.intersection(_LEFT_HEMISPHERE_LABELS)
        assert not midline.intersection(_RIGHT_HEMISPHERE_LABELS)

    def test_reasonable_sizes(self):
        # 14 subcortical + 31 DKT cortical = 45 per side
        assert len(_LEFT_HEMISPHERE_LABELS) == 45
        assert len(_RIGHT_HEMISPHERE_LABELS) == 45
```

**Step 2: Run test to confirm failure**

```bash
.venv/bin/pytest tests/test_hemisphere_labels.py -v
```
Expected: `ImportError: cannot import name '_LEFT_HEMISPHERE_LABELS'`

**Step 3: Add the constants to SEEGFellow.py**

Insert after the `from __future__ import annotations` line and existing imports in `SEEGFellow/SEEGFellow/SEEGFellow.py`, before the `class SEEGFellow` definition:

```python
# SynthSeg DKT label sets for hemisphere 3-D surface visualization.
# Left: subcortical base labels + DKT 1000-series cortical parcels.
# Right: subcortical base labels + DKT 2000-series cortical parcels.
# Midline labels (14 3rd Ventricle, 15 4th Ventricle, 16 Brain Stem, 24 CSF) excluded.
_LEFT_HEMISPHERE_LABELS: frozenset[int] = frozenset([
    # Subcortical
    2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28,
    # DKT cortical (1000-series; offsets match _DKT_NAMES keys in contact_labeler.py)
    1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013,
    1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024,
    1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035,
])

_RIGHT_HEMISPHERE_LABELS: frozenset[int] = frozenset([
    # Subcortical
    41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60,
    # DKT cortical (2000-series)
    2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
    2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024,
    2025, 2026, 2027, 2028, 2029, 2030, 2031, 2034, 2035,
])
```

**Step 4: Run test to confirm it passes**

```bash
.venv/bin/pytest tests/test_hemisphere_labels.py -v
```
Expected: 7 tests PASSED.

**Step 5: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellow.py tests/test_hemisphere_labels.py
git commit -m "feat: add hemisphere label constants for 3D surface segments"
```

---

### Task 2: Implement `_add_hemisphere_segments` helper

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellow.py`
  - Add `_add_hemisphere_segments` method to `SEEGFellowLogic` (insert after the Brain segment creation block in `run_intracranial_mask`, around line 1034)

**Step 1: Add the helper method**

Insert as a new private method of `SEEGFellowLogic`, after `run_intracranial_mask` and before `run_metal_threshold`:

```python
def _add_hemisphere_segments(self) -> None:
    """Add Left/Right Hemisphere segments from the SynthSeg parcellation.

    Uses all left- and right-hemisphere DKT labels to build a binary mask
    whose outer 3-D surface shows cortical gyri and sulci. Each hemisphere is
    resampled to CT space using the same pipeline as the Brain segment.
    Segments are set to 50 % 3-D opacity so electrodes remain visible.

    Called automatically at the end of run_intracranial_mask when a
    SynthSegBrainMask strategy was used.

    Example::

        logic.run_intracranial_mask()  # calls _add_hemisphere_segments internally
    """
    import numpy as np
    import vtk
    import slicer
    from slicer.util import arrayFromVolume, updateVolumeFromArray

    hemispheres = [
        ("Left Hemisphere",  _LEFT_HEMISPHERE_LABELS,  (0.6,  0.65, 0.75)),
        ("Right Hemisphere", _RIGHT_HEMISPHERE_LABELS, (0.75, 0.65, 0.65)),
    ]

    for name, labels, color in hemispheres:
        mask = np.isin(self._parcellation, sorted(labels)).astype(np.uint8)

        label_node = None
        label_ct = None
        try:
            # --- Temp labelmap in parcellation (SynthSeg 1 mm) space ---
            safe_name = name.replace(" ", "")
            label_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode", f"_SEEGFellow_{safe_name}_MRI"
            )
            mask_mat = vtk.vtkMatrix4x4()
            for r_idx in range(4):
                for c_idx in range(4):
                    mask_mat.SetElement(
                        r_idx, c_idx, float(self._parcellation_affine[r_idx, c_idx])
                    )
            label_node.SetIJKToRASMatrix(mask_mat)
            updateVolumeFromArray(label_node, mask)

            # Inherit and harden T1 parent transform (CT→T1 registration)
            t1_transform = self._t1_node.GetParentTransformNode()
            if t1_transform is not None:
                label_node.SetAndObserveTransformNodeID(t1_transform.GetID())
            slicer.vtkSlicerTransformLogic.hardenTransform(label_node)

            # --- Resample to CT space ---
            label_ct = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode", f"_SEEGFellow_{safe_name}_CT"
            )
            ct_transform_id = None
            if self._ct_node.GetParentTransformNode() is not None:
                ct_transform_id = self._ct_node.GetParentTransformNodeID()
                slicer.vtkSlicerTransformLogic.hardenTransform(self._ct_node)

            params = {
                "inputVolume":       label_node.GetID(),
                "referenceVolume":   self._ct_node.GetID(),
                "outputVolume":      label_ct.GetID(),
                "interpolationMode": "NearestNeighbor",
            }
            slicer.cli.runSync(
                slicer.modules.resamplescalarvectordwivolume, None, params
            )

            if ct_transform_id:
                self._ct_node.SetAndObserveTransformNodeID(ct_transform_id)

            hemi_mask = (np.array(arrayFromVolume(label_ct)) > 0).astype(np.uint8)
            print(
                f"[SEEGFellow] {name} voxel count in CT space: {hemi_mask.sum()}"
            )

            # --- Add segment ---
            seg = self._segmentation_node.GetSegmentation()
            existing_id = seg.GetSegmentIdBySegmentName(name)
            if existing_id:
                seg.RemoveSegment(existing_id)

            segment_id = seg.AddEmptySegment(name, name)
            seg.GetSegment(segment_id).SetColor(*color)
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                hemi_mask, self._segmentation_node, segment_id, label_ct
            )

            # 50 % opacity so electrodes remain visible through the surface
            display_node = self._segmentation_node.GetDisplayNode()
            display_node.SetSegmentOpacity3D(segment_id, 0.5)

        finally:
            for tmp in (label_node, label_ct):
                if tmp is not None:
                    slicer.mrmlScene.RemoveNode(tmp)
```

**Step 2: Call the helper from `run_intracranial_mask`**

At the end of `run_intracranial_mask`, after the `finally` block that cleans up `brain_label_node` / `brain_label_ct` (around line 1033), add:

```python
        # Add hemisphere surface segments for 3-D visualization
        from SEEGFellowLib.brain_mask import SynthSegBrainMask

        if isinstance(strategy, SynthSegBrainMask) and self._parcellation is not None:
            self._add_hemisphere_segments()
```

Note: this goes *outside* the `try/finally` block that wraps the Brain segment code — i.e., after the `finally:` closes, at the same indentation level as the `try:`.

**Step 3: Run existing tests to confirm nothing broke**

```bash
.venv/bin/pytest tests/ -v
```
Expected: all tests (72 + 7 new) PASS.

**Step 4: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellow.py
git commit -m "feat: add Left/Right Hemisphere 3D surface segments after SynthSeg"
```

---

## Verification

After both tasks are done, all 79 tests should pass:

```bash
.venv/bin/pytest tests/ -v
```

Manual verification in Slicer:
1. Run the full pipeline through Step 3a (Compute Brain Mask).
2. Open the 3D view — two new translucent hemisphere segments should appear alongside the existing Brain/Metal segments.
3. Toggle to Segment Editor: confirm "Left Hemisphere" is blue-grey, "Right Hemisphere" is rose-grey.
4. Confirm gyral folding is visible on the cortical surface in 3D.
