# Session Restore Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Auto-reconnect SEEGFellowLogic to existing MRML scene nodes so the user can resume from step 4 after a Slicer restart.

**Architecture:** Add `try_restore_from_scene()` to Logic that scans nodes by name convention, then call it from Widget.setup(). No new dependencies.

**Tech Stack:** 3D Slicer MRML API, Python

**Design doc:** `docs/plans/2026-02-26-session-restore-design.md`

---

### Task 1: Add `try_restore_from_scene()` to Logic

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellow.py:364-383` (SEEGFellowLogic class)

**Step 1: Write the method**

Add after `cleanup()` (line 395), before the Step 1 comment (line 396):

```python
    # -------------------------------------------------------------------------
    # Session restore
    # -------------------------------------------------------------------------

    def try_restore_from_scene(self) -> bool:
        """Reconnect to existing scene nodes from a saved Slicer scene.

        Scans for nodes by name convention:
        - Transform: "CT_to_T1_Registration"
        - CT: scalar volume with that transform as parent
        - T1: the other scalar volume
        - Segmentation: "SEEGFellow Segmentation" (optional)

        Returns True if CT + T1 + registration transform were found.
        """
        # --- Find registration transform ---
        transform = slicer.util.getFirstNodeByClassByName(
            "vtkMRMLLinearTransformNode", "CT_to_T1_Registration"
        )
        if transform is None:
            return False

        # --- Find CT (volume under the registration transform) ---
        scene = slicer.mrmlScene
        ct_node = None
        other_volumes = []
        n = scene.GetNumberOfNodesByClass("vtkMRMLScalarVolumeNode")
        for i in range(n):
            vol = scene.GetNthNodeByClass(i, "vtkMRMLScalarVolumeNode")
            if vol.GetParentTransformNode() == transform:
                if ct_node is not None:
                    return False  # ambiguous: 2+ volumes under transform
                ct_node = vol
            else:
                other_volumes.append(vol)

        if ct_node is None or len(other_volumes) == 0:
            return False

        t1_node = other_volumes[0]

        # --- Assign core nodes ---
        self._registration_transform_node = transform
        self._ct_node = ct_node
        self._t1_node = t1_node

        # --- Optionally restore segmentation and masks ---
        seg_node = slicer.util.getFirstNodeByClassByName(
            "vtkMRMLSegmentationNode", "SEEGFellow Segmentation"
        )
        if seg_node is not None:
            self._segmentation_node = seg_node
            seg = seg_node.GetSegmentation()

            brain_id = seg.GetSegmentIdBySegmentName("Brain")
            if brain_id:
                import numpy as np

                brain_array = slicer.util.arrayFromSegmentBinaryLabelmap(
                    seg_node, brain_id, ct_node
                )
                self._head_mask = (brain_array > 0).astype(np.uint8)

            metal_id = seg.GetSegmentIdBySegmentName("Metal")
            if metal_id:
                import numpy as np

                metal_array = slicer.util.arrayFromSegmentBinaryLabelmap(
                    seg_node, metal_id, ct_node
                )
                self._metal_mask = (metal_array > 0).astype(np.uint8)

        return True
```

**Step 2: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellow.py
git commit -m "feat: add try_restore_from_scene to Logic"
```

---

### Task 2: Call restore from Widget.setup()

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellow.py:27-35` (SEEGFellowWidget.setup)

**Step 1: Add restore call at the end of setup()**

After the electrode table setup (after line 101), add:

```python
        # Auto-restore from saved scene
        self._try_restore_session()
```

**Step 2: Add the `_try_restore_session` method to SEEGFellowWidget**

Add after `setup()`, before `cleanup()`:

```python
    def _try_restore_session(self):
        """Attempt to reconnect to nodes from a saved Slicer scene."""
        if not self.logic.try_restore_from_scene():
            return

        # Determine furthest-reached step and uncollapse that panel
        has_brain = getattr(self.logic, "_head_mask", None) is not None
        has_metal = getattr(self.logic, "_metal_mask", None) is not None

        if has_metal:
            self.ui.contactDetectionCollapsibleButton.collapsed = False
        elif has_brain:
            self.ui.metalThresholdCollapsibleButton.collapsed = False
        else:
            self.ui.intracranialMaskCollapsibleButton.collapsed = False

        slicer.util.showStatusMessage("Restored session from scene.")
```

**Step 3: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellow.py
git commit -m "feat: auto-restore session on module load"
```

---

### Task 3: Manual test in Slicer

**Step 1: Load data and run through steps 1–3 normally**

**Step 2: File → Save Scene (as .mrb)**

**Step 3: Restart Slicer, load the .mrb scene**

**Step 4: Switch to SEEGFellow module**

Expected: status bar shows "Restored session from scene.", step 4a panel is uncollapsed, and clicking "Compute Brain Mask" works without errors.

**Step 5: Repeat with scene saved after step 4a (brain mask computed)**

Expected: step 4b panel is uncollapsed on restore.
