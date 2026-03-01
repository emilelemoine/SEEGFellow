# Brain Mask Alignment Fix — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the crash and misalignment in `run_intracranial_mask` and `run_metal_threshold` caused by calling `GetMatrixTransformToWorld` on a volume node (method only exists on transform nodes).

**Architecture:** Replace the broken `GetMatrixTransformToWorld` calls with a helper that correctly composes the world-space IJKToRAS by getting the parent transform node first — the same pattern already working in `ElectrodeDetector._get_ijk_to_ras_matrix()`. Add diagnostic prints and cleanup-on-error. Remove the stale `inspect.signature` guard.

**Tech Stack:** 3D Slicer (Python), VTK

---

### Task 1: Add `_world_ijk_to_ras_vtk` helper to `SEEGFellowLogic`

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellow.py:544` (after `_parcellation_affine = None`)

**Step 1: Add the static method**

Insert after line 544 (`self._parcellation_affine = None`), before `def cleanup(self):`:

```python
    @staticmethod
    def _world_ijk_to_ras_vtk(volume_node):
        """Return a vtkMatrix4x4 mapping IJK → world RAS for *volume_node*.

        Composes the node's local IJKToRAS with any parent transform.
        Uses GetMatrixTransformToWorld on the *transform node* (not the
        volume), matching the proven pattern in
        ElectrodeDetector._get_ijk_to_ras_matrix().
        """
        import vtk

        local = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(local)

        parent = volume_node.GetParentTransformNode()
        if parent is None:
            return local

        world_xform = vtk.vtkMatrix4x4()
        parent.GetMatrixTransformToWorld(world_xform)

        result = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Multiply4x4(world_xform, local, result)
        return result
```

**Step 2: Run tests**

Run: `.venv/bin/pytest tests/ -v`
Expected: 72/72 PASS (no behavioral change yet)

**Step 3: Commit**

```
feat: add _world_ijk_to_ras_vtk helper for correct world-space geometry
```

---

### Task 2: Fix `run_intracranial_mask` — remove broken code, add diagnostics

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellow.py:691-862`

**Step 1: Remove `inspect.signature` guard (lines 732-740)**

Replace:
```python
        import inspect

        from SEEGFellowLib.brain_mask import SynthSegBrainMask

        compute_sig = inspect.signature(strategy.compute)
        if output_dir and "output_dir" in compute_sig.parameters:
            brain_mask_t1 = strategy.compute(t1_array, affine, output_dir=output_dir)
        else:
            brain_mask_t1 = strategy.compute(t1_array, affine)
```

With:
```python
        brain_mask_t1 = strategy.compute(t1_array, affine, output_dir=output_dir)
```

**Step 2: Replace broken `GetMatrixTransformToWorld` block (lines 798-814)**

Replace:
```python
        # The resampling CLI and updateSegmentBinaryLabelmapFromArray both read only
        # the LOCAL IJKToRAS of the reference node; parent transforms are ignored.
        # Instead of hardening the actual CT node (which risks incorrect geometry if
        # hardenTransform does not update IJKToRAS), compose the world-space matrix
        # manually and place it on a temporary reference node.
        ct_world_matrix = vtk.vtkMatrix4x4()
        self._ct_node.GetMatrixTransformToWorld(ct_world_matrix)
        ct_local_ijk_to_ras = vtk.vtkMatrix4x4()
        self._ct_node.GetIJKToRASMatrix(ct_local_ijk_to_ras)
        ct_world_ijk_to_ras = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Multiply4x4(
            ct_world_matrix, ct_local_ijk_to_ras, ct_world_ijk_to_ras
        )
        ct_world_ref = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", "_SEEGFellow_CT_WorldRef"
        )
        ct_world_ref.SetIJKToRASMatrix(ct_world_ijk_to_ras)
```

With:
```python
        ct_world_ijk_to_ras = self._world_ijk_to_ras_vtk(self._ct_node)
        ct_world_ref = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", "_SEEGFellow_CT_WorldRef"
        )
        ct_world_ref.SetIJKToRASMatrix(ct_world_ijk_to_ras)
```

**Step 3: Add diagnostic prints**

After the existing "Brain mask voxel count in MRI space" print (line ~773), add:
```python
        print(
            f"[SEEGFellow] T1 IJKToRAS diagonal (spacing): "
            f"{[ijkToRAS.GetElement(i, i) for i in range(3)]}"
        )
        print(
            f"[SEEGFellow] T1 has parent transform: "
            f"{self._t1_node.GetParentTransformNode() is not None}"
        )
```

After the `ct_world_ref.SetIJKToRASMatrix(ct_world_ijk_to_ras)` line, add:
```python
        ct_local = vtk.vtkMatrix4x4()
        self._ct_node.GetIJKToRASMatrix(ct_local)
        print(
            f"[SEEGFellow] CT local IJKToRAS origin: "
            f"{[ct_local.GetElement(i, 3) for i in range(3)]}"
        )
        print(
            f"[SEEGFellow] CT world IJKToRAS origin: "
            f"{[ct_world_ijk_to_ras.GetElement(i, 3) for i in range(3)]}"
        )
        print(
            f"[SEEGFellow] CT has parent transform: "
            f"{self._ct_node.GetParentTransformNode() is not None}"
        )
```

**Step 4: Add try/finally for cleanup of temporary nodes**

Wrap the section from `brain_label_node` creation (line ~780) through the end of the method in try/finally so temporary nodes are always cleaned up:

```python
        # --- Create temporary nodes for resampling ---
        brain_label_node = None
        brain_label_ct = None
        ct_world_ref = None
        try:
            brain_label_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode", "_SEEGFellow_BrainMask_MRI"
            )
            # ... existing code through segment creation ...
        finally:
            for tmp in (brain_label_node, brain_label_ct):
                if tmp is not None:
                    slicer.mrmlScene.RemoveNode(tmp)
```

Note: `ct_world_ref` must be removed AFTER segment creation (it's used as reference geometry), so it stays in the try block and is removed after `updateSegmentBinaryLabelmapFromArray`, same as now.

**Step 5: Run tests**

Run: `.venv/bin/pytest tests/ -v`
Expected: 72/72 PASS

**Step 6: Commit**

```
fix: use correct API for CT world-space geometry in run_intracranial_mask
```

---

### Task 3: Fix `run_metal_threshold` — same `GetMatrixTransformToWorld` bug

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellow.py:864-911`

**Step 1: Replace broken block (lines 888-903)**

Replace:
```python
        # updateSegmentBinaryLabelmapFromArray uses only the local IJKToRAS of the
        # reference node (parent transforms are not followed).  Compose the world-space
        # IJKToRAS manually and place it on a temporary reference node so the CT node
        # itself is never modified.
        ct_world_matrix = vtk.vtkMatrix4x4()
        self._ct_node.GetMatrixTransformToWorld(ct_world_matrix)
        ct_local_ijk_to_ras = vtk.vtkMatrix4x4()
        self._ct_node.GetIJKToRASMatrix(ct_local_ijk_to_ras)
        ct_world_ijk_to_ras = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Multiply4x4(
            ct_world_matrix, ct_local_ijk_to_ras, ct_world_ijk_to_ras
        )
        ct_world_ref = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", "_SEEGFellow_CT_WorldRef"
        )
        ct_world_ref.SetIJKToRASMatrix(ct_world_ijk_to_ras)
```

With:
```python
        ct_world_ijk_to_ras = self._world_ijk_to_ras_vtk(self._ct_node)
        ct_world_ref = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", "_SEEGFellow_CT_WorldRef"
        )
        ct_world_ref.SetIJKToRASMatrix(ct_world_ijk_to_ras)
```

**Step 2: Run tests**

Run: `.venv/bin/pytest tests/ -v`
Expected: 72/72 PASS

**Step 3: Commit**

```
fix: use correct API for CT world-space geometry in run_metal_threshold
```

---

## Verification (in Slicer)

After all tasks, reload the SEEGFellow module in Slicer and run the pipeline:

1. **Step 1 (Load Data):** Load T1 + CT
2. **Step 2 (Register):** Run registration
3. **Step 3a (Brain Mask):** Click Compute Brain Mask. Check Python console output:
   - `T1 has parent transform: False` (T1 should be the reference frame)
   - `CT has parent transform: True`
   - `CT local IJKToRAS origin` and `CT world IJKToRAS origin` should differ (registration offset)
   - `Brain mask voxel count in CT space` should be >1M
   - No crash
4. **Segment Editor:** Click Edit — should open Segment Editor with the Brain segment visible
5. **Visual check:** Brain segment should cover the brain in slice views, not be a small blob in the corner
6. **Step 3b (Metal Threshold):** Click Apply — should work without `NoneType` error
7. **Metal segment:** Should be inside the brain, not offset
