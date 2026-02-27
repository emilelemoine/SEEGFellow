# Improved Metal Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve electrode detection accuracy by adding a head mask, fixing the elongation filter, and folding the segmentation step into detection with a live voxel-count feedback slider.

**Architecture:** Three changes in sequence: (1) improve `metal_segmenter.py` with `compute_head_mask` and a fixed `cleanup_metal_mask`; (2) update `ElectrodeDetector.detect_all` to take a CT node + threshold directly instead of a pre-built labelmap; (3) remove the separate segmentation UI step and add the threshold slider with live count to the detection panel.

**Tech Stack:** Python 3.9+, numpy, scipy.ndimage, 3D Slicer Qt UI (.ui XML)

---

## Task 1: Improve `cleanup_metal_mask` and add `compute_head_mask`

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/metal_segmenter.py`
- Modify: `tests/test_metal_segmenter.py`

### Step 1: Write failing tests

Add to `tests/test_metal_segmenter.py` (after existing tests):

```python
from SEEGFellowLib.metal_segmenter import compute_head_mask


class TestComputeHeadMask:
    def test_excludes_air_background(self):
        """Voxels at -1000 HU (air) should be excluded from the head mask."""
        vol = np.full((20, 20, 20), -1000.0, dtype=np.float32)
        # A blob of soft tissue in the centre
        vol[8:12, 8:12, 8:12] = 40.0
        mask = compute_head_mask(vol)
        assert mask[10, 10, 10] == 1
        assert mask[0, 0, 0] == 0

    def test_fills_internal_air(self):
        """Air pockets inside the head (sinuses) should be filled."""
        vol = np.full((30, 30, 30), -1000.0, dtype=np.float32)
        # Head shell
        vol[5:25, 5:25, 5:25] = 40.0
        # Internal air pocket
        vol[12:18, 12:18, 12:18] = -900.0
        mask = compute_head_mask(vol)
        # Internal air should be inside the head mask
        assert mask[15, 15, 15] == 1


class TestCleanupMetalMaskImproved:
    def test_head_mask_excludes_external_metal(self):
        """Metal outside the head mask should be removed."""
        ct = np.full((40, 40, 40), -1000.0, dtype=np.float32)
        # Patient head in centre
        ct[10:30, 10:30, 10:30] = 40.0
        # Electrode inside head
        for i in range(12, 28):
            ct[i, 20, 20] = 3000.0
        # Metal object outside head (e.g. headframe)
        for i in range(0, 8):
            ct[i, 5, 5] = 3000.0

        from SEEGFellowLib.metal_segmenter import compute_head_mask
        head_mask = compute_head_mask(ct)
        metal = threshold_volume(ct, threshold=2500)
        cleaned = cleanup_metal_mask(metal, head_mask=head_mask)

        # Electrode inside head survives
        assert np.any(cleaned[12:28, 20, 20] == 1)
        # External metal removed
        assert np.sum(cleaned[0:8, 5, 5]) == 0

    def test_medium_bulky_component_rejected(self):
        """A medium-sized bulky component (30-500 voxels) should be rejected."""
        mask = np.zeros((50, 50, 50), dtype=np.uint8)
        # Medium bulky blob (~5x5x5 = 125 voxels, elongation ~1)
        mask[10:15, 10:15, 10:15] = 1
        # Elongated electrode (thin line)
        for i in range(20, 45):
            mask[i, 25, 25] = 1
        cleaned = cleanup_metal_mask(mask)
        # Bulky blob removed
        assert np.sum(cleaned[10:15, 10:15, 10:15]) == 0
        # Electrode kept
        assert np.any(cleaned[20:45, 25, 25] == 1)

    def test_small_fragment_kept(self):
        """Small fragments (< 30 voxels) survive — they may be electrode sub-clusters."""
        mask = np.zeros((30, 30, 30), dtype=np.uint8)
        # 10-voxel fragment (e.g. 2 contacts of a gapped electrode)
        for i in range(10, 15):
            mask[i, 15, 15] = 1
            mask[i, 16, 15] = 1
        cleaned = cleanup_metal_mask(mask)
        assert np.any(cleaned[10:15, 15:17, 15] == 1)
```

**Step 2: Run tests to confirm they fail**

```bash
.venv/bin/pytest tests/test_metal_segmenter.py::TestComputeHeadMask tests/test_metal_segmenter.py::TestCleanupMetalMaskImproved -v
```

Expected: `FAILED` (functions missing or wrong behaviour)

**Step 3: Implement `compute_head_mask` and update `cleanup_metal_mask`**

Replace the entire `metal_segmenter.py` with:

```python
# SEEGFellow/SEEGFellow/SEEGFellowLib/metal_segmenter.py
"""Metal segmentation from CT volume.

Core algorithm functions operate on numpy arrays for testability.
The MetalSegmenter class wraps these for use with Slicer volume nodes.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def threshold_volume(volume: np.ndarray, threshold: float = 2500) -> np.ndarray:
    """Threshold a CT volume to isolate high-intensity voxels.

    Example::

        mask = threshold_volume(ct_array, threshold=2500)
    """
    return (volume >= threshold).astype(np.uint8)


def compute_head_mask(volume: np.ndarray) -> np.ndarray:
    """Create a binary mask of the patient head by excluding air voxels.

    Thresholds at -200 HU (distinguishes tissue/bone from air), fills internal
    air pockets (sinuses, ventricles), and keeps the largest connected component
    to exclude anything outside the patient.

    Args:
        volume: CT array in HU.

    Returns:
        Binary uint8 mask with 1 inside the patient head.

    Example::

        head_mask = compute_head_mask(ct_array)
    """
    rough = (volume > -200).astype(np.uint8)
    filled = ndimage.binary_fill_holes(rough)
    labeled, n = ndimage.label(filled)
    if n == 0:
        return rough
    sizes = ndimage.sum(filled, labeled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return (labeled == largest).astype(np.uint8)


def cleanup_metal_mask(
    mask: np.ndarray,
    head_mask: np.ndarray | None = None,
    min_component_size: int = 5,
    min_elongation_size: int = 30,
    min_elongation_ratio: float = 5.0,
) -> np.ndarray:
    """Remove noise and false positives from a binary metal mask.

    Steps:
    1. Apply head mask to exclude external metal (headframes, dental outside skull).
    2. Connected component analysis.
    3. Remove tiny components (< min_component_size voxels).
    4. For medium/large components (>= min_elongation_size voxels), require high
       elongation (longest_bbox_axis / shortest_bbox_axis >= min_elongation_ratio).
       SEEG electrodes have ratio ~15-30; dental implants ~3-4.
       Small components are kept unconditionally to preserve fragments of
       gapped electrodes that will be merged by the detector.

    Args:
        mask: Binary mask (0/1 uint8).
        head_mask: Optional binary head mask from compute_head_mask(). When
            provided, metal outside the head is removed before analysis.
        min_component_size: Minimum voxel count to keep any component.
        min_elongation_size: Components >= this size must pass the elongation check.
        min_elongation_ratio: Minimum elongation for medium/large components.

    Example::

        head_mask = compute_head_mask(ct_array)
        cleaned = cleanup_metal_mask(binary_mask, head_mask=head_mask)
    """
    if head_mask is not None:
        mask = (mask & head_mask).astype(np.uint8)

    struct = ndimage.generate_binary_structure(3, 1)
    labeled, num_features = ndimage.label(mask, structure=struct)

    result = np.zeros_like(mask)
    for comp_id in range(1, num_features + 1):
        component = labeled == comp_id
        volume = int(np.sum(component))

        if volume < min_component_size:
            continue

        if volume >= min_elongation_size:
            coords = np.argwhere(component)
            bbox_size = coords.max(axis=0) - coords.min(axis=0) + 1
            sorted_dims = np.sort(bbox_size)
            elongation = (
                sorted_dims[-1] / sorted_dims[0] if sorted_dims[0] > 0 else 1.0
            )
            if elongation < min_elongation_ratio:
                continue

        result[component] = 1

    return result
```

**Step 4: Run all metal segmenter tests**

```bash
.venv/bin/pytest tests/test_metal_segmenter.py -v
```

Expected: all pass (including the two pre-existing test classes)

**Step 5: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/metal_segmenter.py tests/test_metal_segmenter.py
git commit -m "feat: add head mask and fix elongation filter in metal segmenter"
```

---

## Task 2: Fold segmentation into `ElectrodeDetector.detect_all`

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py`
- Modify: `tests/test_electrode_detector.py`

The `detect_all` Slicer wrapper currently takes `(metal_volume_node, ct_volume_node)`. Change it to `(ct_volume_node, threshold=2500)`, computing head mask + cleanup internally.

**Step 1: Write failing test**

Add to `tests/test_electrode_detector.py` (the existing numpy-level tests are unaffected):

```python
class TestDetectElectrodesWithNoise:
    """Verify detect_electrodes ignores non-electrode metal coords."""

    def _make_electrode(self, start, direction, n_contacts, spacing=3.5):
        direction = np.array(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        points = []
        for i in range(n_contacts):
            center = np.array(start) + i * spacing * direction
            for _ in range(5):
                points.append(center + np.random.randn(3) * 0.3)
        return np.array(points)

    def test_ignores_bulky_noise_cluster(self):
        """A bulky cluster of metal coords (bone-like) should produce no electrode."""
        np.random.seed(0)
        # Electrode
        e1 = self._make_electrode([0, 0, 0], [1, 0, 0], 8)
        # Bulky noise (e.g. residual bone) — spread in all directions equally
        noise = np.random.uniform(-5, 5, (200, 3)) + np.array([0, 50, 0])
        all_coords = np.vstack([e1, noise])
        electrodes = detect_electrodes(all_coords)
        # Only the real electrode should be found
        assert len(electrodes) == 1
        assert electrodes[0].num_contacts == 8
```

**Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_electrode_detector.py::TestDetectElectrodesWithNoise -v
```

Expected: FAILED (bulky noise cluster currently gets through)

**Step 3: No code change needed here**

The `detect_electrodes` function already has a `len(cluster) < 5` guard but the bulky noise will still be processed. Looking at the pipeline: `cluster_into_electrodes` groups it into one cluster, `fit_electrode_axis` + `detect_contacts_along_axis` won't find regular spacing, so `len(contact_positions) < min_contacts` will reject it. If the test passes already, that's fine — move on. If it fails, check what `detect_contacts_along_axis` returns for random noise (it should find no clear peaks).

**Step 4: Update `ElectrodeDetector.detect_all` signature**

In `electrode_detector.py`, replace the `detect_all` method:

```python
def detect_all(self, ct_volume_node, threshold: float = 2500) -> list[Electrode]:
    """From a CT volume, segment metal and find all electrodes and their contacts.

    Applies head masking and improved shape filtering internally — no separate
    segmentation step is required.

    Args:
        ct_volume_node: vtkMRMLScalarVolumeNode with the post-implant CT.
        threshold: HU threshold for metal (default 2500).

    Returns:
        List of Electrode objects.
    """
    from slicer.util import arrayFromVolume
    from SEEGFellowLib.metal_segmenter import (
        compute_head_mask,
        threshold_volume,
        cleanup_metal_mask,
    )

    ct_array = arrayFromVolume(ct_volume_node)
    head_mask = compute_head_mask(ct_array)
    metal_mask = threshold_volume(ct_array, threshold)
    cleaned = cleanup_metal_mask(metal_mask, head_mask=head_mask)

    ijk_to_ras = self._get_ijk_to_ras_matrix(ct_volume_node)

    ijk_coords = np.argwhere(cleaned > 0).astype(float)
    if len(ijk_coords) == 0:
        return []

    ones = np.ones((len(ijk_coords), 1))
    ijk_h = np.hstack([ijk_coords, ones])
    ras_h = (ijk_to_ras @ ijk_h.T).T
    ras_coords = ras_h[:, :3]

    return detect_electrodes(
        ras_coords,
        min_contacts=self.min_contacts,
        expected_spacing=self.expected_spacing,
        collinearity_tolerance=self.collinearity_tolerance,
        gap_ratio_threshold=self.gap_ratio_threshold,
    )
```

**Step 5: Run all electrode detector tests**

```bash
.venv/bin/pytest tests/test_electrode_detector.py -v
```

Expected: all pass

**Step 6: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py tests/test_electrode_detector.py
git commit -m "feat: fold metal segmentation into ElectrodeDetector.detect_all"
```

---

## Task 3: Update UI and logic (remove segmentation step, add threshold slider + live count)

**Files:**
- Modify: `SEEGFellow/SEEGFellow/Resources/UI/SEEGFellow.ui`
- Modify: `SEEGFellow/SEEGFellow/SEEGFellow.py`

No new tests are needed for widget code (Slicer-only, not unit-testable).

**Step 1: Update the `.ui` file**

Changes:
- Remove the entire Step 4 Metal Segmentation `ctkCollapsibleButton` block
- Rename Step 5 → Step 4, Step 6 → Step 5 in the text labels
- Add threshold slider + voxel count label to the detection panel

The detection panel section should become:

```xml
<!-- Step 4: Electrode Detection -->
<item>
 <widget class="ctkCollapsibleButton" name="detectionCollapsibleButton">
  <property name="text"><string>Step 4: Electrode Detection</string></property>
  <property name="collapsed"><bool>true</bool></property>
  <layout class="QVBoxLayout" name="detectionLayout">
   <item>
    <layout class="QFormLayout">
     <item row="0" column="0"><widget class="QLabel"><property name="text"><string>HU Threshold:</string></property></widget></item>
     <item row="0" column="1"><widget class="ctkSliderWidget" name="thresholdSlider"><property name="minimum"><double>500</double></property><property name="maximum"><double>4000</double></property><property name="value"><double>2500</double></property><property name="singleStep"><double>50</double></property></widget></item>
     <item row="1" column="1"><widget class="QLabel" name="voxelCountLabel"><property name="text"><string>—</string></property></widget></item>
    </layout>
   </item>
   <item><widget class="QPushButton" name="detectElectrodesButton"><property name="text"><string>Detect All Electrodes</string></property></widget></item>
   <item><widget class="QTableWidget" name="electrodeTable"/></item>
   <item><widget class="QPushButton" name="applyNamesButton"><property name="text"><string>Apply Names</string></property></widget></item>
  </layout>
 </widget>
</item>
```

Also rename Manual Fallback from Step 6 → Step 5 in the `text` property.

**Step 2: Update `SEEGFellowLogic` in `SEEGFellow.py`**

- Remove `_metal_labelmap` attribute and `run_metal_segmentation` method.
- Update `run_electrode_detection` to accept `threshold`:

```python
def run_electrode_detection(self, threshold: float = 2500) -> None:
    """Run full automated electrode detection pipeline.

    Example::

        logic.run_electrode_detection(threshold=2500)
    """
    from SEEGFellowLib.electrode_detector import ElectrodeDetector

    detector = ElectrodeDetector()
    self.electrodes = detector.detect_all(self._ct_node, threshold=threshold)
    self._create_fiducials_for_electrodes()
```

Also remove from `__init__`: `self._metal_labelmap = None`
And remove the `cleanup` method's reference to `_metal_labelmap` if present (it's not currently there).

**Step 3: Update `SEEGFellowWidget` in `SEEGFellow.py`**

Remove all Step 4 segmentation wiring:
```python
# DELETE these three lines from setup():
self.ui.segmentMetalButton.clicked.connect(self._on_segment_metal_clicked)
self.ui.acceptSegmentationButton.clicked.connect(self._on_accept_segmentation_clicked)
self.ui.adjustSegmentationButton.clicked.connect(self._on_segment_metal_clicked)
```

Remove the handlers `_on_segment_metal_clicked` and `_on_accept_segmentation_clicked`.

Update `_on_accept_registration_clicked` to open the detection panel instead of segmentation:
```python
def _on_accept_registration_clicked(self):
    self.ui.detectionCollapsibleButton.collapsed = False
```

Add to `setup()` (after other Step 4 wiring):
```python
# Step 4: Electrode Detection
self.ui.thresholdSlider.valueChanged.connect(self._on_threshold_changed)
```

Add the new handler:
```python
def _on_threshold_changed(self, value: float) -> None:
    ct_node = self.logic._ct_node
    if ct_node is None:
        self.ui.voxelCountLabel.setText("—")
        return
    import numpy as np
    from slicer.util import arrayFromVolume
    ct_array = arrayFromVolume(ct_node)
    count = int(np.sum(ct_array >= value))
    self.ui.voxelCountLabel.setText(f"{count:,} voxels above threshold")
```

Update `_on_detect_electrodes_clicked` to pass the threshold:
```python
def _on_detect_electrodes_clicked(self):
    threshold = self.ui.thresholdSlider.value
    try:
        slicer.util.showStatusMessage("Detecting electrodes...")
        self.logic.run_electrode_detection(threshold)
        self._populate_electrode_table()
        slicer.util.showStatusMessage(
            f"Detected {len(self.logic.electrodes)} electrode(s)."
        )
        self.ui.electrodeListCollapsibleButton.collapsed = False
    except Exception as e:
        slicer.util.errorDisplay(f"Electrode detection failed: {e}")
```

**Step 4: Run full test suite**

```bash
.venv/bin/pytest tests/ -v
```

Expected: all 30+ tests pass

**Step 5: Commit**

```bash
git add SEEGFellow/SEEGFellow/Resources/UI/SEEGFellow.ui SEEGFellow/SEEGFellow/SEEGFellow.py
git commit -m "feat: remove segmentation step, add threshold slider with live voxel count to detection panel"
```
