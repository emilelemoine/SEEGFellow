# LoG-Based Contact Center Detection — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the monolithic threshold + shape-heuristic pipeline with LoG blob detection for contact center localization, producing an interactive 3-sub-step segmentation workflow.

**Architecture:** Steps 4a (intracranial mask), 4b (metal threshold), 4c (LoG contact centers) each produce a visible, editable result. `detect_electrodes` switches from raw-voxel input to pre-computed (N,3) contact center coordinates. UI gets sub-panels with "Edit in Segment Editor" buttons and a sigma slider.

**Tech Stack:** numpy, scipy (`ndimage.gaussian_laplace`, `ndimage.minimum_filter`), 3D Slicer (segmentation nodes, Segment Editor integration)

---

### Task 1: Add `detect_contact_centers` to `metal_segmenter.py` (test-first)

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/metal_segmenter.py:1-108`
- Modify: `tests/test_metal_segmenter.py:1-139`

**Step 1: Write the failing test**

Add to `tests/test_metal_segmenter.py`:

```python
from SEEGFellowLib.metal_segmenter import detect_contact_centers


class TestDetectContactCenters:
    def test_finds_isolated_blobs(self):
        """Three bright blobs at known positions should produce three centers."""
        ct = np.zeros((60, 60, 60), dtype=np.float32)
        metal_mask = np.zeros((60, 60, 60), dtype=np.uint8)
        # Place 3 bright spots (simulating contacts) at known IJK positions
        centers_ijk = [(15, 30, 30), (25, 30, 30), (35, 30, 30)]
        for ci, cj, ck in centers_ijk:
            # Gaussian-like blob: bright center, fading neighbours
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    for dk in range(-2, 3):
                        dist = (di**2 + dj**2 + dk**2) ** 0.5
                        ct[ci + di, cj + dj, ck + dk] = 3000.0 * max(0, 1 - dist / 3)
                        if dist <= 2:
                            metal_mask[ci + di, cj + dj, ck + dk] = 1

        detected = detect_contact_centers(ct, metal_mask, sigma=1.2)
        assert detected.shape[1] == 3
        assert len(detected) == 3
        # Each detected center should be within 2 voxels of a true center
        for true_center in centers_ijk:
            dists = np.linalg.norm(detected - np.array(true_center), axis=1)
            assert dists.min() < 2.0, f"No detection near {true_center}"

    def test_rejects_blobs_outside_metal_mask(self):
        """A bright spot outside the metal mask should not be detected."""
        ct = np.zeros((40, 40, 40), dtype=np.float32)
        metal_mask = np.zeros((40, 40, 40), dtype=np.uint8)
        # Blob inside metal mask
        ct[10, 20, 20] = 3000.0
        metal_mask[9:12, 19:22, 19:22] = 1
        # Blob outside metal mask (e.g. bone)
        ct[30, 20, 20] = 3000.0
        # metal_mask stays 0 at (30,20,20)

        detected = detect_contact_centers(ct, metal_mask, sigma=1.2)
        assert len(detected) == 1
        assert abs(detected[0, 0] - 10) < 2.0

    def test_empty_mask_returns_empty(self):
        """An all-zero metal mask should return no centers."""
        ct = np.full((20, 20, 20), 3000.0, dtype=np.float32)
        metal_mask = np.zeros((20, 20, 20), dtype=np.uint8)
        detected = detect_contact_centers(ct, metal_mask, sigma=1.2)
        assert len(detected) == 0
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_metal_segmenter.py::TestDetectContactCenters -v`
Expected: FAIL with `ImportError: cannot import name 'detect_contact_centers'`

**Step 3: Write minimal implementation**

Add to `SEEGFellow/SEEGFellow/SEEGFellowLib/metal_segmenter.py`:

```python
def detect_contact_centers(
    ct_array: np.ndarray,
    metal_mask: np.ndarray,
    sigma: float = 1.2,
) -> np.ndarray:
    """Detect contact centers using Laplacian of Gaussian blob detection.

    Applies LoG at the given scale (sigma) to the CT array, finds local
    minima of the LoG response, and keeps only those within the metal mask.

    Args:
        ct_array: CT volume in HU (3D numpy array).
        metal_mask: Binary mask (1 = candidate metal voxel).
        sigma: LoG scale in voxels (typical 1.0–1.5 mm, adjust for voxel size).

    Returns:
        (N, 3) array of contact center positions in IJK coordinates.

    Example::

        centers_ijk = detect_contact_centers(ct_array, metal_mask, sigma=1.2)
    """
    if not np.any(metal_mask):
        return np.empty((0, 3))

    # LoG response (negative at blob centers for bright blobs)
    log_response = ndimage.gaussian_laplace(ct_array.astype(np.float64), sigma=sigma)

    # Local minima: voxel is smaller than all 26 neighbours
    neighbourhood_size = 2 * int(np.ceil(sigma)) + 1
    local_min = ndimage.minimum_filter(log_response, size=neighbourhood_size)
    is_local_min = (log_response == local_min) & (log_response < 0)

    # Keep only minima that fall within the metal mask
    candidates = is_local_min & (metal_mask > 0)

    centers = np.argwhere(candidates)
    return centers
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_metal_segmenter.py::TestDetectContactCenters -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add tests/test_metal_segmenter.py SEEGFellow/SEEGFellow/SEEGFellowLib/metal_segmenter.py
git commit -m "feat: add LoG-based detect_contact_centers"
```

---

### Task 2: Refactor `detect_electrodes` to accept contact center coordinates

The current `detect_electrodes` takes raw metal voxel coordinates (many voxels per contact) and internally detects contacts via 1D histogram peak-finding (`detect_contacts_along_axis`). With LoG, each input point IS a contact center, so the flow simplifies: cluster centers into electrodes, fit axis, read off contact positions directly.

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py:342-427`
- Modify: `tests/test_electrode_detector.py:192-248`

**Step 1: Write updated tests for `detect_electrodes`**

Replace `TestDetectElectrodes` and `TestDetectElectrodesWithNoise` in `tests/test_electrode_detector.py`. The new tests pass contact center coordinates (one point per contact) instead of raw voxel blobs:

```python
class TestDetectElectrodes:
    def _make_centers(self, start, direction, n_contacts, spacing=3.5):
        """Create synthetic contact centers: one point per contact."""
        direction = np.array(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        return np.array([np.array(start) + i * spacing * direction
                         for i in range(n_contacts)])

    def test_detects_two_electrodes(self):
        np.random.seed(42)
        e1 = self._make_centers([0, 0, 0], [1, 0, 0], 8)
        e2 = self._make_centers([0, 50, 0], [0, 1, 0], 6)
        all_centers = np.vstack([e1, e2])

        electrodes = detect_electrodes(all_centers)
        assert len(electrodes) == 2
        contact_counts = sorted([e.num_contacts for e in electrodes])
        assert contact_counts == [6, 8]

    def test_rejects_too_few_contacts(self):
        e1 = self._make_centers([0, 0, 0], [1, 0, 0], 2)
        electrodes = detect_electrodes(e1, min_contacts=3)
        assert len(electrodes) == 0


class TestDetectElectrodesWithNoise:
    def _make_centers(self, start, direction, n_contacts, spacing=3.5):
        direction = np.array(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        return np.array([np.array(start) + i * spacing * direction
                         for i in range(n_contacts)])

    def test_ignores_scattered_noise(self):
        """Random scattered points should not produce an electrode."""
        np.random.seed(0)
        e1 = self._make_centers([0, 0, 0], [1, 0, 0], 8)
        noise = np.random.uniform(-5, 5, (20, 3)) + np.array([0, 50, 0])
        all_centers = np.vstack([e1, noise])
        electrodes = detect_electrodes(all_centers)
        assert len(electrodes) == 1
        assert electrodes[0].num_contacts == 8
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_electrode_detector.py::TestDetectElectrodes tests/test_electrode_detector.py::TestDetectElectrodesWithNoise -v`
Expected: FAIL (the new tests call `detect_electrodes` with center coords, but the old implementation expects dense voxel clouds and uses `detect_contacts_along_axis` internally)

**Step 3: Rewrite `detect_electrodes`**

Replace the `detect_electrodes` function in `electrode_detector.py`. The new version treats each input point as a contact center:

```python
def detect_electrodes(
    contact_centers: np.ndarray,
    min_contacts: int = 3,
    expected_spacing: float = 3.5,
    collinearity_tolerance: float = 10.0,
    gap_ratio_threshold: float = 1.8,
) -> list[Electrode]:
    """Full detection pipeline from pre-detected contact centers.

    Each row in contact_centers is one contact center in RAS coordinates.
    Steps: cluster -> merge collinear -> fit axis -> orient -> build Electrode.

    Args:
        contact_centers: (N, 3) array of contact center RAS coordinates.
        min_contacts: Minimum contacts to accept an electrode candidate.
        expected_spacing: Expected contact spacing in mm.
        collinearity_tolerance: Max angle for merging collinear fragments.
        gap_ratio_threshold: Threshold for gap detection.

    Returns:
        List of Electrode objects with auto-numbered contacts (unnamed).

    Example::

        electrodes = detect_electrodes(contact_centers_ras)
    """
    # 1. Cluster centers into electrode candidates
    clusters = cluster_into_electrodes(contact_centers, distance_threshold=expected_spacing * 2)

    # 2. Merge collinear fragments (for gapped electrodes)
    clusters = merge_collinear_clusters(
        clusters, angle_tolerance=collinearity_tolerance
    )

    electrodes = []
    for cluster in clusters:
        if len(cluster) < min_contacts:
            continue

        # 3. Fit axis through centers
        center, direction = fit_electrode_axis(cluster)

        # 4. Project centers onto axis — each projection IS a contact position
        projections = np.dot(cluster - center, direction)
        sorted_indices = np.argsort(projections)
        sorted_projections = projections[sorted_indices]

        # 5. Analyze spacing
        spacing_info = analyze_spacing(sorted_projections, gap_ratio_threshold)

        # 6. Orient deepest first
        sorted_positions, oriented_dir = orient_deepest_first(
            sorted_projections, center, direction
        )

        # 7. Build Electrode
        params = ElectrodeParams(
            contact_length=2.0,
            contact_spacing=spacing_info["contact_spacing"],
            contact_diameter=0.8,
            gap_spacing=spacing_info["gap_spacing"],
            contacts_per_group=spacing_info["contacts_per_group"],
        )

        contacts = []
        for i, pos in enumerate(sorted_positions):
            ras = center + pos * oriented_dir
            contacts.append(Contact(index=i + 1, position_ras=tuple(ras)))

        electrode = Electrode(
            name="",
            params=params,
            contacts=contacts,
            trajectory_direction=tuple(oriented_dir),
        )
        electrodes.append(electrode)

    return electrodes
```

Key changes from the old version:
- `distance_threshold` for clustering is now `expected_spacing * 2` (contacts are individual points spaced ~3.5 mm apart, not dense voxel clouds)
- Removed `detect_contacts_along_axis` call — each input point is already a contact center
- Removed the `len(cluster) < 5` voxel check — replaced by `len(cluster) < min_contacts`
- Removed the implausible-spacing rejection (was for histogram artefacts, not applicable here)

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_electrode_detector.py::TestDetectElectrodes tests/test_electrode_detector.py::TestDetectElectrodesWithNoise -v`
Expected: PASS

**Step 5: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py tests/test_electrode_detector.py
git commit -m "refactor: detect_electrodes takes contact centers instead of raw voxels"
```

---

### Task 3: Update `cluster_into_electrodes` tests for center-point input

The existing `TestClusterIntoElectrodes` uses dense voxel clouds (`contact_voxels=3`). Update to use single-point-per-contact input consistent with the new pipeline.

**Files:**
- Modify: `tests/test_electrode_detector.py:36-63`

**Step 1: Update the tests**

```python
class TestClusterIntoElectrodes:
    def _make_centers(self, start, direction, n_contacts, spacing=3.5):
        """Create synthetic contact centers: one point per contact."""
        direction = np.array(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        return np.array([np.array(start) + i * spacing * direction
                         for i in range(n_contacts)])

    def test_two_separate_electrodes(self):
        e1 = self._make_centers([0, 0, 0], [1, 0, 0], 8)
        e2 = self._make_centers([0, 50, 0], [0, 1, 0], 6)
        all_centers = np.vstack([e1, e2])
        clusters = cluster_into_electrodes(all_centers, distance_threshold=7.0)
        assert len(clusters) == 2

    def test_single_electrode(self):
        e1 = self._make_centers([0, 0, 0], [1, 0, 0], 10)
        clusters = cluster_into_electrodes(e1, distance_threshold=7.0)
        assert len(clusters) == 1
```

**Step 2: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_electrode_detector.py::TestClusterIntoElectrodes -v`
Expected: PASS (the function itself hasn't changed, just the test inputs)

**Step 3: Commit**

```bash
git add tests/test_electrode_detector.py
git commit -m "test: update clustering tests for center-point input"
```

---

### Task 4: Remove dead code (`extract_metal_coords`, `detect_contacts_along_axis`, `cleanup_metal_mask`)

These functions are no longer used in the new pipeline:
- `extract_metal_coords` — replaced by LoG + IJK-to-RAS conversion
- `detect_contacts_along_axis` — replaced by treating each center as a contact
- `cleanup_metal_mask` — replaced by LoG + Segment Editor manual correction

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py` (remove `extract_metal_coords`, `detect_contacts_along_axis`)
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/metal_segmenter.py` (remove `cleanup_metal_mask`)
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/__init__.py` (remove `cleanup_metal_mask` from imports)
- Modify: `tests/test_electrode_detector.py` (remove `TestExtractMetalCoords`, `TestDetectContactsAlongAxis`, and their imports)
- Modify: `tests/test_metal_segmenter.py` (remove `TestCleanupMetalMask`, `TestCleanupMetalMaskImproved`, and their imports)

**Step 1: Remove `extract_metal_coords` and `detect_contacts_along_axis` from electrode_detector.py**

Delete lines 15–41 (`extract_metal_coords`) and lines 119–162 (`detect_contacts_along_axis`) from `electrode_detector.py`.

**Step 2: Remove `cleanup_metal_mask` from metal_segmenter.py**

Delete lines 51–107 (`cleanup_metal_mask`) from `metal_segmenter.py`.

**Step 3: Update `__init__.py`**

Remove `cleanup_metal_mask` from the import in `SEEGFellowLib/__init__.py`.

**Step 4: Remove dead tests**

In `tests/test_electrode_detector.py`:
- Remove the import of `extract_metal_coords` (line 11)
- Remove `TestExtractMetalCoords` class (lines 16-33)
- Remove the import of `detect_contacts_along_axis` (line 67)
- Remove `TestDetectContactsAlongAxis` class (lines 97-122)

In `tests/test_metal_segmenter.py`:
- Remove the import of `cleanup_metal_mask` (line 10)
- Remove `TestCleanupMetalMaskImproved` class (lines 54-100)
- Remove `TestCleanupMetalMask` class (lines 103-138)

**Step 5: Run all tests to verify nothing breaks**

Run: `.venv/bin/pytest tests/ -v`
Expected: All remaining tests PASS

**Step 6: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py SEEGFellow/SEEGFellow/SEEGFellowLib/metal_segmenter.py SEEGFellow/SEEGFellow/SEEGFellowLib/__init__.py tests/test_electrode_detector.py tests/test_metal_segmenter.py
git commit -m "refactor: remove dead code (extract_metal_coords, detect_contacts_along_axis, cleanup_metal_mask)"
```

---

### Task 5: Rewire `ElectrodeDetector.detect_all` to use new pipeline

The Slicer wrapper class currently calls `cleanup_metal_mask` + `extract_metal_coords` (via `np.argwhere`) + old `detect_electrodes`. Rewire it to: head mask -> threshold -> LoG contact centers -> new `detect_electrodes`.

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py` (the `ElectrodeDetector.detect_all` method)

**Step 1: Rewrite `detect_all`**

```python
def detect_all(self, ct_volume_node, threshold: float = 2500, sigma: float = 1.2) -> list[Electrode]:
    """From a CT volume, detect all electrodes using LoG contact center detection.

    Pipeline: head mask -> threshold within mask -> LoG blob detection -> electrode grouping.

    Args:
        ct_volume_node: vtkMRMLScalarVolumeNode with the post-implant CT.
        threshold: HU threshold for metal (default 2500).
        sigma: LoG scale in voxels for contact detection.

    Returns:
        List of Electrode objects.

    Example::

        detector = ElectrodeDetector()
        electrodes = detector.detect_all(ct_volume_node, threshold=2500, sigma=1.2)
    """
    from slicer.util import arrayFromVolume
    from SEEGFellowLib.metal_segmenter import (
        compute_head_mask,
        threshold_volume,
        detect_contact_centers,
    )

    ct_array = arrayFromVolume(ct_volume_node)
    head_mask = compute_head_mask(ct_array)
    metal_mask = threshold_volume(ct_array, threshold) & head_mask

    # LoG blob detection for contact centers (IJK coordinates)
    centers_ijk = detect_contact_centers(ct_array, metal_mask, sigma=sigma)
    if len(centers_ijk) == 0:
        return []

    # Convert IJK to RAS
    ijk_to_ras = self._get_ijk_to_ras_matrix(ct_volume_node)
    ones = np.ones((len(centers_ijk), 1))
    ijk_h = np.hstack([centers_ijk.astype(float), ones])
    ras_h = (ijk_to_ras @ ijk_h.T).T
    centers_ras = ras_h[:, :3]

    return detect_electrodes(
        centers_ras,
        min_contacts=self.min_contacts,
        expected_spacing=self.expected_spacing,
        collinearity_tolerance=self.collinearity_tolerance,
        gap_ratio_threshold=self.gap_ratio_threshold,
    )
```

**Step 2: Run all tests**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests PASS (the wrapper is only called from Slicer, existing pure-function tests are unaffected)

**Step 3: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py
git commit -m "feat: rewire ElectrodeDetector.detect_all to use LoG pipeline"
```

---

### Task 6: Update `SEEGFellowLogic` to pass sigma and expose intermediate results

The Logic class needs to store the head mask and metal mask as segmentation nodes (for steps 4a/4b), and pass sigma to the detector for step 4c.

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellow.py:295-391`

**Step 1: Add segmentation node management and sigma parameter**

In `SEEGFellowLogic.__init__`, add:
```python
self._segmentation_node = None  # for head mask + metal mask segments
```

Add method `run_intracranial_mask`:
```python
def run_intracranial_mask(self) -> None:
    """Compute intracranial mask and display it as a segmentation segment.

    Example::

        logic.run_intracranial_mask()
    """
    from slicer.util import arrayFromVolume, updateVolumeFromArray
    from SEEGFellowLib.metal_segmenter import compute_head_mask
    import vtk

    ct_array = arrayFromVolume(self._ct_node)
    head_mask = compute_head_mask(ct_array)

    # Create or reuse segmentation node
    if self._segmentation_node is None:
        self._segmentation_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode", "SEEGFellow Segmentation"
        )
        self._segmentation_node.CreateDefaultDisplayNodes()
        self._segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(
            self._ct_node
        )

    # Add "Intracranial" segment
    seg = self._segmentation_node.GetSegmentation()
    if seg.GetSegmentIdBySegmentName("Intracranial"):
        seg.RemoveSegment(seg.GetSegmentIdBySegmentName("Intracranial"))

    segment_id = seg.AddEmptySegment("Intracranial", "Intracranial")
    segment = seg.GetSegment(segment_id)
    segment.SetColor(0.0, 0.5, 1.0)  # blue

    # Write mask into segment as labelmap
    import slicer.util
    slicer.util.updateSegmentBinaryLabelmapFromArray(
        head_mask, self._segmentation_node, segment_id, self._ct_node
    )
    self._head_mask = head_mask
```

Add method `run_metal_threshold`:
```python
def run_metal_threshold(self, threshold: float = 2500) -> None:
    """Threshold CT within intracranial mask and display as a segment.

    Example::

        logic.run_metal_threshold(threshold=2500)
    """
    from slicer.util import arrayFromVolume
    from SEEGFellowLib.metal_segmenter import threshold_volume

    ct_array = arrayFromVolume(self._ct_node)
    metal_mask = threshold_volume(ct_array, threshold)
    if hasattr(self, '_head_mask') and self._head_mask is not None:
        metal_mask = metal_mask & self._head_mask

    seg = self._segmentation_node.GetSegmentation()
    if seg.GetSegmentIdBySegmentName("Metal"):
        seg.RemoveSegment(seg.GetSegmentIdBySegmentName("Metal"))

    segment_id = seg.AddEmptySegment("Metal", "Metal")
    segment = seg.GetSegment(segment_id)
    segment.SetColor(1.0, 1.0, 0.0)  # yellow

    import slicer.util
    slicer.util.updateSegmentBinaryLabelmapFromArray(
        metal_mask, self._segmentation_node, segment_id, self._ct_node
    )
    self._metal_mask = metal_mask
```

Modify `run_electrode_detection` to accept sigma:
```python
def run_electrode_detection(self, threshold: float = 2500, sigma: float = 1.2) -> None:
    """Run full automated electrode detection pipeline.

    Example::

        logic.run_electrode_detection(threshold=2500, sigma=1.2)
    """
    from SEEGFellowLib.electrode_detector import ElectrodeDetector

    detector = ElectrodeDetector()
    self.electrodes = detector.detect_all(self._ct_node, threshold=threshold, sigma=sigma)
    self._create_fiducials_for_electrodes()
```

**Step 2: Run all tests**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests PASS (Logic class is only tested through Slicer)

**Step 3: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellow.py
git commit -m "feat: add intracranial mask and metal threshold segmentation to Logic"
```

---

### Task 7: Update UI — sub-panels for 4a, 4b, 4c with Segment Editor buttons

**Files:**
- Modify: `SEEGFellow/SEEGFellow/Resources/UI/SEEGFellow.ui`
- Modify: `SEEGFellow/SEEGFellow/SEEGFellow.py` (widget `setup` and new handlers)

**Step 1: Update `.ui` file**

Replace the Step 4 `detectionCollapsibleButton` with three sub-steps:

```xml
<!-- Step 4a: Intracranial Mask -->
<item>
 <widget class="ctkCollapsibleButton" name="intracranialMaskCollapsibleButton">
  <property name="text"><string>Step 4a: Intracranial Mask</string></property>
  <property name="collapsed"><bool>true</bool></property>
  <layout class="QVBoxLayout" name="intracranialMaskLayout">
   <item><widget class="QPushButton" name="computeHeadMaskButton"><property name="text"><string>Compute Intracranial Mask</string></property></widget></item>
   <item>
    <layout class="QHBoxLayout">
     <item><widget class="QPushButton" name="editHeadMaskButton"><property name="text"><string>Edit in Segment Editor</string></property></widget></item>
     <item><widget class="QPushButton" name="acceptHeadMaskButton"><property name="text"><string>Accept</string></property></widget></item>
    </layout>
   </item>
  </layout>
 </widget>
</item>

<!-- Step 4b: Metal Threshold -->
<item>
 <widget class="ctkCollapsibleButton" name="metalThresholdCollapsibleButton">
  <property name="text"><string>Step 4b: Metal Threshold</string></property>
  <property name="collapsed"><bool>true</bool></property>
  <layout class="QVBoxLayout" name="metalThresholdLayout">
   <item>
    <layout class="QFormLayout">
     <item row="0" column="0"><widget class="QLabel"><property name="text"><string>HU Threshold:</string></property></widget></item>
     <item row="0" column="1"><widget class="ctkSliderWidget" name="thresholdSlider"><property name="minimum"><double>500</double></property><property name="maximum"><double>4000</double></property><property name="value"><double>2500</double></property><property name="singleStep"><double>50</double></property></widget></item>
     <item row="1" column="1"><widget class="QLabel" name="voxelCountLabel"><property name="text"><string>—</string></property></widget></item>
    </layout>
   </item>
   <item><widget class="QPushButton" name="applyMetalThresholdButton"><property name="text"><string>Apply Threshold</string></property></widget></item>
   <item>
    <layout class="QHBoxLayout">
     <item><widget class="QPushButton" name="editMetalMaskButton"><property name="text"><string>Edit in Segment Editor</string></property></widget></item>
     <item><widget class="QPushButton" name="acceptMetalMaskButton"><property name="text"><string>Accept</string></property></widget></item>
    </layout>
   </item>
  </layout>
 </widget>
</item>

<!-- Step 4c: Contact Detection (LoG) -->
<item>
 <widget class="ctkCollapsibleButton" name="contactDetectionCollapsibleButton">
  <property name="text"><string>Step 4c: Detect Contacts (LoG)</string></property>
  <property name="collapsed"><bool>true</bool></property>
  <layout class="QVBoxLayout" name="contactDetectionLayout">
   <item>
    <layout class="QFormLayout">
     <item row="0" column="0"><widget class="QLabel"><property name="text"><string>Sigma (mm):</string></property></widget></item>
     <item row="0" column="1"><widget class="ctkSliderWidget" name="sigmaSlider"><property name="minimum"><double>0.5</double></property><property name="maximum"><double>3.0</double></property><property name="value"><double>1.2</double></property><property name="singleStep"><double>0.1</double></property></widget></item>
    </layout>
   </item>
   <item><widget class="QPushButton" name="detectElectrodesButton"><property name="text"><string>Detect All Electrodes</string></property></widget></item>
   <item><widget class="QTableWidget" name="electrodeTable"/></item>
   <item><widget class="QPushButton" name="applyNamesButton"><property name="text"><string>Apply Names</string></property></widget></item>
  </layout>
 </widget>
</item>
```

Remove the old `detectionCollapsibleButton` block entirely.

**Step 2: Wire up new buttons in widget `setup`**

In `SEEGFellowWidget.setup()`, replace the Step 4 connections with:

```python
# Step 4a: Intracranial Mask
self.ui.computeHeadMaskButton.clicked.connect(self._on_compute_head_mask_clicked)
self.ui.editHeadMaskButton.clicked.connect(self._on_edit_head_mask_clicked)
self.ui.acceptHeadMaskButton.clicked.connect(self._on_accept_head_mask_clicked)

# Step 4b: Metal Threshold
self.ui.thresholdSlider.valueChanged.connect(self._on_threshold_changed)
self.ui.applyMetalThresholdButton.clicked.connect(self._on_apply_metal_threshold_clicked)
self.ui.editMetalMaskButton.clicked.connect(self._on_edit_metal_mask_clicked)
self.ui.acceptMetalMaskButton.clicked.connect(self._on_accept_metal_mask_clicked)

# Step 4c: Contact Detection
self.ui.detectElectrodesButton.clicked.connect(self._on_detect_electrodes_clicked)
self.ui.applyNamesButton.clicked.connect(self._on_apply_names_clicked)
```

**Step 3: Add handler methods**

```python
# -------------------------------------------------------------------------
# Step 4a: Intracranial Mask
# -------------------------------------------------------------------------

def _on_compute_head_mask_clicked(self):
    try:
        slicer.util.showStatusMessage("Computing intracranial mask...")
        self.logic.run_intracranial_mask()
        slicer.util.showStatusMessage("Intracranial mask computed.")
    except Exception as e:
        slicer.util.errorDisplay(f"Failed to compute head mask: {e}")

def _on_edit_head_mask_clicked(self):
    if self.logic._segmentation_node is None:
        slicer.util.errorDisplay("Run 'Compute Intracranial Mask' first.")
        return
    slicer.util.selectModule("SegmentEditor")
    editor_widget = slicer.modules.segmenteditor.widgetRepresentation().self()
    editor_widget.setSegmentationNode(self.logic._segmentation_node)
    editor_widget.setSourceVolumeNode(self.logic._ct_node)

def _on_accept_head_mask_clicked(self):
    self.ui.metalThresholdCollapsibleButton.collapsed = False
    slicer.util.showStatusMessage("Head mask accepted.")

# -------------------------------------------------------------------------
# Step 4b: Metal Threshold
# -------------------------------------------------------------------------

def _on_apply_metal_threshold_clicked(self):
    threshold = self.ui.thresholdSlider.value
    try:
        slicer.util.showStatusMessage("Applying metal threshold...")
        self.logic.run_metal_threshold(threshold)
        slicer.util.showStatusMessage("Metal threshold applied.")
    except Exception as e:
        slicer.util.errorDisplay(f"Failed to apply threshold: {e}")

def _on_edit_metal_mask_clicked(self):
    if self.logic._segmentation_node is None:
        slicer.util.errorDisplay("Run 'Apply Threshold' first.")
        return
    slicer.util.selectModule("SegmentEditor")
    editor_widget = slicer.modules.segmenteditor.widgetRepresentation().self()
    editor_widget.setSegmentationNode(self.logic._segmentation_node)
    editor_widget.setSourceVolumeNode(self.logic._ct_node)

def _on_accept_metal_mask_clicked(self):
    self.ui.contactDetectionCollapsibleButton.collapsed = False
    slicer.util.showStatusMessage("Metal mask accepted.")
```

Update `_on_detect_electrodes_clicked` to pass sigma:

```python
def _on_detect_electrodes_clicked(self):
    threshold = self.ui.thresholdSlider.value
    sigma = self.ui.sigmaSlider.value
    try:
        slicer.util.showStatusMessage("Detecting electrodes...")
        self.logic.run_electrode_detection(threshold, sigma=sigma)
        self._populate_electrode_table()
        slicer.util.showStatusMessage(
            f"Detected {len(self.logic.electrodes)} electrode(s)."
        )
        self.ui.electrodeListCollapsibleButton.collapsed = False
    except Exception as e:
        slicer.util.errorDisplay(f"Electrode detection failed: {e}")
```

Update `_on_accept_registration_clicked` to open step 4a instead of old step 4:

```python
def _on_accept_registration_clicked(self):
    self.ui.intracranialMaskCollapsibleButton.collapsed = False
```

**Step 4: Run all tests**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add SEEGFellow/SEEGFellow/Resources/UI/SEEGFellow.ui SEEGFellow/SEEGFellow/SEEGFellow.py
git commit -m "feat: split detection into 4a/4b/4c sub-panels with Segment Editor integration"
```

---

### Task 8: Export `detect_contact_centers` from `__init__.py`

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/__init__.py`

**Step 1: Add import**

Add `detect_contact_centers` to the metal_segmenter import line:

```python
from SEEGFellowLib.metal_segmenter import (
    compute_head_mask,
    threshold_volume,
    detect_contact_centers,
)
```

**Step 2: Run all tests**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/__init__.py
git commit -m "chore: export detect_contact_centers from __init__"
```

---

### Task 9: Final verification

**Step 1: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests PASS

**Step 2: Verify no stale imports**

Run: `grep -rn "cleanup_metal_mask\|extract_metal_coords\|detect_contacts_along_axis" SEEGFellow/ tests/`
Expected: No matches

**Step 3: Review design doc alignment**

Verify the implementation matches the design doc `docs/plans/2026-02-26-log-contact-detection-design.md`:
- [x] `cleanup_metal_mask` removed
- [x] `detect_contact_centers` added to `metal_segmenter.py`
- [x] `detect_electrodes` takes (N,3) contact centers
- [x] `detect_contacts_along_axis` removed
- [x] `extract_metal_coords` removed
- [x] UI has 4a/4b/4c sub-panels
- [x] Segment Editor integration via "Edit" buttons
- [x] Sigma slider for 4c
- [x] Threshold slider in 4b
