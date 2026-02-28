# RANSAC Electrode Grouping Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace single-linkage electrode grouping with RANSAC line fitting that handles crossings, preserves LoG positions, and orients via brain centroid.

**Architecture:** Iterative RANSAC extracts one electrode at a time from the contact point cloud. Each iteration fits a line, claims inliers within a perpendicular distance tolerance, validates spacing, then removes them. Post-grouping uses PCA refit, brain-centroid orientation, and existing spacing analysis. All RANSAC parameters are exposed in the GUI.

**Tech Stack:** numpy, scipy (ndimage for existing code), 3D Slicer markups API

---

### Task 1: Write `ransac_group_contacts()` with TDD

**Files:**
- Create: `tests/test_ransac_grouping.py`
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py`

**Step 1: Write failing tests for `ransac_group_contacts`**

```python
# tests/test_ransac_grouping.py
import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow")
)

import numpy as np
import pytest


def _make_electrode(start, direction, n_contacts, spacing=3.5):
    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    return np.array(
        [np.array(start) + i * spacing * direction for i in range(n_contacts)]
    )


class TestRansacGroupContacts:
    def test_two_parallel_electrodes(self):
        """Two well-separated parallel electrodes are found."""
        from SEEGFellowLib.electrode_detector import ransac_group_contacts

        e1 = _make_electrode([0, 0, 0], [1, 0, 0], 8)
        e2 = _make_electrode([0, 50, 0], [1, 0, 0], 6)
        pool = np.vstack([e1, e2])

        groups = ransac_group_contacts(pool, expected_spacing=3.5)
        assert len(groups) == 2
        sizes = sorted([len(g) for g in groups])
        assert sizes == [6, 8]

    def test_crossing_electrodes(self):
        """Two electrodes that cross in 3D are correctly separated."""
        from SEEGFellowLib.electrode_detector import ransac_group_contacts

        # Electrode along X, centered at origin
        e1 = _make_electrode([-14, 0, 0], [1, 0, 0], 8)
        # Electrode along Y, crossing near origin
        e2 = _make_electrode([0, -10.5, 0], [0, 1, 0], 6)
        pool = np.vstack([e1, e2])

        groups = ransac_group_contacts(
            pool, expected_spacing=3.5, distance_tolerance=2.0
        )
        assert len(groups) == 2
        sizes = sorted([len(g) for g in groups])
        assert sizes == [6, 8]

    def test_gapped_electrode(self):
        """An electrode with a gap is found as one group."""
        from SEEGFellowLib.electrode_detector import ransac_group_contacts

        # 6 contacts, 24mm gap, 6 contacts — all along X
        g1 = _make_electrode([0, 0, 0], [1, 0, 0], 6)
        g2 = _make_electrode([41.5, 0, 0], [1, 0, 0], 6)  # gap of 24mm
        pool = np.vstack([g1, g2])

        groups = ransac_group_contacts(pool, expected_spacing=3.5)
        assert len(groups) == 1
        assert len(groups[0]) == 12

    def test_rejects_noise(self):
        """Scattered noise points are not grouped into an electrode."""
        from SEEGFellowLib.electrode_detector import ransac_group_contacts

        np.random.seed(42)
        noise = np.random.uniform(-50, 50, (20, 3))
        groups = ransac_group_contacts(noise, expected_spacing=3.5, min_contacts=3)
        # Noise should produce no valid electrodes (or very few)
        total_assigned = sum(len(g) for g in groups)
        assert total_assigned < 5

    def test_too_few_contacts_returns_empty(self):
        """Fewer contacts than min_contacts yields no groups."""
        from SEEGFellowLib.electrode_detector import ransac_group_contacts

        e1 = _make_electrode([0, 0, 0], [1, 0, 0], 2)
        groups = ransac_group_contacts(e1, min_contacts=3)
        assert groups == []

    def test_preserves_original_coordinates(self):
        """Returned coordinates must be the exact LoG positions, not reprojected."""
        from SEEGFellowLib.electrode_detector import ransac_group_contacts

        np.random.seed(42)
        e1 = _make_electrode([0, 0, 0], [1, 0, 0], 8)
        # Add small perpendicular noise
        e1 += np.random.randn(*e1.shape) * 0.3
        original = e1.copy()
        pool = e1

        groups = ransac_group_contacts(pool, expected_spacing=3.5)
        assert len(groups) == 1
        # Each returned point must be bitwise identical to an input point
        returned = groups[0]
        for pt in returned:
            assert any(np.array_equal(pt, orig) for orig in original)

    def test_deterministic_with_seed(self):
        """Same seed produces same result."""
        from SEEGFellowLib.electrode_detector import ransac_group_contacts

        e1 = _make_electrode([0, 0, 0], [1, 0, 0], 8)
        e2 = _make_electrode([0, 50, 0], [0, 1, 0], 6)
        pool = np.vstack([e1, e2])

        g1 = ransac_group_contacts(pool, expected_spacing=3.5, random_seed=42)
        g2 = ransac_group_contacts(pool, expected_spacing=3.5, random_seed=42)
        assert len(g1) == len(g2)
        for a, b in zip(
            sorted(g1, key=lambda x: len(x)),
            sorted(g2, key=lambda x: len(x)),
        ):
            np.testing.assert_array_equal(a, b)
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_ransac_grouping.py -v`
Expected: FAIL with `ImportError: cannot import name 'ransac_group_contacts'`

**Step 3: Implement `ransac_group_contacts`**

Add to `electrode_detector.py`, replacing `cluster_into_electrodes` and `merge_collinear_clusters`:

```python
def ransac_group_contacts(
    contact_coords: np.ndarray,
    expected_spacing: float = 3.5,
    distance_tolerance: float = 2.0,
    max_iterations: int = 1000,
    min_contacts: int = 3,
    spacing_low_factor: float = 0.5,
    spacing_high_factor: float = 2.0,
    random_seed: int | None = None,
) -> list[np.ndarray]:
    """Group contact centers into electrodes using iterative RANSAC line fitting.

    Repeatedly fits lines through the point cloud. Each iteration samples two
    points, finds all contacts within ``distance_tolerance`` perpendicular
    distance of that line, validates spacing, and claims the best-supported
    line's inliers. Claimed contacts are removed and the process repeats.

    Args:
        contact_coords: (N, 3) array of contact center RAS coordinates.
        expected_spacing: Expected contact spacing in mm.
        distance_tolerance: Max perpendicular distance (mm) to count as inlier.
        max_iterations: RANSAC trials per electrode.
        min_contacts: Minimum inliers to accept an electrode.
        spacing_low_factor: Lower bound on median spacing as fraction of
            expected_spacing (e.g. 0.5 means >= 1.75 mm for 3.5 mm spacing).
        spacing_high_factor: Upper bound on median spacing as fraction of
            expected_spacing (e.g. 2.0 means <= 7.0 mm for 3.5 mm spacing).
        random_seed: Optional seed for reproducibility.

    Returns:
        List of (M, 3) arrays, one per electrode. Coordinates are the original
        input values (not reprojected).

    Example::

        groups = ransac_group_contacts(centers_ras, expected_spacing=3.5)
    """
    if len(contact_coords) < min_contacts:
        return []

    rng = np.random.default_rng(random_seed)
    pool_indices = np.arange(len(contact_coords))
    coords = contact_coords.copy()
    groups: list[np.ndarray] = []

    spacing_lo = expected_spacing * spacing_low_factor
    spacing_hi = expected_spacing * spacing_high_factor

    while len(pool_indices) >= min_contacts:
        best_inlier_idx = None
        best_count = 0

        pool_coords = coords[pool_indices]

        for _ in range(max_iterations):
            # Sample 2 distinct points
            sample = rng.choice(len(pool_coords), size=2, replace=False)
            p1, p2 = pool_coords[sample[0]], pool_coords[sample[1]]
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length < 1e-8:
                continue
            direction /= length

            # Perpendicular distance of all pool points to line through p1
            diff = pool_coords - p1
            proj = np.dot(diff, direction)
            perp = diff - proj[:, np.newaxis] * direction
            dists = np.linalg.norm(perp, axis=1)

            inlier_mask = dists < distance_tolerance
            n_inliers = np.sum(inlier_mask)
            if n_inliers < min_contacts:
                continue

            # Validate spacing: median neighbor distance along the line
            inlier_proj = np.sort(proj[inlier_mask])
            spacings = np.diff(inlier_proj)
            if len(spacings) == 0:
                continue
            median_sp = np.median(spacings)
            if not (spacing_lo <= median_sp <= spacing_hi):
                continue

            if n_inliers > best_count:
                best_count = n_inliers
                best_inlier_idx = np.where(inlier_mask)[0]

        if best_inlier_idx is None:
            break

        # Refit on inliers and reclaim
        inlier_global = pool_indices[best_inlier_idx]
        inlier_coords = coords[inlier_global]
        center, direction = fit_electrode_axis(inlier_coords)
        diff = coords[pool_indices] - center
        proj = np.dot(diff, direction)
        perp = diff - proj[:, np.newaxis] * direction
        dists = np.linalg.norm(perp, axis=1)
        refined_mask = dists < distance_tolerance

        # Re-validate spacing after refit
        refined_proj = np.sort(proj[refined_mask])
        refined_spacings = np.diff(refined_proj)
        if len(refined_spacings) > 0:
            refined_median = np.median(refined_spacings)
            if spacing_lo <= refined_median <= spacing_hi:
                best_inlier_idx = np.where(refined_mask)[0]

        claimed_global = pool_indices[best_inlier_idx]
        groups.append(coords[claimed_global])
        pool_indices = np.setdiff1d(pool_indices, claimed_global)

    return groups
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_ransac_grouping.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```
git add tests/test_ransac_grouping.py SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py
git commit -m "feat: add ransac_group_contacts with TDD tests"
```

---

### Task 2: Update `orient_deepest_first` to use brain centroid

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py:228-253`
- Modify: `tests/test_electrode_detector.py` (add new tests)

**Step 1: Write failing tests**

Add to `tests/test_electrode_detector.py`:

```python
from SEEGFellowLib.electrode_detector import orient_deepest_first


class TestOrientDeepestFirst:
    def test_deepest_is_closest_to_brain_centroid(self):
        """Contact nearest to brain centroid should be index 0."""
        # Electrode along X from 10 to 40
        positions = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
        axis_origin = np.array([25.0, 0.0, 0.0])
        axis_direction = np.array([1.0, 0.0, 0.0])
        # Brain centroid at (5, 0, 0) — deepest end is at x=10
        brain_centroid = np.array([5.0, 0.0, 0.0])

        sorted_idx, oriented_dir = orient_deepest_first(
            positions, axis_origin, axis_direction, brain_centroid=brain_centroid
        )
        # First index should correspond to x=10 (closest to centroid)
        first_ras = axis_origin + positions[sorted_idx[0]] * axis_direction
        last_ras = axis_origin + positions[sorted_idx[-1]] * axis_direction
        assert np.linalg.norm(first_ras - brain_centroid) < np.linalg.norm(
            last_ras - brain_centroid
        )

    def test_already_correct_orientation(self):
        """If first contact is already deepest, no reversal needed."""
        positions = np.array([0.0, 3.5, 7.0])
        axis_origin = np.array([50.0, 0.0, 0.0])
        axis_direction = np.array([-1.0, 0.0, 0.0])
        # Brain centroid at (0, 0, 0) — x=50 maps to RAS (50,0,0),
        # positions[0]=0 → RAS = 50 + 0*(-1) = (50,0,0)
        # positions[2]=7.0 → RAS = 50 + 7*(-1) = (43,0,0) — closer to centroid
        brain_centroid = np.array([0.0, 0.0, 0.0])

        sorted_idx, oriented_dir = orient_deepest_first(
            positions, axis_origin, axis_direction, brain_centroid=brain_centroid
        )
        # Should reverse: positions[2] is deepest
        first_ras = axis_origin + positions[sorted_idx[0]] * axis_direction
        assert np.linalg.norm(first_ras - brain_centroid) < np.linalg.norm(
            (axis_origin + positions[sorted_idx[-1]] * axis_direction) - brain_centroid
        )

    def test_fallback_to_origin_when_no_centroid(self):
        """When brain_centroid is None, fall back to (0,0,0)."""
        positions = np.array([0.0, 3.5, 7.0])
        axis_origin = np.array([10.0, 0.0, 0.0])
        axis_direction = np.array([1.0, 0.0, 0.0])

        sorted_idx, _ = orient_deepest_first(
            positions, axis_origin, axis_direction, brain_centroid=None
        )
        # With origin (0,0,0): first contact at RAS (10,0,0), last at (17,0,0)
        # (10,0,0) is closer to origin → already correct, no reversal
        assert sorted_idx[0] == 0
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_electrode_detector.py::TestOrientDeepestFirst -v`
Expected: FAIL (signature mismatch — no `brain_centroid` param yet)

**Step 3: Update `orient_deepest_first`**

Replace lines 228–253 of `electrode_detector.py`:

```python
def orient_deepest_first(
    contact_positions: np.ndarray,
    axis_origin: np.ndarray,
    axis_direction: np.ndarray,
    brain_centroid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Orient contacts so index 0 = deepest (closest to brain center).

    Args:
        contact_positions: Sorted 1D array of projection positions along axis.
        axis_origin: 3D center of the electrode cluster (PCA mean).
        axis_direction: Unit vector along electrode axis.
        brain_centroid: 3D RAS coordinates of brain center. If None, falls back
            to (0, 0, 0).

    Returns:
        (sorted_indices, oriented_direction): integer indices into
        contact_positions sorted deepest-first, and direction vector pointing
        from deepest to most lateral.
    """
    if brain_centroid is None:
        brain_centroid = np.zeros(3)

    first_ras = axis_origin + contact_positions[0] * axis_direction
    last_ras = axis_origin + contact_positions[-1] * axis_direction

    indices = np.arange(len(contact_positions))
    if np.linalg.norm(first_ras - brain_centroid) > np.linalg.norm(
        last_ras - brain_centroid
    ):
        return indices[::-1], -axis_direction
    return indices, axis_direction
```

Key changes vs old version:
- Accepts `brain_centroid` parameter (defaults to `(0,0,0)` as fallback)
- Returns **indices** instead of re-zeroed positions (no reprojection)
- Uses brain centroid distance instead of RAS origin distance

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_electrode_detector.py::TestOrientDeepestFirst -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```
git add SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py tests/test_electrode_detector.py
git commit -m "feat: orient_deepest_first uses brain centroid, returns indices"
```

---

### Task 3: Rewrite `detect_electrodes` to use RANSAC + preserved positions

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py:256-347`
- Modify: `tests/test_electrode_detector.py`

**Step 1: Write failing tests**

Update `TestDetectElectrodes` in `tests/test_electrode_detector.py`. The existing tests should mostly still work with the new signature. Add a test for crossing electrodes and position preservation:

```python
class TestDetectElectrodes:
    def _make_centers(self, start, direction, n_contacts, spacing=3.5):
        direction = np.array(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        return np.array(
            [np.array(start) + i * spacing * direction for i in range(n_contacts)]
        )

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

    def test_crossing_electrodes_separated(self):
        """Two crossing electrodes must not be merged."""
        e1 = self._make_centers([-14, 0, 0], [1, 0, 0], 8)
        e2 = self._make_centers([0, -10.5, 0], [0, 1, 0], 6)
        all_centers = np.vstack([e1, e2])
        electrodes = detect_electrodes(all_centers)
        assert len(electrodes) == 2
        contact_counts = sorted([e.num_contacts for e in electrodes])
        assert contact_counts == [6, 8]

    def test_preserves_log_positions(self):
        """Contact RAS positions must be original LoG values, not reprojected."""
        np.random.seed(42)
        e1 = self._make_centers([0, 0, 0], [1, 0, 0], 8)
        e1 += np.random.randn(*e1.shape) * 0.3  # add noise
        original = set(map(tuple, e1))
        electrodes = detect_electrodes(e1)
        assert len(electrodes) == 1
        for c in electrodes[0].contacts:
            assert c.position_ras in original
```

**Step 2: Run tests to verify the crossing test fails with old code**

Run: `.venv/bin/pytest tests/test_electrode_detector.py::TestDetectElectrodes -v`
Expected: `test_crossing_electrodes_separated` FAILS (old code merges them)

**Step 3: Rewrite `detect_electrodes`**

Replace lines 256–347 of `electrode_detector.py`:

```python
def detect_electrodes(
    contact_centers: np.ndarray,
    min_contacts: int = 3,
    expected_spacing: float = 3.5,
    distance_tolerance: float = 2.0,
    max_iterations: int = 1000,
    gap_ratio_threshold: float = 1.8,
    spacing_cutoff_factor: float = 0.65,
    brain_centroid: np.ndarray | None = None,
) -> list[Electrode]:
    """Full detection pipeline: RANSAC grouping + axis fitting + orientation.

    Each row in contact_centers is one LoG-detected contact in RAS coordinates.
    Original positions are preserved (no reprojection onto fitted axis).

    Args:
        contact_centers: (N, 3) array of contact center RAS coordinates.
        min_contacts: Minimum contacts to accept an electrode.
        expected_spacing: Expected contact spacing in mm.
        distance_tolerance: RANSAC perpendicular distance tolerance in mm.
        max_iterations: RANSAC trials per electrode.
        gap_ratio_threshold: Threshold for gap detection in spacing analysis.
        spacing_cutoff_factor: Fraction of expected_spacing below which a
            cluster is rejected as noise.
        brain_centroid: 3D RAS coordinates of brain center for orientation.
            Falls back to (0, 0, 0) if None.

    Returns:
        List of Electrode objects with auto-numbered contacts (unnamed).
        Contact positions are the original input coordinates.

    Example::

        electrodes = detect_electrodes(centers_ras, brain_centroid=centroid)
    """
    groups = ransac_group_contacts(
        contact_centers,
        expected_spacing=expected_spacing,
        distance_tolerance=distance_tolerance,
        max_iterations=max_iterations,
        min_contacts=min_contacts,
    )

    electrodes = []
    for group_coords in groups:
        # Fit axis
        center, direction = fit_electrode_axis(group_coords)

        # Project onto axis, sort
        projections = np.dot(group_coords - center, direction)
        sorted_indices = np.argsort(projections)
        sorted_projections = projections[sorted_indices]

        # Analyze spacing
        spacing_info = analyze_spacing(sorted_projections, gap_ratio_threshold)

        # Reject implausibly small spacing
        if (
            0
            < spacing_info["contact_spacing"]
            < expected_spacing * spacing_cutoff_factor
        ):
            continue

        # Orient deepest first
        orient_indices, oriented_dir = orient_deepest_first(
            sorted_projections, center, direction, brain_centroid=brain_centroid
        )
        # Map back through sorted_indices
        final_indices = sorted_indices[orient_indices]

        # Build Electrode with original coordinates
        params = ElectrodeParams(
            contact_length=2.0,
            contact_spacing=spacing_info["contact_spacing"],
            contact_diameter=0.8,
            gap_spacing=spacing_info["gap_spacing"],
            contacts_per_group=spacing_info["contacts_per_group"],
        )

        contacts = []
        for i, idx in enumerate(final_indices):
            contacts.append(
                Contact(index=i + 1, position_ras=tuple(group_coords[idx]))
            )

        electrode = Electrode(
            name="",
            params=params,
            contacts=contacts,
            trajectory_direction=tuple(oriented_dir),
        )
        electrodes.append(electrode)

    return electrodes
```

**Step 4: Run all detector tests**

Run: `.venv/bin/pytest tests/test_electrode_detector.py tests/test_ransac_grouping.py -v`
Expected: All tests PASS

**Step 5: Commit**

```
git add SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py tests/test_electrode_detector.py
git commit -m "feat: rewrite detect_electrodes to use RANSAC, preserve LoG positions"
```

---

### Task 4: Remove dead code (`cluster_into_electrodes`, `merge_collinear_clusters`)

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py:15-164`
- Modify: `tests/test_electrode_detector.py`

**Step 1: Delete `cluster_into_electrodes` (lines 15-54) and `merge_collinear_clusters` (lines 79-164) from `electrode_detector.py`**

Also remove their imports from `tests/test_electrode_detector.py` and delete `TestClusterIntoElectrodes` and `TestMergeCollinearClusters` test classes.

**Step 2: Run all tests**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests PASS (no remaining references to deleted functions)

**Step 3: Commit**

```
git add SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py tests/test_electrode_detector.py
git commit -m "refactor: remove single-linkage clustering and collinear merging"
```

---

### Task 5: Wire RANSAC parameters through `SEEGFellowLogic` and `ElectrodeDetector`

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellow.py:769-843` (`run_electrode_detection`)
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py` (`ElectrodeDetector`)

**Step 1: Update `ElectrodeDetector.__init__` and `detect_all`**

Add `distance_tolerance` and `max_iterations` parameters. Replace the call to `detect_electrodes` and pass `brain_centroid`. Remove `collinearity_tolerance` (no longer used).

In `ElectrodeDetector.__init__`:
```python
def __init__(
    self,
    min_contacts: int = 3,
    expected_spacing: float = 3.5,
    distance_tolerance: float = 2.0,
    max_iterations: int = 1000,
    gap_ratio_threshold: float = 1.8,
    spacing_cutoff_factor: float = 0.65,
):
    self.min_contacts = min_contacts
    self.expected_spacing = expected_spacing
    self.distance_tolerance = distance_tolerance
    self.max_iterations = max_iterations
    self.gap_ratio_threshold = gap_ratio_threshold
    self.spacing_cutoff_factor = spacing_cutoff_factor
```

In `detect_all`, pass the new params to `detect_electrodes`:
```python
electrodes = detect_electrodes(
    centers_ras,
    min_contacts=self.min_contacts,
    expected_spacing=self.expected_spacing,
    distance_tolerance=self.distance_tolerance,
    max_iterations=self.max_iterations,
    gap_ratio_threshold=self.gap_ratio_threshold,
    spacing_cutoff_factor=self.spacing_cutoff_factor,
    brain_centroid=brain_centroid,
)
```

Note: `detect_all` needs a new `brain_centroid` parameter.

**Step 2: Update `run_electrode_detection` in `SEEGFellow.py`**

This method currently bypasses grouping. Re-enable it and pass the new RANSAC params + brain centroid:

```python
def run_electrode_detection(
    self,
    sigma: float = 1.2,
    expected_spacing: float = 3.5,
    min_contacts: int = 3,
    max_component_voxels: int = 500,
    spacing_cutoff_factor: float = 0.65,
    distance_tolerance: float = 2.0,
    max_iterations: int = 1000,
) -> None:
```

Compute brain centroid from `self._head_mask` (the brain mask from step 4a):

```python
brain_centroid = None
if self._head_mask is not None:
    brain_voxels = np.argwhere(self._head_mask > 0)
    if len(brain_voxels) > 0:
        # Convert centroid from IJK to RAS
        centroid_ijk = brain_voxels.mean(axis=0)[::-1]  # KJI → IJK
        ijk_to_ras = ElectrodeDetector._get_ijk_to_ras_matrix(self._ct_node)
        centroid_h = np.append(centroid_ijk, 1.0)
        brain_centroid = (ijk_to_ras @ centroid_h)[:3]
```

Then call `detect_electrodes` with all parameters, store results in `self.electrodes`, and create fiducial nodes.

**Step 3: Run all tests**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```
git add SEEGFellow/SEEGFellow/SEEGFellow.py SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py
git commit -m "feat: wire RANSAC params through logic and detector, compute brain centroid"
```

---

### Task 6: Add GUI widgets for RANSAC parameters

**Files:**
- Modify: `SEEGFellow/SEEGFellow/Resources/UI/SEEGFellow.ui`
- Modify: `SEEGFellow/SEEGFellow/SEEGFellow.py` (`_on_detect_electrodes_clicked`)

**Step 1: Add widgets to UI file**

Add two new widgets to the Step 4c section of `SEEGFellow.ui`, alongside the existing tuning parameters:

- `distanceToleranceSpinBox`: `QDoubleSpinBox`, min=0.5, max=5.0, default=2.0, step=0.5, suffix=" mm"
- `maxIterationsSpinBox`: `QSpinBox`, min=100, max=10000, default=1000, step=100

**Step 2: Update event handler**

In `_on_detect_electrodes_clicked`, read the new widgets and pass to logic:

```python
distance_tolerance = self.ui.distanceToleranceSpinBox.value
max_iterations = self.ui.maxIterationsSpinBox.value
# ... pass to self.logic.run_electrode_detection(...)
```

**Step 3: Commit**

```
git add SEEGFellow/SEEGFellow/Resources/UI/SEEGFellow.ui SEEGFellow/SEEGFellow/SEEGFellow.py
git commit -m "feat: add RANSAC tuning parameter widgets to Step 4c UI"
```

---

### Task 7: Update existing tests for new signatures

**Files:**
- Modify: `tests/test_electrode_detector.py`

**Step 1: Fix any remaining test failures**

The old `TestDetectElectrodes` tests call `detect_electrodes` with `spacing_cutoff_factor` and `collinearity_tolerance` parameters. Update calls:

- Remove `collinearity_tolerance` argument (no longer exists)
- Keep `spacing_cutoff_factor` (still supported)
- `test_electrodes_6mm_apart_not_merged_into_one`: RANSAC handles this by default due to perpendicular distance tolerance; verify it still passes
- `TestDetectElectrodesWithNoise.test_ignores_scattered_noise`: should work with RANSAC spacing validation

**Step 2: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests PASS

**Step 3: Commit**

```
git add tests/test_electrode_detector.py
git commit -m "test: update detector tests for RANSAC-based grouping"
```
