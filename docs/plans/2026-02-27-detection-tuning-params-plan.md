# Detection Tuning Parameters Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose five detection parameters as interactive UI widgets in Step 4c so the user can tune and re-run without reloading the module.

**Architecture:** Three layers of change — (1) add `spacing_cutoff_factor` to `detect_electrodes()` and extract `_filter_contact_mask()` helper with `max_component_voxels`, (2) wire through `ElectrodeDetector` and `run_electrode_detection()` with re-run cleanup, (3) add four widget rows to the `.ui` file.

**Tech Stack:** Python, Qt `.ui` file (XML), scipy ndimage, pytest

---

### Task 1: Add `spacing_cutoff_factor` to `detect_electrodes()`

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py` (`detect_electrodes` function)
- Test: `tests/test_electrode_detector.py`

The filter `if 0 < spacing_info["contact_spacing"] < expected_spacing * 0.65:` has the `0.65` hardcoded. Make it a parameter.

**Step 1: Write the failing test**

Add to `TestDetectElectrodes` in `tests/test_electrode_detector.py`:

```python
def test_spacing_cutoff_factor_controls_acceptance(self):
    """A cluster at 1.5 mm spacing is rejected at factor=0.65 but accepted at factor=0.30."""
    # 1.5 mm contacts — below the default cutoff (3.5 * 0.65 = 2.275 mm)
    centers = self._make_centers([0, 0, 0], [1, 0, 0], 8, spacing=1.5)
    assert detect_electrodes(centers, spacing_cutoff_factor=0.65) == []
    assert len(detect_electrodes(centers, expected_spacing=1.5, spacing_cutoff_factor=0.65)) == 1
```

**Step 2: Run to verify it fails**

```bash
.venv/bin/pytest tests/test_electrode_detector.py::TestDetectElectrodes::test_spacing_cutoff_factor_controls_acceptance -v
```

Expected: `TypeError: detect_electrodes() got an unexpected keyword argument 'spacing_cutoff_factor'`

**Step 3: Add the parameter to `detect_electrodes()`**

In `electrode_detector.py`, change the signature:

```python
def detect_electrodes(
    contact_centers: np.ndarray,
    min_contacts: int = 3,
    expected_spacing: float = 3.5,
    collinearity_tolerance: float = 10.0,
    gap_ratio_threshold: float = 1.8,
    spacing_cutoff_factor: float = 0.65,   # ← add this
) -> list[Electrode]:
```

Replace the hardcoded filter line:

```python
        # Reject clusters whose spacing is implausibly small (scattered noise)
        if 0 < spacing_info["contact_spacing"] < expected_spacing * spacing_cutoff_factor:
            continue
```

Also update the debug print in `detect_all()` that mirrors this filter (search for `spacing_cutoff`):

```python
        spacing_cutoff = self.expected_spacing * self.spacing_cutoff_factor
```

(You will also add `spacing_cutoff_factor` to `ElectrodeDetector.__init__` in Task 2 — for now just leave `detect_all()` using `self.expected_spacing * 0.65` as a temporary placeholder; Task 2 fixes it completely.)

**Step 4: Run to verify it passes**

```bash
.venv/bin/pytest tests/test_electrode_detector.py -v
```

Expected: all pass.

**Step 5: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py tests/test_electrode_detector.py
git commit -m "feat: add spacing_cutoff_factor param to detect_electrodes()"
```

---

### Task 2: Extract `_filter_contact_mask()` and add `max_component_voxels` + `spacing_cutoff_factor` to `ElectrodeDetector`

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py`
- Test: `tests/test_electrode_detector.py`

The component-size filter in `detect_all()` has `500` hardcoded. Extract it to a testable module-level function, then make it a parameter. Also add `spacing_cutoff_factor` to `ElectrodeDetector`.

**Step 1: Write the failing tests**

Add a new test class to `tests/test_electrode_detector.py`:

```python
from SEEGFellowLib.electrode_detector import _filter_contact_mask

class TestFilterContactMask:
    def _make_mask(self, shape=(20, 20, 20)):
        return np.zeros(shape, dtype=np.uint8)

    def test_keeps_components_within_range(self):
        mask = self._make_mask()
        # Component of 10 voxels — within [3, 500]
        mask[0:10, 0, 0] = 1
        result = _filter_contact_mask(mask, min_voxels=3, max_voxels=500)
        assert result.sum() == 10

    def test_removes_component_above_max(self):
        mask = self._make_mask()
        # Component of 600 voxels — above max_voxels=500
        mask[0:10, 0:10, 0:6] = 1  # 10*10*6 = 600
        result = _filter_contact_mask(mask, min_voxels=3, max_voxels=500)
        assert result.sum() == 0

    def test_removes_component_below_min(self):
        mask = self._make_mask()
        mask[0, 0, 0] = 1  # 1 voxel — below min=3
        result = _filter_contact_mask(mask, min_voxels=3, max_voxels=500)
        assert result.sum() == 0

    def test_max_voxels_parameter_is_respected(self):
        mask = self._make_mask()
        mask[0:10, 0, 0] = 1  # 10 voxels
        # With max_voxels=5 it should be excluded
        assert _filter_contact_mask(mask, min_voxels=3, max_voxels=5).sum() == 0
        # With max_voxels=15 it should be included
        assert _filter_contact_mask(mask, min_voxels=3, max_voxels=15).sum() == 10
```

**Step 2: Run to verify it fails**

```bash
.venv/bin/pytest tests/test_electrode_detector.py::TestFilterContactMask -v
```

Expected: `ImportError: cannot import name '_filter_contact_mask'`

**Step 3: Extract `_filter_contact_mask()` and wire it into `detect_all()`**

Add the following function to `electrode_detector.py` (before `ElectrodeDetector`):

```python
def _filter_contact_mask(
    metal_mask: np.ndarray,
    min_voxels: int = 3,
    max_voxels: int = 500,
) -> np.ndarray:
    """Keep connected components whose size falls within [min_voxels, max_voxels].

    Removes components that are too small (noise) or too large (entry bolts,
    bone screws) to be SEEG electrode contacts.

    Args:
        metal_mask: Binary uint8 mask of candidate metal voxels.
        min_voxels: Minimum component size to keep.
        max_voxels: Maximum component size to keep.

    Returns:
        Filtered binary mask of the same shape.

    Example::

        contact_mask = _filter_contact_mask(metal_mask, min_voxels=3, max_voxels=500)
    """
    from scipy import ndimage as _ndi

    labeled, n_comp = _ndi.label(metal_mask)
    if n_comp == 0:
        return np.zeros_like(metal_mask)
    comp_sizes = _ndi.sum(metal_mask, labeled, range(1, n_comp + 1))
    result = np.zeros_like(metal_mask)
    for idx, size in enumerate(comp_sizes):
        if min_voxels <= size <= max_voxels:
            result[labeled == (idx + 1)] = 1
    return result
```

In `detect_all()`, replace the inline component-filter block (the `from scipy import ndimage as _ndi` ... `print(f"[SEEGFellow] contact_mask nonzero=...")` section) with:

```python
        contact_mask = _filter_contact_mask(
            metal_mask, min_voxels=3, max_voxels=max_component_voxels
        )
        max_comp = int(contact_mask.max()) if contact_mask.any() else 0
        print(
            f"[SEEGFellow] metal_mask nonzero={int(np.sum(metal_mask))}"
            f"  components={int(np.sum(metal_mask) - np.sum(contact_mask))}"
            f"  (filtered to max {max_component_voxels} voxels)"
        )
        print(f"[SEEGFellow] contact_mask nonzero={int(np.sum(contact_mask))}")
```

Now add `spacing_cutoff_factor` and `max_component_voxels` to `ElectrodeDetector`:

```python
class ElectrodeDetector:
    def __init__(
        self,
        min_contacts: int = 3,
        expected_spacing: float = 3.5,
        collinearity_tolerance: float = 10.0,
        gap_ratio_threshold: float = 1.8,
        spacing_cutoff_factor: float = 0.65,   # ← add
    ):
        self.min_contacts = min_contacts
        self.expected_spacing = expected_spacing
        self.collinearity_tolerance = collinearity_tolerance
        self.gap_ratio_threshold = gap_ratio_threshold
        self.spacing_cutoff_factor = spacing_cutoff_factor   # ← add
```

Update `detect_all()` signature:

```python
    def detect_all(
        self,
        ct_volume_node,
        metal_mask: np.ndarray,
        sigma: float = 1.2,
        max_component_voxels: int = 500,   # ← add
    ) -> list[Electrode]:
```

Update the `detect_electrodes(...)` call at the end of `detect_all()`:

```python
        electrodes = detect_electrodes(
            centers_ras,
            min_contacts=self.min_contacts,
            expected_spacing=self.expected_spacing,
            collinearity_tolerance=self.collinearity_tolerance,
            gap_ratio_threshold=self.gap_ratio_threshold,
            spacing_cutoff_factor=self.spacing_cutoff_factor,   # ← add
        )
```

Also fix the debug spacing_cutoff line in `detect_all()` (currently uses hardcoded 0.65):

```python
        spacing_cutoff = self.expected_spacing * self.spacing_cutoff_factor
```

**Step 4: Run to verify all pass**

```bash
.venv/bin/pytest tests/ -v
```

Expected: all pass.

**Step 5: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py tests/test_electrode_detector.py
git commit -m "feat: add max_component_voxels and spacing_cutoff_factor to ElectrodeDetector"
```

---

### Task 3: Update `SEEGFellow.py` — new params + re-run cleanup

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellow.py`

No new tests needed here (Slicer-only code). Two changes: wire UI → logic, and fix re-run accumulation.

**Step 1: Update `run_electrode_detection()` signature and cleanup**

Replace the current method:

```python
    def run_electrode_detection(self, sigma: float = 1.2) -> None:
        """Run full automated electrode detection pipeline.

        Step 4b (metal threshold) must be completed before calling this method.

        Example::

            logic.run_electrode_detection(sigma=1.2)
        """
        if self._metal_mask is None:
            raise RuntimeError(
                "Metal mask not computed. Run step 4b (metal threshold) first."
            )

        from SEEGFellowLib.electrode_detector import ElectrodeDetector

        detector = ElectrodeDetector()
        self.electrodes = detector.detect_all(
            self._ct_node, metal_mask=self._metal_mask, sigma=sigma
        )
        self._create_fiducials_for_electrodes()
```

With:

```python
    def run_electrode_detection(
        self,
        sigma: float = 1.2,
        expected_spacing: float = 3.5,
        min_contacts: int = 3,
        max_component_voxels: int = 500,
        spacing_cutoff_factor: float = 0.65,
    ) -> None:
        """Run full automated electrode detection pipeline.

        Step 4b (metal threshold) must be completed before calling this method.

        Example::

            logic.run_electrode_detection(sigma=1.2)
        """
        if self._metal_mask is None:
            raise RuntimeError(
                "Metal mask not computed. Run step 4b (metal threshold) first."
            )

        # Remove fiducials from any previous run before creating new ones.
        for electrode in self.electrodes:
            node = slicer.mrmlScene.GetNodeByID(electrode.markups_node_id)
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)
        self.electrodes = []

        from SEEGFellowLib.electrode_detector import ElectrodeDetector

        detector = ElectrodeDetector(
            expected_spacing=expected_spacing,
            min_contacts=min_contacts,
            spacing_cutoff_factor=spacing_cutoff_factor,
        )
        self.electrodes = detector.detect_all(
            self._ct_node,
            metal_mask=self._metal_mask,
            sigma=sigma,
            max_component_voxels=max_component_voxels,
        )
        self._create_fiducials_for_electrodes()
```

**Step 2: Update `_on_detect_electrodes_clicked()` to read the new widgets**

Replace:

```python
    def _on_detect_electrodes_clicked(self):
        self._ensure_session_restored()
        sigma = self.ui.sigmaSlider.value
        try:
            slicer.util.showStatusMessage("Detecting electrodes...")
            self.logic.run_electrode_detection(sigma=sigma)
```

With:

```python
    def _on_detect_electrodes_clicked(self):
        self._ensure_session_restored()
        sigma = self.ui.sigmaSlider.value
        expected_spacing = self.ui.expectedSpacingSpinBox.value
        min_contacts = self.ui.minContactsSpinBox.value
        max_component_voxels = self.ui.maxComponentVoxelsSpinBox.value
        spacing_cutoff_factor = self.ui.spacingCutoffSlider.value / 100.0
        try:
            slicer.util.showStatusMessage("Detecting electrodes...")
            self.logic.run_electrode_detection(
                sigma=sigma,
                expected_spacing=expected_spacing,
                min_contacts=min_contacts,
                max_component_voxels=max_component_voxels,
                spacing_cutoff_factor=spacing_cutoff_factor,
            )
```

**Step 3: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellow.py
git commit -m "feat: wire detection tuning params through SEEGFellowLogic and widget handler"
```

---

### Task 4: Add widgets to the UI file

**Files:**
- Modify: `SEEGFellow/SEEGFellow/Resources/UI/SEEGFellow.ui`

The contact detection QFormLayout currently has one row (sigma) followed by `detectElectrodesButton`. Add four rows between sigma and the button.

**Step 1: Open the .ui file and find the insertion point**

Find this block:

```xml
       <layout class="QFormLayout">
        <item row="0" column="0"><widget class="QLabel"><property name="text"><string>Sigma (mm):</string></property></widget></item>
        <item row="0" column="1"><widget class="ctkSliderWidget" name="sigmaSlider"><property name="minimum"><double>0.5</double></property><property name="maximum"><double>3.0</double></property><property name="value"><double>1.2</double></property><property name="singleStep"><double>0.1</double></property></widget></item>
       </layout>
```

**Step 2: Replace it with the expanded form layout**

```xml
       <layout class="QFormLayout">
        <item row="0" column="0"><widget class="QLabel"><property name="text"><string>Sigma (mm):</string></property></widget></item>
        <item row="0" column="1"><widget class="ctkSliderWidget" name="sigmaSlider"><property name="minimum"><double>0.5</double></property><property name="maximum"><double>3.0</double></property><property name="value"><double>1.2</double></property><property name="singleStep"><double>0.1</double></property></widget></item>
        <item row="1" column="0"><widget class="QLabel"><property name="text"><string>Contact spacing (mm):</string></property></widget></item>
        <item row="1" column="1"><widget class="QDoubleSpinBox" name="expectedSpacingSpinBox"><property name="minimum"><double>1.5</double></property><property name="maximum"><double>5.0</double></property><property name="value"><double>3.5</double></property><property name="singleStep"><double>0.5</double></property></widget></item>
        <item row="2" column="0"><widget class="QLabel"><property name="text"><string>Min. contacts:</string></property></widget></item>
        <item row="2" column="1"><widget class="QSpinBox" name="minContactsSpinBox"><property name="minimum"><number>2</number></property><property name="maximum"><number>16</number></property><property name="value"><number>3</number></property></widget></item>
        <item row="3" column="0"><widget class="QLabel"><property name="text"><string>Max component voxels:</string></property></widget></item>
        <item row="3" column="1"><widget class="QSpinBox" name="maxComponentVoxelsSpinBox"><property name="minimum"><number>50</number></property><property name="maximum"><number>2000</number></property><property name="value"><number>500</number></property><property name="singleStep"><number>50</number></property></widget></item>
        <item row="4" column="0"><widget class="QLabel"><property name="text"><string>Spacing cutoff (%):</string></property></widget></item>
        <item row="4" column="1"><widget class="ctkSliderWidget" name="spacingCutoffSlider"><property name="minimum"><double>30</double></property><property name="maximum"><double>90</double></property><property name="value"><double>65</double></property><property name="singleStep"><double>5</double></property></widget></item>
       </layout>
```

**Step 3: Run all tests to confirm nothing broke**

```bash
.venv/bin/pytest tests/ -v
```

Expected: all pass. (UI changes are not testable in pytest — verify visually in Slicer.)

**Step 4: Commit**

```bash
git add SEEGFellow/SEEGFellow/Resources/UI/SEEGFellow.ui
git commit -m "feat: add detection tuning parameter widgets to Step 4c UI"
```
