# Design: Interactive Detection Tuning Parameters

**Date:** 2026-02-27

## Problem

Running `Detect All Electrodes` repeatedly to tune parameters requires reloading the Slicer
module each time settings change in code. Detection also accumulates duplicate fiducial nodes
in the scene on each re-run.

## Goal

Expose the five most impactful detection parameters as interactive widgets in Step 4c so the
user can tweak and re-run without reloading.

## Parameters to Expose

| Widget | Parameter | Range | Default | Step |
|---|---|---|---|---|
| `sigmaSlider` (existing) | LoG sigma | 0.5–3.0 mm | 1.2 | 0.1 |
| `expectedSpacingSpinBox` (new) | `expected_spacing` | 1.5–5.0 mm | 3.5 | 0.5 |
| `minContactsSpinBox` (new) | `min_contacts` | 2–16 | 3 | 1 |
| `maxComponentVoxelsSpinBox` (new) | `max_component_voxels` | 50–2000 | 500 | 50 |
| `spacingCutoffSlider` (new) | `spacing_cutoff_factor` | 30–90 % | 65 | 5 |

## UI Changes

In `SEEGFellow.ui`, add four new QFormLayout rows inside the existing
`contactDetectionCollapsibleButton` QFormLayout, before `detectElectrodesButton`:

- Row 1: `Contact spacing (mm):` + `QDoubleSpinBox` named `expectedSpacingSpinBox`
- Row 2: `Min. contacts:` + `QSpinBox` named `minContactsSpinBox`
- Row 3: `Max component voxels:` + `QSpinBox` named `maxComponentVoxelsSpinBox`
- Row 4: `Spacing cutoff (%):` + `ctkSliderWidget` named `spacingCutoffSlider`

## Logic Changes

### `run_electrode_detection()` signature

```python
def run_electrode_detection(
    self,
    sigma: float = 1.2,
    expected_spacing: float = 3.5,
    min_contacts: int = 3,
    max_component_voxels: int = 500,
    spacing_cutoff_factor: float = 0.65,
) -> None:
```

Pass `expected_spacing`, `min_contacts`, and `spacing_cutoff_factor` into
`ElectrodeDetector(...)`, and `max_component_voxels` into `detector.detect_all(...)`.

### `detect_all()` signature

Add `max_component_voxels: int = 500` parameter (currently hardcoded as `500`).

### Re-run cleanup

Before creating new fiducials in `run_electrode_detection()`, remove any fiducial nodes from
the previous run and clear `self.electrodes`. This lets the user hit Detect repeatedly without
accumulating duplicate markups.

## Out of Scope

- `collinearity_tolerance` and `gap_ratio_threshold` — rarely need tuning, stay hardcoded
- No changes to other pipeline steps
