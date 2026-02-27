# Modular Brain Mask — Design Document

## Problem

The current `compute_brain_mask()` in `metal_segmenter.py` uses a single scipy-based morphological approach that is rough in quality. The previous attempt to integrate DeepBet was reverted due to PyTorch/NumPy dependency conflicts when pip-installing torch into Slicer's Python. We need:

1. A modular architecture so we can swap between different brain mask strategies
2. A high-quality CNN-based option (SynthStrip) alongside the fast scipy fallback
3. A pattern that will generalize to other pipeline steps later

## Design

### Strategy Pattern

A new file `SEEGFellowLib/brain_mask.py` contains a protocol and concrete implementations:

```python
class BrainMaskStrategy(Protocol):
    name: str

    def compute(self, volume: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """Return a binary uint8 mask (1 = brain, 0 = outside)."""
        ...

    def is_available(self) -> bool:
        """Return True if this strategy can run on the current system."""
        ...
```

### Strategies

| Strategy | Class | Quality | Speed | Requirements |
|----------|-------|---------|-------|-------------|
| Scipy morphological | `ScipyBrainMask` | Rough | ~1s | None (scipy bundled with Slicer) |
| SynthStrip CLI | `SynthStripBrainMask` | High | ~30s CPU | FreeSurfer installed, `mri_synthstrip` on PATH or `FREESURFER_HOME` set |

#### ScipyBrainMask

The existing morphological approach, moved from `metal_segmenter.py`:
1. Threshold at 5% of max intensity
2. Morphological closing + hole filling
3. Keep largest connected component
4. Erode ~5mm to strip the skull

`is_available()` always returns True.

#### SynthStripBrainMask

A CLI wrapper around FreeSurfer's `mri_synthstrip`:

1. Write the T1 volume + affine to a temp NIfTI file using nibabel
2. Run `mri_synthstrip -i input.nii.gz -o brain.nii.gz -m mask.nii.gz` as a subprocess
3. Read back the mask NIfTI via nibabel
4. Return as binary uint8 array

`is_available()` checks for `mri_synthstrip` on PATH or under `$FREESURFER_HOME/bin/`.

### Discovery Function

```python
def get_available_strategies() -> list[BrainMaskStrategy]:
    """Return all strategies, with available ones first."""
```

Used by the UI to populate the dropdown.

### Dependencies

- **Keep**: `nibabel` (NIfTI I/O for SynthStrip wrapper)
- **Remove**: `deepbet`
- **System prerequisite**: FreeSurfer (documented, not enforced — SynthStrip just won't appear in dropdown if missing)
- **No new Python deps**

### UI Changes

A `QComboBox` ("Method") added to the Brain Mask section of Step 4, populated from `get_available_strategies()`. Only strategies where `is_available()` returns True are shown. The existing "Compute Brain Mask" button dispatches to the selected strategy.

### What Stays the Same

- `run_intracranial_mask()` in `SEEGFellow.py` — receives a binary mask regardless of strategy, then handles T1→CT resampling as before
- `compute_head_mask()` — unrelated, stays in `metal_segmenter.py`
- Test structure — scipy tests remain similar, SynthStrip tests mock the subprocess call

## File Changes

| File | Change |
|------|--------|
| `SEEGFellowLib/brain_mask.py` | **New** — strategy protocol, `ScipyBrainMask`, `SynthStripBrainMask`, `get_available_strategies()` |
| `SEEGFellowLib/metal_segmenter.py` | Remove `compute_brain_mask()` and `_compute_brain_mask_scipy()` |
| `SEEGFellowLib/__init__.py` | Update exports |
| `SEEGFellow.py` | Import from `brain_mask`, add combo box, pass selected strategy to logic |
| `pyproject.toml` | Remove `deepbet`, keep `nibabel` |
| `tests/test_brain_mask.py` | **New** — tests for both strategies (SynthStrip mocked) |
| `tests/test_metal_segmenter.py` | Remove brain mask tests (moved) |
