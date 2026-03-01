# SynthSeg Integration & Contact Labeling Design

## Summary

Replace the current brain mask step (SynthStrip / Scipy) with FreeSurfer's
SynthSeg, which produces a volumetric parcellation of ~100 brain regions (DKT
atlas). This single segmentation serves two purposes:

1. **Brain mask** — derived by binarizing the parcellation (label > 0).
2. **Contact labeling** — each detected contact's RAS coordinate is mapped to
   its nearest SynthSeg label, producing a region name per contact.

FreeSurfer becomes a hard requirement (no Scipy fallback).

## SynthSeg Integration (Replacing Brain Mask Step)

### CLI Invocation

```
mri_synthseg --i input.nii.gz --o seg.nii.gz --parc [--robust|--fast] --cpu --threads N
```

- `--parc`: enables DKT cortical parcellation (~31 regions per hemisphere)
- `--robust` / `--fast`: user-selectable mode (default: robust)
- `--cpu`: explicit CPU mode (hospital PCs lack CUDA)
- `--threads N`: user-configurable thread count (default: 1)
- Runtime: ~2–5 min CPU (robust), ~1–2 min (fast)

### Code Architecture

**`SynthSegBrainMask`** in `brain_mask.py` replaces both `SynthStripBrainMask`
and `ScipyBrainMask`. It conforms to the existing `BrainMaskStrategy` protocol:

- `compute(volume, affine)` → binary uint8 mask (1 = brain, 0 = outside)
- After `compute()`, the full parcellation volume is available via
  `self.parcellation` (int32 array, same shape as input after transposing from
  NIfTI to Slicer axis order) and `self.parcellation_affine` (4×4 matrix).
- `is_available()` checks for `mri_synthseg` on PATH or under
  `$FREESURFER_HOME/bin/`.
- Constructor accepts `robust: bool = True` and `threads: int = 1`.

`ScipyBrainMask` and `SynthStripBrainMask` are removed.
`get_available_strategies()` returns `[SynthSegBrainMask()]`.

### FreeSurfer Location Resolution

1. Check `FREESURFER_HOME` environment variable.
2. Check Slicer application settings (`SEEGFellow/FreeSurferHome`).
3. If neither found, show a directory browser in the UI. On selection, validate
   (`bin/mri_synthseg` must exist) and persist to Slicer settings.
4. Display a status label: "FreeSurfer: found at /path" or "Not configured".

### Brain Mask Derivation

The binary mask is simply `(parcellation > 0).astype(np.uint8)`. This replaces
the morphological (Scipy) and SynthStrip approaches.

### Logic Integration

`SEEGFellowLogic.run_intracranial_mask()`:

- Runs `SynthSegBrainMask.compute(t1_array, affine)`.
- Stores `self._parcellation` and `self._parcellation_affine` from the strategy
  object (for later contact labeling).
- Derives the binary mask and resamples to CT space as before.
- Stores `self._head_mask` (binary, CT space) for metal filtering.

## Contact Labeling

### Algorithm

Contact RAS coordinates are already in T1-aligned world space (the IJK→RAS
conversion in `ElectrodeDetector` composes the CT's parent registration
transform). No additional transform is needed.

For each contact:

1. **RAS → parcellation voxel IJK**: Apply `inv(parcellation_affine)` to the
   contact's `position_ras` to get voxel indices in the SynthSeg volume.
   Round to nearest integer.
2. **Look up label**: Read `parcellation[K, J, I]` (Slicer axis convention:
   array indexed as K, J, I after transposing from NIfTI's I, J, K).
3. **Map to name**: Use `SYNTHSEG_LUT[label]` → human-readable name (e.g.
   `1028 → "Left Superior Frontal"`).
4. **WM fallback**: If the label is cerebral white matter (2 or 41), search a
   ~3 mm radius sphere for the nearest parcellated cortical label. Report as
   "WM near {region}".

### Label Lookup Table

SynthSeg with `--parc` outputs integer labels following the FreeSurfer LUT:

**Subcortical (base labels):**

| Label | Structure | Label | Structure |
|-------|-----------|-------|-----------|
| 2 | Left Cerebral WM | 41 | Right Cerebral WM |
| 3 | Left Cerebral Cortex | 42 | Right Cerebral Cortex |
| 4 | Left Lateral Ventricle | 43 | Right Lateral Ventricle |
| 5 | Left Inf Lat Ventricle | 44 | Right Inf Lat Ventricle |
| 7 | Left Cerebellum WM | 46 | Right Cerebellum WM |
| 8 | Left Cerebellum Cortex | 47 | Right Cerebellum Cortex |
| 10 | Left Thalamus | 49 | Right Thalamus |
| 11 | Left Caudate | 50 | Right Caudate |
| 12 | Left Putamen | 51 | Right Putamen |
| 13 | Left Pallidum | 52 | Right Pallidum |
| 14 | 3rd Ventricle | 15 | 4th Ventricle |
| 16 | Brain Stem | 24 | CSF |
| 17 | Left Hippocampus | 53 | Right Hippocampus |
| 18 | Left Amygdala | 54 | Right Amygdala |
| 26 | Left Accumbens | 58 | Right Accumbens |
| 28 | Left Ventral DC | 60 | Right Ventral DC |

**Cortical parcellation (DKT, labels 1000–1035 left, 2000–2035 right):**

Includes: caudal anterior cingulate, caudal middle frontal, cuneus,
entorhinal, fusiform, inferior parietal, inferior temporal, isthmus cingulate,
lateral occipital, lateral orbitofrontal, lingual, medial orbitofrontal, middle
temporal, parahippocampal, paracentral, pars opercularis, pars orbitalis, pars
triangularis, pericalcarine, postcentral, posterior cingulate, precentral,
precuneus, rostral anterior cingulate, rostral middle frontal, superior
frontal, superior parietal, superior temporal, supramarginal, transverse
temporal, insula.

### Code Location

New file `SEEGFellowLib/contact_labeler.py`:

- `SYNTHSEG_LUT: dict[int, str]` — complete label-to-name mapping.
- `label_contacts(contacts_ras, parcellation, parcellation_affine) → list[str]`
  — core labeling function.
- `_nearest_cortical_label(parcellation, ijk, radius_voxels)` — WM fallback
  helper.

### Logic Integration

`SEEGFellowLogic.run_contact_labeling()`:

- Requires `self._parcellation` and `self.electrodes` to be populated.
- Calls `label_contacts()` for each electrode's contacts.
- Stores region names on each `Contact` object (new `region: str` field on the
  `Contact` dataclass, default `""`).

## UI Changes

### Modified Steps

| Step | Name | Change |
|------|------|--------|
| 1 | Load Data | Unchanged |
| 2 | Co-registration | Unchanged |
| 3 | Brain Segmentation | **Replaces** "Intracranial Mask". Removes strategy combo. Adds: mode toggle (Fast / Robust, default Robust), thread count spinbox (default 1, range 1–16), "Compute Segmentation" button, FreeSurfer path browser (if not auto-detected), status label. |
| 3b | Metal Threshold | Unchanged |
| 3c | Contact Detection | Unchanged |
| 4 | Rename Electrodes | Unchanged |
| **5** | **Label Contacts** | **New**. "Label All Contacts" button. Anatomy table (see below). |
| 6 | Export | Adds `Region` column to CSV. |

### Anatomy Table (Step 5)

A `QTableWidget` in the module panel:

- **Rows**: one per electrode, sorted by name.
- **Columns**: Contact 1, Contact 2, …, Contact N (max across all electrodes).
- **Cell content**: region name (e.g. "Left Hippocampus", "WM near Left
  Insula").
- Empty cells for contacts that don't exist on shorter electrodes.

### CSV Export

Additional `Region` column:

```
Electrode,Contact,R,A,S,Region
A,1,-25.3,-12.1,5.4,Left Hippocampus
A,2,-23.1,-11.8,6.2,WM near Left Amygdala
```

## Data Model Changes

`Contact` dataclass gains a `region: str = ""` field.

## Session Restore

`try_restore_from_scene()` is extended to look for a stored SynthSeg
parcellation node (`_SEEGFellow_SynthSeg_Parcellation`). If found, restores
`self._parcellation` and `self._parcellation_affine` so the labeling step
works without re-running SynthSeg.

## Testing

- `contact_labeler.py` is pure numpy/dict code — fully testable outside Slicer.
- Tests: label lookup correctness, WM fallback radius search, edge cases
  (contact outside brain, boundary voxels).
- `SynthSegBrainMask`: tested via mocking subprocess calls (same pattern as
  existing SynthStrip tests).
