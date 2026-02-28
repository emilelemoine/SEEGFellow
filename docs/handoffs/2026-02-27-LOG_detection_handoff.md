# Handoff: SEEGFellow Electrode Detection Debugging

  Goal

  detect_all() in electrode_detector.py returns 0 (or 1 spurious) electrodes. The pipeline runs in 3D Slicer with a real post-implant CT. The user confirms the
  metal mask is correctly placed on the electrodes.

  ---
  CT facts (confirmed in Slicer console)

  - Voxel spacing: (0.457, 0.457, 1.0) mm → sub-mm in XY, 1 mm in Z
  - CT range: -1024 to 3071 HU (normal)
  - Metal mask (thresholded at 2000 HU): ~6000–7000 nonzero voxels; confirmed visually correct
  - Array axis convention: arrayFromVolume returns (K, J, I) = (z, y, x); GetIJKToRASMatrix expects [I, J, K, 1]

  ---
  Root causes found and fixed so far

  1. ✅ Default HU threshold 2500 → 2000

  In metal_segmenter.py, SEEGFellow.py, and the UI slider.

  2. ✅ detect_all() recomputed metal mask internally

  Removed internal compute_head_mask + threshold_volume calls. Caller now passes metal_mask from step 4b. Guard added: raises RuntimeError if _metal_mask is None.

  3. ✅ NMS window too small for sub-mm CT

  detect_contact_centers had a fixed isotropic NMS window of 2*ceil(sigma)+1 = 5 voxels. At 0.457 mm/voxel in XY that is only 2.28 mm — smaller than the 3.5 mm
  contact spacing. Multiple LoG minima per contact survived. Fixed: added nms_size parameter (int or 3-tuple), computed anisotropically in detect_all() as
  2*ceil(1.75/spacing)+1 per axis → (5, 9, 9) for this CT.

  4. ✅ IJK axis order bug in IJK→RAS conversion

  np.argwhere on a (K,J,I) shaped array returns [k, j, i] indices. The code was passing these directly to GetIJKToRASMatrix which expects [i, j, k, 1]. Fixed with
   centers_ijk[:, ::-1] before the matrix multiply. This was causing all detected contacts to be placed in wrong RAS positions.

  5. ✅ Large metallic hardware in metal mask

  Large metallic structures (entry bolts, bone screws, ~thousands of voxels) were generating many LoG edge responses, collapsing all 224 centers into one giant
  cluster with 0.20 mm median projected spacing → rejected by the spacing filter. Fixed: connected component analysis on metal mask before LoG; only keep
  components with 3–500 voxels.

  ---
  Current state of detect_all() pipeline

  arrayFromVolume(ct_node)
    → component size filter on metal_mask (keep 3–500 voxels) → contact_mask
    → detect_contact_centers(ct_array, contact_mask, sigma=1.2, nms_size=(5,9,9))
    → centers_ijk[:, ::-1]  ← axis fix
    → IJK→RAS via GetIJKToRASMatrix (including parent transform)
    → cluster_into_electrodes (distance_threshold = expected_spacing * 2 = 7 mm)
    → merge_collinear_clusters
    → detect_electrodes (spacing filter: reject if spacing < expected_spacing * 0.65 = 2.275 mm)

  ---
  What has NOT been tested yet

  The last two fixes (axis order + component filter) were just committed. detect_all() has not been re-run in Slicer yet. The next agent should:

  1. Run detection in Slicer and read the [SEEGFellow] console output
  2. Check that contact_mask nonzero is much lower than the raw metal mask (confirms bolt filtering works)
  3. Check that RAS range now matches the known electrode locations in the brain
  4. Check cluster sizes and spacings — the per-cluster lines will show exactly which filter (if any) is still rejecting electrodes

  ---
  Key files

  ┌─────────────────────────────────────┬─────────────────────────────────────────────────────────────────┐
  │                File                 │                          What changed                           │
  ├─────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ SEEGFellowLib/electrode_detector.py │ detect_all(): component filter, axis fix, NMS, debug prints     │
  ├─────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ SEEGFellowLib/metal_segmenter.py    │ detect_contact_centers(): added nms_size param                  │
  ├─────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ SEEGFellow.py                       │ run_electrode_detection(): removed threshold param, added guard │
  └─────────────────────────────────────┴─────────────────────────────────────────────────────────────────┘

  Tests

  54/54 passing. Run with .venv/bin/pytest tests/ -v.

  Tunable parameters (if clustering still fails)

  - max_component_voxels=500 in the component filter — if the electrode shaft is metallic and connects contacts into one large component, raise this; if bolts
  still leak through, lower it
  - expected_spacing=3.5 mm — confirm with the user what electrode model is implanted (some use 5 mm spacing)
  - distance_threshold = expected_spacing * 2 = 7.0 mm — if real electrodes are closer than 7 mm to each other laterally, they'll merge into one cluster; lower if
   needed
