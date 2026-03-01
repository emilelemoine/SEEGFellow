# SEEGFellow/SEEGFellow/SEEGFellowLib/contact_labeler.py
"""Map electrode contact positions to anatomical regions via SynthSeg parcellation.

Uses the FreeSurfer SynthSeg label lookup table (DKT atlas) to convert integer
parcellation labels into human-readable region names.
"""

from __future__ import annotations

import numpy as np

# FreeSurfer SynthSeg label → human-readable name.
# Base labels (subcortical + WM + ventricles):
_BASE_LABELS: dict[int, str] = {
    2: "Left Cerebral WM",
    3: "Left Cerebral Cortex",
    4: "Left Lateral Ventricle",
    5: "Left Inf Lat Ventricle",
    7: "Left Cerebellum WM",
    8: "Left Cerebellum Cortex",
    10: "Left Thalamus",
    11: "Left Caudate",
    12: "Left Putamen",
    13: "Left Pallidum",
    14: "3rd Ventricle",
    15: "4th Ventricle",
    16: "Brain Stem",
    17: "Left Hippocampus",
    18: "Left Amygdala",
    24: "CSF",
    26: "Left Accumbens",
    28: "Left Ventral DC",
    41: "Right Cerebral WM",
    42: "Right Cerebral Cortex",
    43: "Right Lateral Ventricle",
    44: "Right Inf Lat Ventricle",
    46: "Right Cerebellum WM",
    47: "Right Cerebellum Cortex",
    49: "Right Thalamus",
    50: "Right Caudate",
    51: "Right Putamen",
    52: "Right Pallidum",
    53: "Right Hippocampus",
    54: "Right Amygdala",
    58: "Right Accumbens",
    60: "Right Ventral DC",
}

# DKT cortical parcellation names (same for left 1000-series and right 2000-series).
_DKT_NAMES: dict[int, str] = {
    2: "Caudal Anterior Cingulate",
    3: "Caudal Middle Frontal",
    5: "Cuneus",
    6: "Entorhinal",
    7: "Fusiform",
    8: "Inferior Parietal",
    9: "Inferior Temporal",
    10: "Isthmus Cingulate",
    11: "Lateral Occipital",
    12: "Lateral Orbitofrontal",
    13: "Lingual",
    14: "Medial Orbitofrontal",
    15: "Middle Temporal",
    16: "Parahippocampal",
    17: "Paracentral",
    18: "Pars Opercularis",
    19: "Pars Orbitalis",
    20: "Pars Triangularis",
    21: "Pericalcarine",
    22: "Postcentral",
    23: "Posterior Cingulate",
    24: "Precentral",
    25: "Precuneus",
    26: "Rostral Anterior Cingulate",
    27: "Rostral Middle Frontal",
    28: "Superior Frontal",
    29: "Superior Parietal",
    30: "Superior Temporal",
    31: "Supramarginal",
    34: "Transverse Temporal",
    35: "Insula",
}

# Build the combined LUT.
SYNTHSEG_LUT: dict[int, str] = dict(_BASE_LABELS)
for _offset, _name in _DKT_NAMES.items():
    SYNTHSEG_LUT[1000 + _offset] = f"Left {_name}"
    SYNTHSEG_LUT[2000 + _offset] = f"Right {_name}"

# Labels that represent cerebral white matter (trigger nearest-cortex search).
_WM_LABELS = {2, 41}

# Labels in the 1000–1035 and 2000–2035 range are cortical parcels.
_CORTICAL_LABEL_RANGES = (range(1000, 1036), range(2000, 2036))


def _is_cortical_parcel(label: int) -> bool:
    return any(label in r for r in _CORTICAL_LABEL_RANGES)


def label_contacts(
    contacts_ras: np.ndarray,
    parcellation: np.ndarray,
    parcellation_affine: np.ndarray,
    search_radius_mm: float = 3.0,
) -> list[str]:
    """Map contact RAS positions to anatomical region names.

    Args:
        contacts_ras: (N, 3) array of contact positions in RAS world space.
        parcellation: 3-D int32 array of SynthSeg labels (Slicer K,J,I order).
        parcellation_affine: 4x4 voxel-to-world matrix of the parcellation
            volume (NIfTI convention: maps I,J,K -> R,A,S).
        search_radius_mm: Radius for nearest-cortical-label search when a
            contact lands in white matter.

    Returns:
        List of region name strings, one per contact.

    Example::

        labels = label_contacts(
            np.array([[10.0, 20.0, 30.0]]),
            parcellation_array,
            parcellation_affine,
        )
    """
    inv_affine = np.linalg.inv(parcellation_affine)
    voxel_sizes = np.sqrt((parcellation_affine[:3, :3] ** 2).sum(axis=0))
    search_radius_vox = search_radius_mm / float(voxel_sizes.min())

    results: list[str] = []
    for ras in contacts_ras:
        # RAS -> IJK (NIfTI convention)
        ijk_h = inv_affine @ np.append(ras, 1.0)
        ijk = np.round(ijk_h[:3]).astype(int)

        # Parcellation is stored in Slicer (K,J,I) order, so index as [K,J,I]
        k, j, i = ijk[2], ijk[1], ijk[0]

        # Bounds check
        if (
            k < 0
            or k >= parcellation.shape[0]
            or j < 0
            or j >= parcellation.shape[1]
            or i < 0
            or i >= parcellation.shape[2]
        ):
            results.append("Outside brain")
            continue

        label = int(parcellation[k, j, i])

        if label == 0:
            results.append("Outside brain")
        elif label in _WM_LABELS:
            nearest = _nearest_cortical_label(parcellation, k, j, i, search_radius_vox)
            if nearest is not None:
                results.append(
                    f"WM near {SYNTHSEG_LUT.get(nearest, f'Label {nearest}')}"
                )
            else:
                results.append(SYNTHSEG_LUT.get(label, f"Label {label}"))
        else:
            results.append(SYNTHSEG_LUT.get(label, f"Label {label}"))

    return results


def _nearest_cortical_label(
    parcellation: np.ndarray,
    k: int,
    j: int,
    i: int,
    radius_vox: float,
) -> int | None:
    """Search a sphere around (k,j,i) for the nearest cortical parcel label.

    Returns the label integer, or None if no cortical parcel is found within radius.
    """
    r = int(np.ceil(radius_vox))
    k_lo, k_hi = max(0, k - r), min(parcellation.shape[0], k + r + 1)
    j_lo, j_hi = max(0, j - r), min(parcellation.shape[1], j + r + 1)
    i_lo, i_hi = max(0, i - r), min(parcellation.shape[2], i + r + 1)

    patch = parcellation[k_lo:k_hi, j_lo:j_hi, i_lo:i_hi]
    coords = np.argwhere(patch)  # (N, 3) in patch-local coordinates

    best_label: int | None = None
    best_dist = float("inf")

    center = np.array([k - k_lo, j - j_lo, i - i_lo], dtype=float)
    for idx in range(len(coords)):
        lbl = int(patch[coords[idx, 0], coords[idx, 1], coords[idx, 2]])
        if not _is_cortical_parcel(lbl):
            continue
        dist = float(np.linalg.norm(coords[idx] - center))
        if dist <= radius_vox and dist < best_dist:
            best_dist = dist
            best_label = lbl

    return best_label
