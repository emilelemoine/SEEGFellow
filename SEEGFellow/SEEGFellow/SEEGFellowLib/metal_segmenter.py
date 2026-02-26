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
            elongation = sorted_dims[-1] / sorted_dims[0] if sorted_dims[0] > 0 else 1.0
            if elongation < min_elongation_ratio:
                continue

        result[component] = 1

    return result
