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


def _otsu_threshold(volume: np.ndarray, nbins: int = 256) -> float:
    """Compute Otsu's threshold for a volume.

    Args:
        volume: Input array (any dtype, converted to float internally).
        nbins: Number of histogram bins.

    Returns:
        Optimal threshold value that maximises inter-class variance.
    """
    vmin, vmax = float(volume.min()), float(volume.max())
    if vmax == vmin:
        return vmin
    hist, bin_edges = np.histogram(volume.ravel(), bins=nbins, range=(vmin, vmax))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return vmin

    w0 = np.cumsum(hist)
    w1 = total - w0
    mu0_num = np.cumsum(hist * bin_centers)

    mu0 = np.divide(mu0_num, w0, out=np.zeros_like(w0), where=w0 > 0)
    mu_total = mu0_num[-1]
    mu1 = np.divide(mu_total - mu0_num, w1, out=np.zeros_like(w1), where=w1 > 0)

    variance = w0 * w1 * (mu0 - mu1) ** 2
    idx = int(np.argmax(variance))
    return float(bin_centers[idx])


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask."""
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask.astype(np.uint8)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return (labeled == largest).astype(np.uint8)


def _spherical_structuring_element(
    radius_mm: float,
    voxel_size_mm: tuple[float, float, float],
) -> np.ndarray:
    """Create an approximately spherical binary structuring element.

    Accounts for anisotropic voxel spacing so that the physical radius
    is honoured in every direction.

    Args:
        radius_mm: Desired radius in millimetres.
        voxel_size_mm: (spacing_i, spacing_j, spacing_k) in mm.

    Returns:
        Binary uint8 array usable as a structuring element.
    """
    radii_vox = tuple(max(1, int(round(radius_mm / s))) for s in voxel_size_mm)
    grid = np.mgrid[
        -radii_vox[0] : radii_vox[0] + 1,
        -radii_vox[1] : radii_vox[1] + 1,
        -radii_vox[2] : radii_vox[2] + 1,
    ]
    # Normalise each axis so that 1.0 corresponds to the target radius
    dist = np.sqrt(
        (grid[0] * voxel_size_mm[0] / radius_mm) ** 2
        + (grid[1] * voxel_size_mm[1] / radius_mm) ** 2
        + (grid[2] * voxel_size_mm[2] / radius_mm) ** 2
    )
    return (dist <= 1.0).astype(np.uint8)


def compute_brain_mask(
    volume: np.ndarray,
    voxel_size_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    erosion_mm: float = 12.0,
    dilation_mm: float = 10.0,
) -> np.ndarray:
    """Create a binary mask of brain parenchyma from a T1-weighted MRI.

    Algorithm
    ---------
    1. Otsu threshold to separate head from background air.
    2. Largest connected component + hole-filling → head mask.
    3. Erode by *erosion_mm* (via distance transform) to strip scalp,
       skull and meninges.
    4. Largest connected component of eroded volume → brain core seed.
    5. Dilate brain core by *dilation_mm* (via distance transform),
       intersected with the head mask, to recover the brain boundary
       without leaking through skull.
    6. Final hole-fill.

    The slight deficit (erosion > dilation) provides a conservative mask
    that stays within the inner table of the skull.

    Args:
        volume: T1 MRI array (3-D numpy, arbitrary intensity scale).
        voxel_size_mm: Physical voxel dimensions (I, J, K) in mm.
        erosion_mm: Erosion radius; must exceed the combined thickness
            of scalp + skull (~10–14 mm in adults).
        dilation_mm: Dilation radius to recover brain surface after
            erosion.  Slightly less than *erosion_mm* to stay
            conservatively inside the skull.

    Returns:
        Binary uint8 mask (1 = brain parenchyma).

    Example::

        brain = compute_brain_mask(t1_array, voxel_size_mm=(1.0, 1.0, 1.0))
    """
    # ---- Step 1: Otsu threshold -----------------------------------------
    threshold = _otsu_threshold(volume)
    foreground = volume > threshold

    # ---- Step 2: Head mask (largest CC, filled) --------------------------
    filled = ndimage.binary_fill_holes(foreground)
    head_mask = _largest_connected_component(filled.astype(np.uint8)).astype(bool)
    head_mask = ndimage.binary_fill_holes(head_mask)

    # ---- Step 3: Erode to strip skull + scalp ---------------------------
    # Distance from each True voxel to the nearest False voxel (boundary).
    # Thresholding at erosion_mm is equivalent to a spherical erosion.
    dist_inside = ndimage.distance_transform_edt(head_mask, sampling=voxel_size_mm)
    brain_core = dist_inside >= erosion_mm

    # ---- Step 4: Largest CC of eroded mask → brain seed -----------------
    brain_core = _largest_connected_component(brain_core.astype(np.uint8)).astype(bool)
    if not np.any(brain_core):
        # Erosion was too aggressive; fall back to head mask
        return head_mask.astype(np.uint8)

    # ---- Step 5: Dilate back, constrained to head mask ------------------
    # Distance from each False voxel to the nearest True voxel in brain_core.
    # Thresholding the inverse EDT is equivalent to spherical dilation.
    dist_outside = ndimage.distance_transform_edt(~brain_core, sampling=voxel_size_mm)
    brain_mask = (dist_outside <= dilation_mm) & head_mask

    # ---- Step 6: Fill residual holes ------------------------------------
    brain_mask = ndimage.binary_fill_holes(brain_mask)

    return brain_mask.astype(np.uint8)


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
