# SEEGFellow/SEEGFellow/SEEGFellowLib/metal_segmenter.py
"""Metal segmentation from CT volume.

Core algorithm functions operate on numpy arrays for testability.
The MetalSegmenter class wraps these for use with Slicer volume nodes.
"""

from __future__ import annotations

import os

import numpy as np
from scipy import ndimage

try:
    from deepbet import run_bet  # noqa: F401 – imported for patchability in tests
except ImportError:  # deepbet not available (e.g. CI without GPU dependencies)
    run_bet = None  # type: ignore[assignment]


def threshold_volume(volume: np.ndarray, threshold: float = 2500) -> np.ndarray:
    """Threshold a CT volume to isolate high-intensity voxels.

    Example::

        mask = threshold_volume(ct_array, threshold=2500)
    """
    return (volume >= threshold).astype(np.uint8)


def compute_brain_mask(
    volume: np.ndarray,
    affine: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Create a binary mask of brain parenchyma from a T1-weighted MRI.

    Uses deepbet (CNN-based skull stripping) for robust brain extraction.
    The volume is saved to a temporary NIfTI file, processed by deepbet,
    and the resulting mask is loaded back as a numpy array.

    Args:
        volume: T1 MRI array (3-D numpy, arbitrary intensity scale).
        affine: 4x4 voxel-to-world (IJK-to-RAS) transformation matrix.
        threshold: deepbet segmentation threshold (0-1, default 0.5).

    Returns:
        Binary uint8 mask (1 = brain parenchyma).

    Raises:
        RuntimeError: If the resulting brain mask is empty.

    Example::

        brain = compute_brain_mask(t1_array, affine)
    """
    import tempfile
    import nibabel as nib

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, "t1.nii.gz")
        brain_path = os.path.join(tmp_dir, "brain.nii.gz")
        mask_path = os.path.join(tmp_dir, "mask.nii.gz")
        tiv_path = os.path.join(tmp_dir, "tiv.csv")

        nib.save(nib.Nifti1Image(volume, affine), input_path)

        run_bet(
            [input_path],
            [brain_path],
            [mask_path],
            [tiv_path],
            threshold=threshold,
            n_dilate=0,
            no_gpu=True,
        )

        mask_img = nib.load(mask_path)
        mask = np.asarray(mask_img.dataobj, dtype=np.uint8)

    if not np.any(mask):
        raise RuntimeError("Brain mask is empty – deepbet produced no output.")

    return (mask > 0).astype(np.uint8)


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
