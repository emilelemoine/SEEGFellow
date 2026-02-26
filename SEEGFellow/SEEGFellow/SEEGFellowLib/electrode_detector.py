# SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py
"""Automated electrode detection from metal segmentation.

Core algorithm functions operate on numpy arrays. The ElectrodeDetector class
wraps these for use with Slicer volume nodes.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from SEEGFellowLib.electrode_model import Contact, Electrode, ElectrodeParams


def extract_metal_coords(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> np.ndarray:
    """Convert binary mask voxels to RAS coordinates.

    Args:
        mask: Binary 3D array (IJK indexing).
        spacing: Voxel size in mm (I, J, K).
        origin: Volume origin in RAS.

    Returns:
        (N, 3) array of RAS coordinates.

    Example::

        coords = extract_metal_coords(mask, spacing=(0.5, 0.5, 0.5), origin=(0, 0, 0))
    """
    ijk = np.argwhere(mask > 0).astype(float)  # shape (N, 3)
    if len(ijk) == 0:
        return np.empty((0, 3))
    # Convert IJK to RAS (assuming RAS-aligned volume for simplicity;
    # full IJK-to-RAS transform handled in the Slicer wrapper)
    ras = ijk * np.array(spacing) + np.array(origin)
    return ras


def cluster_into_electrodes(
    coords: np.ndarray,
    distance_threshold: float = 10.0,
) -> list[np.ndarray]:
    """Group metal coordinates into electrode candidates using connected components.

    Uses a voxel-grid approach: discretize coords, find connected components,
    then return coordinate arrays per component.

    Args:
        coords: (N, 3) array of RAS coordinates.
        distance_threshold: Max distance (mm) between voxels to consider connected.

    Returns:
        List of (M, 3) arrays, one per electrode candidate.

    Example::

        clusters = cluster_into_electrodes(all_metal_coords)
    """
    if len(coords) == 0:
        return []

    # Discretize into a grid with cell size = distance_threshold
    cell_size = distance_threshold
    grid_coords = np.floor(coords / cell_size).astype(int)

    # Shift to non-negative indices
    grid_min = grid_coords.min(axis=0)
    grid_coords -= grid_min

    # Create binary volume
    grid_max = grid_coords.max(axis=0)
    grid_shape = tuple(grid_max + 1)
    grid = np.zeros(grid_shape, dtype=np.uint8)

    for gc in grid_coords:
        grid[tuple(gc)] = 1

    # Connected components on the grid
    struct = ndimage.generate_binary_structure(3, 3)  # 26-connectivity
    labeled_grid, num_components = ndimage.label(grid, structure=struct)

    # Map each original point to its component
    labels = np.array([labeled_grid[tuple(gc)] for gc in grid_coords])

    clusters = []
    for comp_id in range(1, num_components + 1):
        cluster_mask = labels == comp_id
        clusters.append(coords[cluster_mask])

    return clusters


def fit_electrode_axis(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a line to electrode voxel positions via PCA.

    Args:
        coords: (N, 3) array of RAS coordinates.

    Returns:
        (center, direction): center is the mean point, direction is a unit vector
        along the first principal component.

    Example::

        center, direction = fit_electrode_axis(cluster_coords)
    """
    center = coords.mean(axis=0)
    centered = coords - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]  # first principal component
    direction /= np.linalg.norm(direction)
    return center, direction


def detect_contacts_along_axis(
    projections: np.ndarray,
    expected_spacing: float = 3.5,
    bin_width: float = 0.5,
    min_peak_height_fraction: float = 0.3,
) -> np.ndarray:
    """Detect contact positions from 1D projected metal voxel positions.

    Builds a density histogram along the electrode axis and finds peaks
    corresponding to individual contacts.

    Args:
        projections: 1D array of voxel positions projected onto the electrode axis.
        expected_spacing: Expected contact spacing in mm.
        bin_width: Histogram bin width in mm.
        min_peak_height_fraction: Minimum peak height as fraction of max peak.

    Returns:
        1D array of contact positions along the axis (sorted).

    Example::

        peaks = detect_contacts_along_axis(projected_positions, expected_spacing=3.5)
    """
    from scipy.signal import find_peaks

    if len(projections) == 0:
        return np.array([])

    # Build density histogram
    proj_min = projections.min() - expected_spacing
    proj_max = projections.max() + expected_spacing
    bins = np.arange(proj_min, proj_max + bin_width, bin_width)
    hist, bin_edges = np.histogram(projections, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peaks with minimum distance based on expected spacing
    min_distance = max(1, int(expected_spacing * 0.6 / bin_width))
    min_height = hist.max() * min_peak_height_fraction

    peaks_idx, _ = find_peaks(hist, distance=min_distance, height=min_height)
    peak_positions = bin_centers[peaks_idx]

    return np.sort(peak_positions)
