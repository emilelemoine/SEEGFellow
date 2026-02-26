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
