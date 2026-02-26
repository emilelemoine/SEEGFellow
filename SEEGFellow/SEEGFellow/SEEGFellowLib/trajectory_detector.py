# SEEGFellow/SEEGFellow/SEEGFellowLib/trajectory_detector.py
"""Single-electrode detection from a seed point (manual fallback).

Core functions operate on numpy arrays. The IntensityProfileDetector class
wraps these for Slicer.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from SEEGFellowLib.electrode_model import Contact, ElectrodeParams


def estimate_trajectory(
    metal_coords: np.ndarray,
    direction_hint: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate electrode trajectory from nearby metal voxels.

    Args:
        metal_coords: (N, 3) array of metal voxel coordinates.
        direction_hint: Optional unit vector hint for the trajectory direction.

    Returns:
        Unit vector along the electrode trajectory.

    Example::

        direction = estimate_trajectory(nearby_metal_coords)
    """
    if direction_hint is not None:
        hint = np.array(direction_hint, dtype=float)
        return hint / np.linalg.norm(hint)

    # PCA
    centered = metal_coords - metal_coords.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    direction /= np.linalg.norm(direction)
    return direction


def detect_contacts_from_intensity_profile(
    positions: np.ndarray,
    intensities: np.ndarray,
    num_contacts: int,
    expected_spacing: float = 3.5,
    snap_tolerance: float = 1.0,
) -> np.ndarray:
    """Detect contact positions from an intensity profile along the trajectory.

    Args:
        positions: 1D array of sample positions along trajectory (mm).
        intensities: 1D array of CT intensities at each position.
        num_contacts: Expected number of contacts.
        expected_spacing: Expected center-to-center spacing (mm).
        snap_tolerance: If peaks are within this distance of the expected
            grid, snap to the grid.

    Returns:
        Sorted 1D array of contact positions along the trajectory.

    Example::

        contacts = detect_contacts_from_intensity_profile(
            positions, intensities, num_contacts=8, expected_spacing=3.5
        )
    """
    step = positions[1] - positions[0] if len(positions) > 1 else 0.2
    min_distance = max(1, int(expected_spacing * 0.6 / step))

    peaks_idx, properties = find_peaks(
        intensities,
        distance=min_distance,
        height=np.median(intensities),
    )

    if len(peaks_idx) == 0:
        return np.array([])

    peak_positions = positions[peaks_idx]
    peak_heights = properties["peak_heights"]

    # Take the num_contacts strongest peaks
    if len(peak_positions) > num_contacts:
        top_indices = np.argsort(peak_heights)[-num_contacts:]
        peak_positions = np.sort(peak_positions[top_indices])
    else:
        peak_positions = np.sort(peak_positions)

    # Snap to expected grid if close enough
    if len(peak_positions) >= 2:
        grid_start = peak_positions[0]
        expected_grid = grid_start + np.arange(len(peak_positions)) * expected_spacing
        deviations = np.abs(peak_positions - expected_grid)
        if np.all(deviations < snap_tolerance):
            peak_positions = expected_grid

    return peak_positions


class IntensityProfileDetector:
    """Slicer wrapper: detects contacts along a single electrode from a seed point.

    Example (in Slicer Python console)::

        detector = IntensityProfileDetector()
        contacts = detector.detect(
            seed_ras=(10.0, 20.0, 30.0),
            ct_volume_node=ct_node,
            num_contacts=8,
            params=ElectrodeParams(contact_length=2.0, contact_spacing=3.5, contact_diameter=0.8),
        )
    """

    def detect(
        self,
        seed_ras: tuple[float, float, float],
        ct_volume_node,
        num_contacts: int,
        params: ElectrodeParams,
        direction_hint: tuple[float, float, float] | None = None,
        search_radius: float = 20.0,
        metal_threshold: float = 2500,
    ) -> list[Contact]:
        """Given a seed point, find contacts along an electrode.

        Args:
            seed_ras: (R, A, S) coordinates of the deepest contact.
            ct_volume_node: Slicer CT volume node.
            num_contacts: Expected number of contacts.
            params: Electrode physical parameters.
            direction_hint: Optional (R, A, S) direction hint.
            search_radius: Radius in mm for local neighborhood.
            metal_threshold: HU threshold for metal.

        Returns:
            List of Contact objects, numbered 1 (deepest) to num_contacts.
        """
        import slicer  # noqa: F401 - Slicer import (lazy)
        import vtk
        from scipy.ndimage import map_coordinates
        from slicer.util import arrayFromVolume

        seed = np.array(seed_ras)
        ct_array = arrayFromVolume(ct_volume_node)

        # Get RAS-to-IJK matrix
        ras_to_ijk = vtk.vtkMatrix4x4()
        ct_volume_node.GetRASToIJKMatrix(ras_to_ijk)

        ijk_to_ras = vtk.vtkMatrix4x4()
        ct_volume_node.GetIJKToRASMatrix(ijk_to_ras)

        # Convert seed to IJK
        seed_h = [*seed_ras, 1.0]
        seed_ijk = [0.0] * 4
        ras_to_ijk.MultiplyPoint(seed_h, seed_ijk)
        seed_ijk = np.array(seed_ijk[:3])

        # Get spacing for radius calculation
        spacing = np.array(ct_volume_node.GetSpacing())

        # Extract local neighborhood
        radius_voxels = (search_radius / spacing).astype(int)
        slices = []
        for dim in range(3):
            lo = max(0, int(seed_ijk[dim]) - radius_voxels[dim])
            hi = min(ct_array.shape[dim], int(seed_ijk[dim]) + radius_voxels[dim] + 1)
            slices.append(slice(lo, hi))

        local_ct = ct_array[slices[0], slices[1], slices[2]]

        # Threshold to isolate metal
        metal_mask = local_ct >= metal_threshold
        metal_ijk = np.argwhere(metal_mask).astype(float)

        if len(metal_ijk) == 0:
            return []

        # Convert local IJK back to RAS
        offset = np.array([s.start for s in slices])
        metal_ijk_global = metal_ijk + offset
        metal_ras = np.array(
            [
                [
                    sum(
                        ijk_to_ras.GetElement(r, c)
                        * (metal_ijk_global[j, c] if c < 3 else 1.0)
                        for c in range(4)
                    )
                    for r in range(3)
                ]
                for j in range(len(metal_ijk_global))
            ]
        )

        # Estimate trajectory
        hint = np.array(direction_hint) if direction_hint else None
        direction = estimate_trajectory(metal_ras, direction_hint=hint)

        # Orient direction away from brain center
        outward = seed + direction * 10
        inward = seed - direction * 10
        if np.linalg.norm(outward) < np.linalg.norm(inward):
            direction = -direction

        # Sample intensity profile along trajectory
        profile_length = (
            num_contacts - 1
        ) * params.contact_spacing + params.contact_spacing * 2
        sample_positions = np.arange(0, profile_length, 0.2)
        sample_ras = seed[:, None] + direction[:, None] * sample_positions[None, :]
        sample_ras = sample_ras.T  # (N, 3)

        # Convert sample points to IJK and interpolate
        sample_ijk = np.array(
            [
                [
                    sum(
                        ras_to_ijk.GetElement(r, c)
                        * (sample_ras[j, c] if c < 3 else 1.0)
                        for c in range(4)
                    )
                    for r in range(3)
                ]
                for j in range(len(sample_ras))
            ]
        )

        intensities = map_coordinates(
            ct_array.astype(float), sample_ijk.T, order=1, mode="constant", cval=0
        )

        # Detect contacts
        contact_offsets = detect_contacts_from_intensity_profile(
            sample_positions, intensities, num_contacts, params.contact_spacing
        )

        contacts = []
        for i, offset in enumerate(contact_offsets):
            pos = seed + direction * offset
            contacts.append(Contact(index=i + 1, position_ras=tuple(pos)))

        return contacts
