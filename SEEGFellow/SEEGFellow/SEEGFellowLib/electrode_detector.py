# SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py
"""Automated electrode detection from metal segmentation.

Core algorithm functions operate on numpy arrays. The ElectrodeDetector class
wraps these for use with Slicer volume nodes.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from SEEGFellowLib.electrode_model import Contact, Electrode, ElectrodeParams


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


def merge_collinear_clusters(
    clusters: list[np.ndarray],
    angle_tolerance: float = 10.0,
    max_gap_mm: float = 30.0,
) -> list[np.ndarray]:
    """Merge clusters that share a collinear trajectory (fragments of a gapped electrode).

    Args:
        clusters: List of (N, 3) coordinate arrays.
        angle_tolerance: Maximum angle in degrees between axes to merge.
        max_gap_mm: Maximum distance between closest points of two clusters to merge.

    Returns:
        List of (M, 3) arrays after merging.

    Example::

        merged = merge_collinear_clusters(clusters, angle_tolerance=10.0)
    """
    if len(clusters) <= 1:
        return clusters

    # Compute axis for each cluster
    axes = []
    centers = []
    for cluster in clusters:
        if len(cluster) < 3:
            axes.append(None)
            centers.append(cluster.mean(axis=0))
        else:
            center, direction = fit_electrode_axis(cluster)
            axes.append(direction)
            centers.append(center)

    # Union-find for merging
    parent = list(range(len(clusters)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    angle_threshold_rad = np.radians(angle_tolerance)

    for i in range(len(clusters)):
        if axes[i] is None:
            continue
        for j in range(i + 1, len(clusters)):
            if axes[j] is None:
                continue
            # Check angle between axes
            cos_angle = abs(np.dot(axes[i], axes[j]))
            cos_angle = min(cos_angle, 1.0)
            angle = np.arccos(cos_angle)
            if angle > angle_threshold_rad:
                continue

            # Check distance: closest points between clusters
            dist = np.linalg.norm(centers[i] - centers[j])
            # Estimate extent of each cluster along its axis
            proj_i = np.dot(clusters[i] - centers[i], axes[i])
            proj_j = np.dot(clusters[j] - centers[j], axes[j])
            extent_i = proj_i.max() - proj_i.min()
            extent_j = proj_j.max() - proj_j.min()
            # Gap is center-to-center minus half-extents
            gap = dist - (extent_i + extent_j) / 2
            if gap < max_gap_mm:
                union(i, j)

    # Build merged clusters
    groups: dict[int, list[int]] = {}
    for i in range(len(clusters)):
        root = find(i)
        groups.setdefault(root, []).append(i)

    merged = []
    for indices in groups.values():
        merged.append(np.vstack([clusters[i] for i in indices]))

    return merged


def analyze_spacing(
    contact_positions: np.ndarray,
    gap_ratio_threshold: float = 1.8,
) -> dict:
    """Analyze inter-contact spacing to detect gaps.

    Args:
        contact_positions: Sorted 1D array of contact positions along axis.
        gap_ratio_threshold: If (long spacing / short spacing) exceeds this,
            classify as gapped electrode.

    Returns:
        Dict with keys: contact_spacing, has_gaps, gap_spacing, contacts_per_group.

    Example::

        info = analyze_spacing(np.array([0.0, 3.5, 7.0, 10.5, 24.5, 28.0]))
    """
    if len(contact_positions) < 2:
        return {
            "contact_spacing": 0.0,
            "has_gaps": False,
            "gap_spacing": None,
            "contacts_per_group": None,
        }

    spacings = np.diff(contact_positions)
    median_spacing = np.median(spacings)

    # Check for bimodal spacing
    short_mask = spacings < median_spacing * gap_ratio_threshold
    long_mask = ~short_mask

    if np.any(long_mask) and np.any(short_mask):
        short_spacings = spacings[short_mask]
        long_spacings = spacings[long_mask]
        ratio = np.median(long_spacings) / np.median(short_spacings)
        if ratio > gap_ratio_threshold:
            # Count contacts per group: find runs of short spacings
            gap_indices = np.where(long_mask)[0]
            group_sizes = []
            prev = 0
            for gi in gap_indices:
                group_sizes.append(gi - prev + 1)
                prev = gi + 1
            group_sizes.append(len(contact_positions) - prev)
            return {
                "contact_spacing": float(np.median(short_spacings)),
                "has_gaps": True,
                "gap_spacing": float(np.median(long_spacings)),
                "contacts_per_group": int(np.median(group_sizes)),
            }

    return {
        "contact_spacing": float(median_spacing),
        "has_gaps": False,
        "gap_spacing": None,
        "contacts_per_group": None,
    }


def orient_deepest_first(
    contact_positions: np.ndarray,
    axis_origin: np.ndarray,
    axis_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Orient contacts so index 1 = deepest (closest to brain center at origin 0,0,0).

    Args:
        contact_positions: Sorted 1D array of positions along axis.
        axis_origin: 3D center of the electrode cluster.
        axis_direction: Unit vector along electrode axis.

    Returns:
        (sorted_positions, oriented_direction): positions sorted deepest-first,
        and direction pointing from deepest to most lateral.
    """
    # Compute RAS coordinates of first and last contact
    first_ras = axis_origin + contact_positions[0] * axis_direction
    last_ras = axis_origin + contact_positions[-1] * axis_direction

    # Deepest = closest to brain center (0,0,0)
    if np.linalg.norm(first_ras) > np.linalg.norm(last_ras):
        # Reverse: last is deeper
        return contact_positions[::-1] - contact_positions[-1], -axis_direction
    else:
        return contact_positions - contact_positions[0], axis_direction


def detect_electrodes(
    contact_centers: np.ndarray,
    min_contacts: int = 3,
    expected_spacing: float = 3.5,
    collinearity_tolerance: float = 10.0,
    gap_ratio_threshold: float = 1.8,
    spacing_cutoff_factor: float = 0.65,
) -> list[Electrode]:
    """Full detection pipeline from pre-detected contact centers.

    Each row in contact_centers is one contact center in RAS coordinates.
    Steps: cluster -> merge collinear -> fit axis -> orient -> build Electrode.

    Args:
        contact_centers: (N, 3) array of contact center RAS coordinates.
        min_contacts: Minimum contacts to accept an electrode candidate.
        expected_spacing: Expected contact spacing in mm.
        collinearity_tolerance: Max angle for merging collinear fragments.
        gap_ratio_threshold: Threshold for gap detection.
        spacing_cutoff_factor: Fraction of expected_spacing below which a cluster's
            contact spacing is considered implausibly small and rejected.
            Default 0.65 preserves the previous hard-coded behavior.

    Returns:
        List of Electrode objects with auto-numbered contacts (unnamed).

    Example::

        electrodes = detect_electrodes(contact_centers_ras)
    """
    # 1. Cluster centers into electrode candidates
    clusters = cluster_into_electrodes(
        contact_centers, distance_threshold=expected_spacing * 2
    )

    # 2. Merge collinear fragments (for gapped electrodes)
    clusters = merge_collinear_clusters(
        clusters, angle_tolerance=collinearity_tolerance
    )

    electrodes = []
    for cluster in clusters:
        if len(cluster) < min_contacts:
            continue

        # 3. Fit axis through centers
        center, direction = fit_electrode_axis(cluster)

        # 4. Project centers onto axis — each projection IS a contact position
        projections = np.dot(cluster - center, direction)
        sorted_indices = np.argsort(projections)
        sorted_projections = projections[sorted_indices]

        # 5. Analyze spacing
        spacing_info = analyze_spacing(sorted_projections, gap_ratio_threshold)

        # Reject clusters whose spacing is implausibly small (scattered noise)
        if (
            0
            < spacing_info["contact_spacing"]
            < expected_spacing * spacing_cutoff_factor
        ):
            continue

        # 6. Orient deepest first
        sorted_positions, oriented_dir = orient_deepest_first(
            sorted_projections, center, direction
        )

        # 7. Build Electrode
        params = ElectrodeParams(
            contact_length=2.0,
            contact_spacing=spacing_info["contact_spacing"],
            contact_diameter=0.8,
            gap_spacing=spacing_info["gap_spacing"],
            contacts_per_group=spacing_info["contacts_per_group"],
        )

        contacts = []
        for i, pos in enumerate(sorted_positions):
            ras = center + pos * oriented_dir
            contacts.append(Contact(index=i + 1, position_ras=tuple(ras)))

        electrode = Electrode(
            name="",
            params=params,
            contacts=contacts,
            trajectory_direction=tuple(oriented_dir),
        )
        electrodes.append(electrode)

    return electrodes


class ElectrodeDetector:
    """Slicer wrapper: detects all electrodes from a metal segmentation volume.

    Example (in Slicer Python console)::

        detector = ElectrodeDetector()
        electrodes = detector.detect_all(metal_labelmap_node, ct_volume_node)
    """

    def __init__(
        self,
        min_contacts: int = 3,
        expected_spacing: float = 3.5,
        collinearity_tolerance: float = 10.0,
        gap_ratio_threshold: float = 1.8,
    ):
        self.min_contacts = min_contacts
        self.expected_spacing = expected_spacing
        self.collinearity_tolerance = collinearity_tolerance
        self.gap_ratio_threshold = gap_ratio_threshold

    def detect_all(
        self, ct_volume_node, threshold: float = 2500, sigma: float = 1.2
    ) -> list[Electrode]:
        """From a CT volume, detect all electrodes using LoG contact center detection.

        Pipeline: head mask -> threshold within mask -> LoG blob detection -> electrode grouping.

        Args:
            ct_volume_node: vtkMRMLScalarVolumeNode with the post-implant CT.
            threshold: HU threshold for metal (default 2500).
            sigma: LoG scale in voxels for contact detection.

        Returns:
            List of Electrode objects.

        Example::

            detector = ElectrodeDetector()
            electrodes = detector.detect_all(ct_volume_node, threshold=2500, sigma=1.2)
        """
        from slicer.util import arrayFromVolume
        from SEEGFellowLib.metal_segmenter import (
            compute_head_mask,
            threshold_volume,
            detect_contact_centers,
        )

        ct_array = arrayFromVolume(ct_volume_node)
        head_mask = compute_head_mask(ct_array)
        metal_mask = threshold_volume(ct_array, threshold) & head_mask

        # LoG blob detection for contact centers (IJK coordinates)
        centers_ijk = detect_contact_centers(ct_array, metal_mask, sigma=sigma)
        if len(centers_ijk) == 0:
            return []

        # Convert IJK to RAS
        ijk_to_ras = self._get_ijk_to_ras_matrix(ct_volume_node)
        ones = np.ones((len(centers_ijk), 1))
        ijk_h = np.hstack([centers_ijk.astype(float), ones])
        ras_h = (ijk_to_ras @ ijk_h.T).T
        centers_ras = ras_h[:, :3]

        return detect_electrodes(
            centers_ras,
            min_contacts=self.min_contacts,
            expected_spacing=self.expected_spacing,
            collinearity_tolerance=self.collinearity_tolerance,
            gap_ratio_threshold=self.gap_ratio_threshold,
        )

    @staticmethod
    def _get_ijk_to_ras_matrix(volume_node) -> np.ndarray:
        """Get 4x4 IJK-to-RAS matrix from a volume node."""
        import vtk

        mat = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(mat)
        return np.array([[mat.GetElement(i, j) for j in range(4)] for i in range(4)])
