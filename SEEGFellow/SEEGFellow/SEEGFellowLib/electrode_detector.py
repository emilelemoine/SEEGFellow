# SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py
"""Automated electrode detection from metal segmentation.

Core algorithm functions operate on numpy arrays. The ElectrodeDetector class
wraps these for use with Slicer volume nodes.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from SEEGFellowLib.electrode_model import Contact, Electrode, ElectrodeParams


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
    brain_centroid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Orient contacts so index 0 = deepest (closest to brain center).

    Args:
        contact_positions: Sorted 1D array of projection positions along axis.
        axis_origin: 3D center of the electrode cluster (PCA mean).
        axis_direction: Unit vector along electrode axis.
        brain_centroid: 3D RAS coordinates of brain center. If None, falls back
            to (0, 0, 0).

    Returns:
        (sorted_indices, oriented_direction): integer indices into
        contact_positions sorted deepest-first, and direction vector pointing
        from deepest to most lateral.
    """
    if brain_centroid is None:
        brain_centroid = np.zeros(3)

    first_ras = axis_origin + contact_positions[0] * axis_direction
    last_ras = axis_origin + contact_positions[-1] * axis_direction

    indices = np.arange(len(contact_positions))
    if np.linalg.norm(first_ras - brain_centroid) > np.linalg.norm(
        last_ras - brain_centroid
    ):
        return indices[::-1], -axis_direction
    return indices, axis_direction


def ransac_group_contacts(
    contact_coords: np.ndarray,
    expected_spacing: float = 3.5,
    distance_tolerance: float = 2.0,
    max_iterations: int = 1000,
    min_contacts: int = 3,
    spacing_low_factor: float = 0.5,
    spacing_high_factor: float = 2.0,
    random_seed: int | None = None,
) -> list[np.ndarray]:
    """Group contact centers into electrodes using iterative RANSAC line fitting.

    Repeatedly fits lines through the point cloud. Each iteration samples two
    points, finds all contacts within ``distance_tolerance`` perpendicular
    distance of that line, validates spacing, and claims the best-supported
    line's inliers. Claimed contacts are removed and the process repeats.

    Args:
        contact_coords: (N, 3) array of contact center RAS coordinates.
        expected_spacing: Expected contact spacing in mm.
        distance_tolerance: Max perpendicular distance (mm) to count as inlier.
        max_iterations: RANSAC trials per electrode.
        min_contacts: Minimum inliers to accept an electrode.
        spacing_low_factor: Lower bound on median spacing as fraction of
            expected_spacing (e.g. 0.5 means >= 1.75 mm for 3.5 mm spacing).
        spacing_high_factor: Upper bound on median spacing as fraction of
            expected_spacing (e.g. 2.0 means <= 7.0 mm for 3.5 mm spacing).
        random_seed: Optional seed for reproducibility.

    Returns:
        List of (M, 3) arrays, one per electrode. Coordinates are the original
        input values (not reprojected).

    Example::

        groups = ransac_group_contacts(centers_ras, expected_spacing=3.5)
    """
    if len(contact_coords) < min_contacts:
        return []

    rng = np.random.default_rng(random_seed)
    pool_indices = np.arange(len(contact_coords))
    coords = contact_coords.copy()
    groups: list[np.ndarray] = []

    spacing_lo = expected_spacing * spacing_low_factor
    spacing_hi = expected_spacing * spacing_high_factor

    while len(pool_indices) >= min_contacts:
        best_inlier_idx = None
        best_count = 0

        pool_coords = coords[pool_indices]

        for _ in range(max_iterations):
            # Sample 2 distinct points
            sample = rng.choice(len(pool_coords), size=2, replace=False)
            p1, p2 = pool_coords[sample[0]], pool_coords[sample[1]]
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length < 1e-8:
                continue
            direction /= length

            # Perpendicular distance of all pool points to line through p1
            diff = pool_coords - p1
            proj = np.dot(diff, direction)
            perp = diff - proj[:, np.newaxis] * direction
            dists = np.linalg.norm(perp, axis=1)

            inlier_mask = dists < distance_tolerance
            if np.sum(inlier_mask) < min_contacts:
                continue

            # Deduplicate: if two inliers project to nearly the same position
            # on the line, keep only the one closest to the line.
            inlier_local = np.where(inlier_mask)[0]
            inlier_proj_vals = proj[inlier_local]
            inlier_dists = dists[inlier_local]
            sort_order = np.argsort(inlier_proj_vals)
            inlier_local = inlier_local[sort_order]
            inlier_proj_vals = inlier_proj_vals[sort_order]
            inlier_dists = inlier_dists[sort_order]
            keep = np.ones(len(inlier_local), dtype=bool)
            for k in range(1, len(inlier_local)):
                if (
                    abs(inlier_proj_vals[k] - inlier_proj_vals[k - 1])
                    < distance_tolerance
                ):
                    # Near-duplicate projection: keep the one closer to the line
                    if inlier_dists[k] < inlier_dists[k - 1]:
                        keep[k - 1] = False
                    else:
                        keep[k] = False
            inlier_local = inlier_local[keep]
            n_inliers = len(inlier_local)
            if n_inliers < min_contacts:
                continue

            # Validate spacing: median neighbor distance along the line
            inlier_proj = inlier_proj_vals[keep]
            spacings = np.diff(inlier_proj)
            if len(spacings) == 0:
                continue
            median_sp = np.median(spacings)
            if not (spacing_lo <= median_sp <= spacing_hi):
                continue

            if n_inliers > best_count:
                best_count = n_inliers
                best_inlier_idx = inlier_local

        if best_inlier_idx is None:
            break

        # Refit axis on RANSAC inliers, then re-check only those same inliers.
        # We do NOT expand the inlier set during refit — that would risk pulling
        # in contacts from a crossing electrode that happens to be close.
        inlier_global = pool_indices[best_inlier_idx]
        inlier_coords = coords[inlier_global]
        center, direction = fit_electrode_axis(inlier_coords)
        diff = inlier_coords - center
        proj = np.dot(diff, direction)
        perp = diff - proj[:, np.newaxis] * direction
        dists = np.linalg.norm(perp, axis=1)
        refined_local_mask = dists < distance_tolerance

        # Re-validate spacing after refit; if valid, drop any inliers that
        # no longer fit the refined axis (can only shrink the set).
        refined_proj = np.sort(proj[refined_local_mask])
        refined_spacings = np.diff(refined_proj)
        if len(refined_spacings) > 0:
            refined_median = np.median(refined_spacings)
            if spacing_lo <= refined_median <= spacing_hi:
                best_inlier_idx = best_inlier_idx[refined_local_mask]

        claimed_global = pool_indices[best_inlier_idx]
        groups.append(coords[claimed_global])
        pool_indices = np.setdiff1d(pool_indices, claimed_global)

    return groups


def detect_electrodes(
    contact_centers: np.ndarray,
    min_contacts: int = 3,
    expected_spacing: float = 3.5,
    distance_tolerance: float = 2.0,
    max_iterations: int = 1000,
    gap_ratio_threshold: float = 1.8,
    spacing_cutoff_factor: float = 0.65,
    brain_centroid: np.ndarray | None = None,
) -> list[Electrode]:
    """Full detection pipeline: RANSAC grouping + axis fitting + orientation.

    Each row in contact_centers is one LoG-detected contact in RAS coordinates.
    Original positions are preserved (no reprojection onto fitted axis).

    Args:
        contact_centers: (N, 3) array of contact center RAS coordinates.
        min_contacts: Minimum contacts to accept an electrode.
        expected_spacing: Expected contact spacing in mm.
        distance_tolerance: RANSAC perpendicular distance tolerance in mm.
        max_iterations: RANSAC trials per electrode.
        gap_ratio_threshold: Threshold for gap detection in spacing analysis.
        spacing_cutoff_factor: Fraction of expected_spacing below which a
            cluster is rejected as noise.
        brain_centroid: 3D RAS coordinates of brain center for orientation.
            Falls back to (0, 0, 0) if None.

    Returns:
        List of Electrode objects with auto-numbered contacts (unnamed).
        Contact positions are the original input coordinates.

    Example::

        electrodes = detect_electrodes(centers_ras, brain_centroid=centroid)
    """
    groups = ransac_group_contacts(
        contact_centers,
        expected_spacing=expected_spacing,
        distance_tolerance=distance_tolerance,
        max_iterations=max_iterations,
        min_contacts=min_contacts,
    )

    electrodes = []
    for group_coords in groups:
        # Fit axis
        center, direction = fit_electrode_axis(group_coords)

        # Project onto axis, sort
        projections = np.dot(group_coords - center, direction)
        sorted_indices = np.argsort(projections)
        sorted_projections = projections[sorted_indices]

        # Analyze spacing
        spacing_info = analyze_spacing(sorted_projections, gap_ratio_threshold)

        # Reject implausibly small spacing
        if (
            0
            < spacing_info["contact_spacing"]
            < expected_spacing * spacing_cutoff_factor
        ):
            continue

        # Orient deepest first
        orient_indices, oriented_dir = orient_deepest_first(
            sorted_projections, center, direction, brain_centroid=brain_centroid
        )
        # Map back through sorted_indices
        final_indices = sorted_indices[orient_indices]

        # Build Electrode with original coordinates
        params = ElectrodeParams(
            contact_length=2.0,
            contact_spacing=spacing_info["contact_spacing"],
            contact_diameter=0.8,
            gap_spacing=spacing_info["gap_spacing"],
            contacts_per_group=spacing_info["contacts_per_group"],
        )

        contacts = []
        for i, idx in enumerate(final_indices):
            contacts.append(Contact(index=i + 1, position_ras=tuple(group_coords[idx])))

        electrode = Electrode(
            name="",
            params=params,
            contacts=contacts,
            trajectory_direction=tuple(oriented_dir),
        )
        electrodes.append(electrode)

    return electrodes


def _filter_contact_mask(
    metal_mask: np.ndarray,
    min_voxels: int = 3,
    max_voxels: int = 500,
) -> np.ndarray:
    """Keep connected components whose size falls within [min_voxels, max_voxels].

    Removes components that are too small (noise) or too large (entry bolts,
    bone screws) to be SEEG electrode contacts.

    Args:
        metal_mask: Binary uint8 mask of candidate metal voxels.
        min_voxels: Minimum component size to keep.
        max_voxels: Maximum component size to keep.

    Returns:
        Filtered binary mask of the same shape.

    Example::

        contact_mask = _filter_contact_mask(metal_mask, min_voxels=3, max_voxels=500)
    """
    from scipy import ndimage as _ndi

    labeled, n_comp = _ndi.label(metal_mask)
    if n_comp == 0:
        return np.zeros_like(metal_mask)
    comp_sizes = _ndi.sum(metal_mask, labeled, range(1, n_comp + 1))
    result = np.zeros_like(metal_mask)
    for idx, size in enumerate(comp_sizes):
        if min_voxels <= size <= max_voxels:
            result[labeled == (idx + 1)] = 1
    return result


class ElectrodeDetector:
    """Slicer wrapper: detects all electrodes from a metal segmentation volume.

    Example (in Slicer Python console)::

        detector = ElectrodeDetector()
        electrodes = detector.detect_all(ct_volume_node, metal_mask)
    """

    def __init__(
        self,
        min_contacts: int = 3,
        expected_spacing: float = 3.5,
        distance_tolerance: float = 2.0,
        max_iterations: int = 1000,
        gap_ratio_threshold: float = 1.8,
        spacing_cutoff_factor: float = 0.65,
    ):
        self.min_contacts = min_contacts
        self.expected_spacing = expected_spacing
        self.distance_tolerance = distance_tolerance
        self.max_iterations = max_iterations
        self.gap_ratio_threshold = gap_ratio_threshold
        self.spacing_cutoff_factor = spacing_cutoff_factor

    def detect_all(
        self,
        ct_volume_node,
        metal_mask: np.ndarray,
        sigma: float = 1.2,
        max_component_voxels: int = 500,
        brain_centroid: np.ndarray | None = None,
    ) -> list[Electrode]:
        """From a CT volume and pre-computed metal mask, detect all electrodes.

        Pipeline: filter metal mask by component size -> LoG blob detection -> electrode grouping.

        Args:
            ct_volume_node: vtkMRMLScalarVolumeNode with the post-implant CT.
            metal_mask: Pre-computed binary metal mask (numpy array, same shape as CT).
            sigma: LoG scale in voxels for contact detection.
            max_component_voxels: Maximum connected-component size (voxels) to keep
                as a contact candidate. Larger components are likely bolts or screws.
            brain_centroid: 3D RAS coordinates of brain center for contact orientation.
                Falls back to (0, 0, 0) if None.

        Returns:
            List of Electrode objects.

        Example::

            detector = ElectrodeDetector()
            electrodes = detector.detect_all(ct_volume_node, metal_mask, sigma=1.2)
        """
        from slicer.util import arrayFromVolume
        from SEEGFellowLib.metal_segmenter import detect_contact_centers

        ct_array = arrayFromVolume(ct_volume_node)

        contact_mask = _filter_contact_mask(
            metal_mask, min_voxels=3, max_voxels=max_component_voxels
        )
        print(
            f"[SEEGFellow] metal_mask nonzero={int(np.sum(metal_mask))}"
            f"  removed_voxels={int(np.sum(metal_mask) - np.sum(contact_mask))}"
            f"  (filtered to max {max_component_voxels} voxels)"
        )
        print(f"[SEEGFellow] contact_mask nonzero={int(np.sum(contact_mask))}")

        # LoG blob detection for contact centers (IJK coordinates)
        centers_ijk = detect_contact_centers(ct_array, contact_mask, sigma=sigma)
        if len(centers_ijk) == 0:
            return []

        # Convert IJK to RAS.
        # arrayFromVolume axes are (K, J, I); GetIJKToRASMatrix expects [I, J, K, 1].
        # Reverse the last axis order before the matrix multiply.
        ijk_to_ras = self._get_ijk_to_ras_matrix(ct_volume_node)
        centers_ijk_reordered = centers_ijk[:, ::-1]  # [k,j,i] → [i,j,k]
        ones = np.ones((len(centers_ijk_reordered), 1))
        ijk_h = np.hstack([centers_ijk_reordered.astype(float), ones])
        ras_h = (ijk_to_ras @ ijk_h.T).T
        centers_ras = ras_h[:, :3]
        print(
            f"[SEEGFellow] RAS range"
            f" R=[{centers_ras[:,0].min():.1f},{centers_ras[:,0].max():.1f}]"
            f" A=[{centers_ras[:,1].min():.1f},{centers_ras[:,1].max():.1f}]"
            f" S=[{centers_ras[:,2].min():.1f},{centers_ras[:,2].max():.1f}]"
        )

        electrodes = detect_electrodes(
            centers_ras,
            min_contacts=self.min_contacts,
            expected_spacing=self.expected_spacing,
            distance_tolerance=self.distance_tolerance,
            max_iterations=self.max_iterations,
            gap_ratio_threshold=self.gap_ratio_threshold,
            spacing_cutoff_factor=self.spacing_cutoff_factor,
            brain_centroid=brain_centroid,
        )
        return electrodes

    @staticmethod
    def _get_ijk_to_ras_matrix(volume_node) -> np.ndarray:
        """Get 4x4 IJK-to-world matrix from a volume node, including any
        parent registration transform applied in the Slicer scene."""
        import vtk

        mat = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(mat)
        intrinsic = np.array(
            [[mat.GetElement(i, j) for j in range(4)] for i in range(4)]
        )

        parent_node = volume_node.GetParentTransformNode()
        parent_4x4 = None
        if parent_node is not None:
            p_mat = vtk.vtkMatrix4x4()
            parent_node.GetMatrixTransformToWorld(p_mat)
            parent_4x4 = np.array(
                [[p_mat.GetElement(i, j) for j in range(4)] for i in range(4)]
            )

        return ElectrodeDetector._compose_ijk_to_world(intrinsic, parent_4x4)

    @staticmethod
    def _compose_ijk_to_world(
        intrinsic: np.ndarray, parent: np.ndarray | None
    ) -> np.ndarray:
        """Compose intrinsic IJK→RAS with an optional parent transform matrix.

        Args:
            intrinsic: 4×4 IJK-to-RAS matrix from the volume node itself.
            parent: 4×4 world transform matrix of the parent node, or None.

        Returns:
            4×4 IJK-to-world matrix: ``parent @ intrinsic`` when a parent
            exists, otherwise ``intrinsic`` unchanged.
        """
        if parent is None:
            return intrinsic
        return parent @ intrinsic
