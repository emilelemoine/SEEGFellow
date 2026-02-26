# SEEGFellow/SEEGFellow/SEEGFellowLib/contact_segmenter.py
"""Creates a Slicer segmentation with per-contact cylindrical segments.

Example (in Slicer Python console)::

    segmenter = ContactSegmenter()
    seg_node = segmenter.create_segmentation(electrodes, ct_volume_node)
"""

from __future__ import annotations

import numpy as np
from SEEGFellowLib.electrode_model import Electrode


# Distinct colors for electrodes (RGB, 0-1)
ELECTRODE_COLORS = [
    (0.9, 0.2, 0.2),  # red
    (0.2, 0.6, 0.9),  # blue
    (0.2, 0.9, 0.3),  # green
    (0.9, 0.7, 0.1),  # yellow
    (0.7, 0.2, 0.9),  # purple
    (0.9, 0.5, 0.1),  # orange
    (0.1, 0.9, 0.8),  # cyan
    (0.9, 0.2, 0.6),  # pink
]


def generate_cylinder_mask(
    shape: tuple[int, int, int],
    center: np.ndarray,
    direction: np.ndarray,
    length: float,
    diameter: float,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> np.ndarray:
    """Generate a binary mask of a cylinder in a volume grid.

    Args:
        shape: Volume dimensions (I, J, K).
        center: Cylinder center in RAS coordinates.
        direction: Unit vector along cylinder axis.
        length: Cylinder length in mm.
        diameter: Cylinder diameter in mm.
        spacing: Voxel spacing (I, J, K) in mm.
        origin: Volume origin in RAS.

    Returns:
        Binary uint8 array of shape `shape`.

    Example::

        mask = generate_cylinder_mask(
            shape=(100, 100, 100),
            center=np.array([50.0, 50.0, 50.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            length=2.0, diameter=0.8,
            spacing=(0.5, 0.5, 0.5), origin=(0.0, 0.0, 0.0),
        )
    """
    radius = diameter / 2.0
    half_length = length / 2.0
    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction)

    mask = np.zeros(shape, dtype=np.uint8)

    # Create grid of RAS coordinates
    ii, jj, kk = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    ras_coords = np.stack(
        [
            ii * spacing[0] + origin[0],
            jj * spacing[1] + origin[1],
            kk * spacing[2] + origin[2],
        ],
        axis=-1,
    )  # shape (I, J, K, 3)

    # Vector from center to each voxel
    diff = ras_coords - center  # (I, J, K, 3)

    # Project onto axis
    along_axis = np.einsum("ijkd,d->ijk", diff, direction)

    # Perpendicular distance
    proj_on_axis = along_axis[..., None] * direction  # (I, J, K, 3)
    perp = diff - proj_on_axis
    perp_dist = np.linalg.norm(perp, axis=-1)

    # Inside cylinder
    inside = (np.abs(along_axis) <= half_length) & (perp_dist <= radius)
    mask[inside] = 1

    return mask


class ContactSegmenter:
    """Creates a Slicer segmentation with per-contact cylindrical segments."""

    def create_segmentation(self, electrodes: list[Electrode], reference_volume_node):
        """Create a segmentation node with one segment per contact.

        Args:
            electrodes: List of Electrode objects with contacts.
            reference_volume_node: Volume node for geometry reference.

        Returns:
            vtkMRMLSegmentationNode.

        Example (in Slicer Python console)::

            segmenter = ContactSegmenter()
            seg_node = segmenter.create_segmentation(electrodes, ct_volume_node)
        """
        import slicer  # noqa: F401

        seg_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode", "ElectrodeContacts"
        )
        seg_node.CreateDefaultDisplayNodes()
        seg_node.SetReferenceImageGeometryParameterFromVolumeNode(reference_volume_node)

        segmentation = seg_node.GetSegmentation()

        for electrode_idx, electrode in enumerate(electrodes):
            color = ELECTRODE_COLORS[electrode_idx % len(ELECTRODE_COLORS)]

            for contact in electrode.contacts:
                segment_id = segmentation.AddEmptySegment(contact.label, contact.label)
                segment = segmentation.GetSegment(segment_id)
                segment.SetColor(*color)

        # Generate closed surface for 3D visualization
        seg_node.CreateClosedSurfaceRepresentation()

        return seg_node
