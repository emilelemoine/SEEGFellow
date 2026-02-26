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


def cleanup_metal_mask(
    mask: np.ndarray,
    min_component_size: int = 5,
    max_component_volume: int = 500,
    max_elongation_ratio: float = 2.0,
) -> np.ndarray:
    """Remove noise and bone fragments from a binary metal mask.

    Steps:
    1. Morphological opening to remove small noise
    2. Connected component analysis
    3. Remove components smaller than min_component_size
    4. Remove large bulky components (bone) based on volume and shape

    Args:
        mask: Binary mask (0/1 uint8).
        min_component_size: Minimum voxel count to keep a component.
        max_component_volume: Components larger than this are candidate bone.
        max_elongation_ratio: Bone has low elongation (bounding box is roughly
            cubic). Components with (longest axis / shortest axis) below this
            ratio AND volume above max_component_volume are removed.

    Example::

        cleaned = cleanup_metal_mask(binary_mask, min_component_size=5)
    """
    struct = ndimage.generate_binary_structure(3, 1)

    # Connected component analysis (size filtering replaces morphological opening
    # so that thin elongated electrode clusters are not eroded away)
    labeled, num_features = ndimage.label(mask, structure=struct)

    result = np.zeros_like(mask)
    for comp_id in range(1, num_features + 1):
        component = labeled == comp_id
        volume = np.sum(component)

        # Remove small noise
        if volume < min_component_size:
            continue

        # Check if large component is bone-like (bulky, not elongated)
        if volume > max_component_volume:
            coords = np.argwhere(component)
            bbox_size = coords.max(axis=0) - coords.min(axis=0) + 1
            sorted_dims = np.sort(bbox_size)
            # Elongation: longest / shortest dimension
            if sorted_dims[0] > 0:
                elongation = sorted_dims[-1] / sorted_dims[0]
            else:
                elongation = 1.0
            # Bone is bulky (low elongation), electrodes are elongated
            if elongation < max_elongation_ratio:
                continue

        result[component] = 1

    return result


class MetalSegmenter:
    """Slicer wrapper: segments metal from a CT volume node.

    Example (in Slicer Python console)::

        segmenter = MetalSegmenter()
        metal_labelmap = segmenter.segment(ct_volume_node, threshold=2500)
    """

    def segment(self, ct_volume_node, threshold: float = 2500):
        """Threshold CT and clean up to isolate metal voxels.

        Args:
            ct_volume_node: vtkMRMLScalarVolumeNode with the CT data.
            threshold: HU threshold for metal.

        Returns:
            vtkMRMLLabelMapVolumeNode containing the metal mask.
        """
        import slicer
        from slicer.util import arrayFromVolume, updateVolumeFromArray

        ct_array = arrayFromVolume(ct_volume_node)
        mask = threshold_volume(ct_array, threshold)
        cleaned = cleanup_metal_mask(mask)

        # Create output labelmap node
        labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", "MetalSegmentation"
        )
        # Copy geometry from CT
        labelmap_node.CopyOrientation(ct_volume_node)

        # Use a volumes logic to copy the volume properties
        volumes_logic = slicer.modules.volumes.logic()
        volumes_logic.CreateLabelVolumeFromVolume(
            slicer.mrmlScene, labelmap_node, ct_volume_node
        )

        updateVolumeFromArray(labelmap_node, cleaned)
        return labelmap_node
