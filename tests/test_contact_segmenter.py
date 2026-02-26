# tests/test_contact_segmenter.py
import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow")
)

import numpy as np
from SEEGFellowLib.contact_segmenter import generate_cylinder_mask


class TestGenerateCylinderMask:
    def test_creates_nonzero_mask(self):
        center = np.array([15.0, 15.0, 15.0])
        direction = np.array([1.0, 0.0, 0.0])
        mask = generate_cylinder_mask(
            shape=(30, 30, 30),
            center=center,
            direction=direction,
            length=2.0,
            diameter=0.8,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )
        assert np.any(mask > 0)

    def test_cylinder_volume_reasonable(self):
        """Volume of filled voxels should approximate pi*r^2*L."""
        center = np.array([25.0, 25.0, 25.0])
        direction = np.array([0.0, 0.0, 1.0])
        length = 4.0
        diameter = 2.0
        spacing = (0.5, 0.5, 0.5)
        mask = generate_cylinder_mask(
            shape=(100, 100, 100),
            center=center,
            direction=direction,
            length=length,
            diameter=diameter,
            spacing=spacing,
            origin=(0.0, 0.0, 0.0),
        )
        voxel_vol = 0.5**3
        filled_vol = np.sum(mask) * voxel_vol
        expected_vol = np.pi * (diameter / 2) ** 2 * length
        # Should be within 50% (discretization error)
        assert abs(filled_vol - expected_vol) / expected_vol < 0.5
