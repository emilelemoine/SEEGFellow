# tests/test_electrode_detector.py
import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow")
)

import numpy as np
from SEEGFellowLib.electrode_detector import (
    extract_metal_coords,
    cluster_into_electrodes,
)


class TestExtractMetalCoords:
    def test_extracts_nonzero_coords(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2, 3, 4] = 1
        mask[5, 6, 7] = 1
        # Dummy spacing and origin
        coords = extract_metal_coords(
            mask, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)
        )
        assert coords.shape == (2, 3)

    def test_applies_spacing_and_origin(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[0, 0, 0] = 1
        coords = extract_metal_coords(
            mask, spacing=(2.0, 3.0, 4.0), origin=(10.0, 20.0, 30.0)
        )
        np.testing.assert_allclose(coords[0], [10.0, 20.0, 30.0])


class TestClusterIntoElectrodes:
    def _make_line_cluster(self, start, direction, n_contacts, contact_voxels=3):
        """Create a synthetic electrode: a line of small clusters."""
        direction = np.array(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        points = []
        for i in range(n_contacts):
            center = np.array(start) + i * 3.5 * direction  # 3.5mm spacing
            # Small blob around each contact
            for _ in range(contact_voxels):
                jitter = np.random.randn(3) * 0.3
                points.append(center + jitter)
        return np.array(points)

    def test_two_separate_electrodes(self):
        np.random.seed(42)
        e1 = self._make_line_cluster([0, 0, 0], [1, 0, 0], 8)
        e2 = self._make_line_cluster([0, 50, 0], [0, 1, 0], 6)
        all_coords = np.vstack([e1, e2])
        clusters = cluster_into_electrodes(all_coords)
        assert len(clusters) == 2

    def test_single_electrode(self):
        np.random.seed(42)
        e1 = self._make_line_cluster([0, 0, 0], [1, 0, 0], 10)
        clusters = cluster_into_electrodes(e1)
        assert len(clusters) == 1
