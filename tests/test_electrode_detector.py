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


from SEEGFellowLib.electrode_detector import (
    fit_electrode_axis,
    detect_contacts_along_axis,
)


class TestFitElectrodeAxis:
    def test_fits_line_along_x(self):
        np.random.seed(42)
        points = np.column_stack(
            [
                np.linspace(0, 30, 50),
                np.random.randn(50) * 0.3,
                np.random.randn(50) * 0.3,
            ]
        )
        center, direction = fit_electrode_axis(points)
        # Direction should be approximately along x-axis
        assert abs(abs(direction[0]) - 1.0) < 0.1

    def test_returns_unit_vector(self):
        points = np.column_stack(
            [
                np.linspace(0, 10, 20),
                np.zeros(20),
                np.zeros(20),
            ]
        )
        _, direction = fit_electrode_axis(points)
        np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-10)


class TestDetectContactsAlongAxis:
    def _make_contacts_1d(self, n_contacts, spacing, noise=0.2):
        """Create 1D density profile with peaks at regular spacing."""
        positions = []
        for i in range(n_contacts):
            center = i * spacing
            # Several points per contact
            for _ in range(5):
                positions.append(center + np.random.randn() * noise)
        return np.array(positions)

    def test_detects_evenly_spaced_contacts(self):
        np.random.seed(42)
        projections = self._make_contacts_1d(8, spacing=3.5)
        peaks = detect_contacts_along_axis(projections, expected_spacing=3.5)
        assert len(peaks) == 8

    def test_detects_gapped_contacts(self):
        """6 contacts - gap - 6 contacts."""
        np.random.seed(42)
        group1 = self._make_contacts_1d(6, spacing=3.5)
        gap = 6 * 3.5 + 10.0  # gap of 10mm after group 1 ends
        group2 = self._make_contacts_1d(6, spacing=3.5) + gap
        projections = np.concatenate([group1, group2])
        peaks = detect_contacts_along_axis(projections, expected_spacing=3.5)
        assert len(peaks) == 12
