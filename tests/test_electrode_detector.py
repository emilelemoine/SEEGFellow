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
    def _make_centers(self, start, direction, n_contacts, spacing=3.5):
        """Create synthetic contact centers: one point per contact."""
        direction = np.array(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        return np.array(
            [np.array(start) + i * spacing * direction for i in range(n_contacts)]
        )

    def test_two_separate_electrodes(self):
        e1 = self._make_centers([0, 0, 0], [1, 0, 0], 8)
        e2 = self._make_centers([0, 50, 0], [0, 1, 0], 6)
        all_centers = np.vstack([e1, e2])
        clusters = cluster_into_electrodes(all_centers, distance_threshold=7.0)
        assert len(clusters) == 2

    def test_single_electrode(self):
        e1 = self._make_centers([0, 0, 0], [1, 0, 0], 10)
        clusters = cluster_into_electrodes(e1, distance_threshold=7.0)
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


from SEEGFellowLib.electrode_detector import (
    merge_collinear_clusters,
    analyze_spacing,
    detect_electrodes,
)
from SEEGFellowLib.electrode_model import Electrode


class TestMergeCollinearClusters:
    def test_merges_two_collinear_fragments(self):
        """Two fragments along the same line should merge."""
        np.random.seed(42)
        frag1 = np.column_stack(
            [
                np.linspace(0, 10, 20),
                np.random.randn(20) * 0.3,
                np.random.randn(20) * 0.3,
            ]
        )
        frag2 = np.column_stack(
            [
                np.linspace(20, 30, 20),  # same axis, gap in between
                np.random.randn(20) * 0.3,
                np.random.randn(20) * 0.3,
            ]
        )
        clusters = [frag1, frag2]
        merged = merge_collinear_clusters(clusters, angle_tolerance=15.0)
        assert len(merged) == 1

    def test_does_not_merge_perpendicular(self):
        np.random.seed(42)
        frag1 = np.column_stack(
            [
                np.linspace(0, 20, 30),
                np.zeros(30),
                np.zeros(30),
            ]
        )
        frag2 = np.column_stack(
            [
                np.zeros(30),
                np.linspace(40, 60, 30),  # perpendicular, far away
                np.zeros(30),
            ]
        )
        clusters = [frag1, frag2]
        merged = merge_collinear_clusters(clusters, angle_tolerance=15.0)
        assert len(merged) == 2


class TestAnalyzeSpacing:
    def test_uniform_spacing(self):
        positions = np.array([0.0, 3.5, 7.0, 10.5, 14.0])
        spacing_info = analyze_spacing(positions)
        assert not spacing_info["has_gaps"]
        assert abs(spacing_info["contact_spacing"] - 3.5) < 0.5

    def test_gapped_spacing(self):
        # 4 contacts, gap, 4 contacts
        group1 = np.array([0.0, 3.5, 7.0, 10.5])
        group2 = group1 + 24.5  # gap of 14mm
        positions = np.concatenate([group1, group2])
        spacing_info = analyze_spacing(positions, gap_ratio_threshold=1.8)
        assert spacing_info["has_gaps"]


class TestDetectElectrodes:
    def _make_centers(self, start, direction, n_contacts, spacing=3.5):
        """Create synthetic contact centers: one point per contact."""
        direction = np.array(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        return np.array(
            [np.array(start) + i * spacing * direction for i in range(n_contacts)]
        )

    def test_detects_two_electrodes(self):
        np.random.seed(42)
        e1 = self._make_centers([0, 0, 0], [1, 0, 0], 8)
        e2 = self._make_centers([0, 50, 0], [0, 1, 0], 6)
        all_centers = np.vstack([e1, e2])

        electrodes = detect_electrodes(all_centers)
        assert len(electrodes) == 2
        contact_counts = sorted([e.num_contacts for e in electrodes])
        assert contact_counts == [6, 8]

    def test_rejects_too_few_contacts(self):
        e1 = self._make_centers([0, 0, 0], [1, 0, 0], 2)
        electrodes = detect_electrodes(e1, min_contacts=3)
        assert len(electrodes) == 0


class TestDetectElectrodesWithNoise:
    def _make_centers(self, start, direction, n_contacts, spacing=3.5):
        direction = np.array(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        return np.array(
            [np.array(start) + i * spacing * direction for i in range(n_contacts)]
        )

    def test_ignores_scattered_noise(self):
        """Random scattered points should not produce an electrode."""
        np.random.seed(0)
        e1 = self._make_centers([0, 0, 0], [1, 0, 0], 8)
        noise = np.random.uniform(-5, 5, (20, 3)) + np.array([0, 50, 0])
        all_centers = np.vstack([e1, noise])
        electrodes = detect_electrodes(all_centers)
        assert len(electrodes) == 1
        assert electrodes[0].num_contacts == 8
