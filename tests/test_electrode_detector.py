# tests/test_electrode_detector.py
import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow")
)

import numpy as np
from SEEGFellowLib.electrode_detector import (
    cluster_into_electrodes,
)
from SEEGFellowLib.electrode_detector import _filter_contact_mask


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

    def test_spacing_cutoff_factor_controls_acceptance(self):
        """A cluster at 1.5 mm spacing is rejected with default expected_spacing (cutoff = 2.275 mm)
        but accepted when expected_spacing matches actual spacing (cutoff = 0.975 mm)."""
        # 1.5 mm contacts — below the default cutoff (3.5 * 0.65 = 2.275 mm)
        centers = self._make_centers([0, 0, 0], [1, 0, 0], 8, spacing=1.5)
        assert detect_electrodes(centers, spacing_cutoff_factor=0.65) == []
        assert (
            len(
                detect_electrodes(
                    centers, expected_spacing=1.5, spacing_cutoff_factor=0.65
                )
            )
            == 1
        )


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


class TestFilterContactMask:
    def _make_mask(self, shape=(20, 20, 20)):
        return np.zeros(shape, dtype=np.uint8)

    def test_keeps_components_within_range(self):
        mask = self._make_mask()
        # Component of 10 voxels — within [3, 500]
        mask[0:10, 0, 0] = 1
        result = _filter_contact_mask(mask, min_voxels=3, max_voxels=500)
        assert result.sum() == 10

    def test_removes_component_above_max(self):
        mask = self._make_mask()
        # Component of 600 voxels — above max_voxels=500
        mask[0:10, 0:10, 0:6] = 1  # 10*10*6 = 600
        result = _filter_contact_mask(mask, min_voxels=3, max_voxels=500)
        assert result.sum() == 0

    def test_removes_component_below_min(self):
        mask = self._make_mask()
        mask[0, 0, 0] = 1  # 1 voxel — below min=3
        result = _filter_contact_mask(mask, min_voxels=3, max_voxels=500)
        assert result.sum() == 0

    def test_max_voxels_parameter_is_respected(self):
        mask = self._make_mask()
        mask[0:10, 0, 0] = 1  # 10 voxels
        # With max_voxels=5 it should be excluded
        assert _filter_contact_mask(mask, min_voxels=3, max_voxels=5).sum() == 0
        # With max_voxels=15 it should be included
        assert _filter_contact_mask(mask, min_voxels=3, max_voxels=15).sum() == 10
