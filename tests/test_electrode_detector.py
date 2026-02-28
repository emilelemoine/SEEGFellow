# tests/test_electrode_detector.py
import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow")
)

import numpy as np
from SEEGFellowLib.electrode_detector import _filter_contact_mask
from SEEGFellowLib.electrode_detector import orient_deepest_first

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
    analyze_spacing,
    detect_electrodes,
)
from SEEGFellowLib.electrode_model import Electrode


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

    def test_crossing_electrodes_separated(self):
        """Two crossing electrodes must not be merged."""
        e1 = self._make_centers([-14, 0, 0], [1, 0, 0], 8)
        e2 = self._make_centers([0, -10.5, 0], [0, 1, 0], 6)
        all_centers = np.vstack([e1, e2])
        electrodes = detect_electrodes(all_centers)
        assert len(electrodes) == 2
        contact_counts = sorted([e.num_contacts for e in electrodes])
        assert contact_counts == [6, 8]

    def test_preserves_log_positions(self):
        """Contact RAS positions must be original LoG values, not reprojected."""
        np.random.seed(42)
        e1 = self._make_centers([0, 0, 0], [1, 0, 0], 8)
        e1 += np.random.randn(*e1.shape) * 0.3  # add noise
        original = set(map(tuple, e1))
        electrodes = detect_electrodes(e1)
        assert len(electrodes) == 1
        for c in electrodes[0].contacts:
            assert tuple(c.position_ras) in original


class TestComposeIjkToWorld:
    """_compose_ijk_to_world must include the parent transform when present."""

    def test_no_parent_returns_intrinsic(self):
        from SEEGFellowLib.electrode_detector import ElectrodeDetector

        intrinsic = np.eye(4)
        intrinsic[0, 3] = 10.0
        result = ElectrodeDetector._compose_ijk_to_world(intrinsic, None)
        np.testing.assert_array_equal(result, intrinsic)

    def test_with_parent_composes_parent_times_intrinsic(self):
        """world = parent @ intrinsic — contacts shift by parent offset."""
        from SEEGFellowLib.electrode_detector import ElectrodeDetector

        intrinsic = np.eye(4)
        intrinsic[0, 3] = 10.0  # 10 mm x-offset in CT space
        parent = np.eye(4)
        parent[1, 3] = 20.0  # 20 mm y-shift (registration transform)
        expected = parent @ intrinsic

        result = ElectrodeDetector._compose_ijk_to_world(intrinsic, parent)
        np.testing.assert_array_almost_equal(result, expected)


class TestDetectAll:
    """detect_all() must use the provided metal_mask and not recompute it."""

    def test_uses_provided_metal_mask(self):
        """detect_all() should call detect_contact_centers with the given mask
        and must NOT call compute_head_mask or threshold_volume internally."""
        from unittest.mock import MagicMock, patch

        from SEEGFellowLib.electrode_detector import ElectrodeDetector

        ct_array = np.zeros((20, 20, 20), dtype=np.float32)
        metal_mask = np.zeros((20, 20, 20), dtype=np.uint8)

        mock_ct_node = MagicMock()
        mock_ct_node.GetSpacing.return_value = (1.0, 1.0, 1.0)
        fake_centers = np.empty((0, 3))

        # arrayFromVolume is imported lazily from slicer.util inside detect_all(),
        # so inject it via sys.modules before the import executes.
        slicer_util_mock = MagicMock()
        slicer_util_mock.arrayFromVolume.return_value = ct_array

        with (
            patch.dict(
                "sys.modules",
                {"slicer": MagicMock(), "slicer.util": slicer_util_mock},
            ),
            patch(
                "SEEGFellowLib.metal_segmenter.detect_contact_centers",
                return_value=fake_centers,
            ) as mock_detect,
            patch("SEEGFellowLib.metal_segmenter.compute_head_mask") as mock_head,
            patch("SEEGFellowLib.metal_segmenter.threshold_volume") as mock_thresh,
        ):
            detector = ElectrodeDetector()
            result = detector.detect_all(mock_ct_node, metal_mask=metal_mask)

        assert result == []
        mock_detect.assert_called_once()
        # The metal_mask passed to detect_contact_centers must be the one we provided
        call_args = mock_detect.call_args
        np.testing.assert_array_equal(call_args[0][1], metal_mask)
        # compute_head_mask and threshold_volume must NOT have been called
        mock_head.assert_not_called()
        mock_thresh.assert_not_called()


class TestDetectElectrodesWithNoise:
    def _make_centers(self, start, direction, n_contacts, spacing=3.5):
        direction = np.array(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        return np.array(
            [np.array(start) + i * spacing * direction for i in range(n_contacts)]
        )

    def test_ignores_scattered_noise(self):
        """Random scattered points must not produce an electrode.

        RANSAC can find collinear subsets in any random cloud, so a meaningful
        min_contacts threshold is required.  Using min_contacts=6 (the smallest
        contact count seen in clinical SEEG practice) ensures accidental
        collinear subsets in dense noise are rejected.
        """
        np.random.seed(0)
        e1 = self._make_centers([0, 0, 0], [1, 0, 0], 8)
        noise = np.random.uniform(-5, 5, (20, 3)) + np.array([0, 50, 0])
        all_centers = np.vstack([e1, noise])
        electrodes = detect_electrodes(all_centers, min_contacts=6)
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


class TestOrientDeepestFirst:
    def test_deepest_is_closest_to_brain_centroid(self):
        """Contact nearest to brain centroid should be index 0."""
        # Electrode along X from 10 to 40
        positions = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
        axis_origin = np.array([25.0, 0.0, 0.0])
        axis_direction = np.array([1.0, 0.0, 0.0])
        # Brain centroid at (5, 0, 0) — deepest end is at x=10
        brain_centroid = np.array([5.0, 0.0, 0.0])

        sorted_idx, oriented_dir = orient_deepest_first(
            positions, axis_origin, axis_direction, brain_centroid=brain_centroid
        )
        # First index should correspond to x=10 (closest to centroid)
        first_ras = axis_origin + positions[sorted_idx[0]] * axis_direction
        last_ras = axis_origin + positions[sorted_idx[-1]] * axis_direction
        assert np.linalg.norm(first_ras - brain_centroid) < np.linalg.norm(
            last_ras - brain_centroid
        )

    def test_already_correct_orientation(self):
        """If first contact is already deepest, no reversal needed."""
        positions = np.array([0.0, 3.5, 7.0])
        axis_origin = np.array([50.0, 0.0, 0.0])
        axis_direction = np.array([-1.0, 0.0, 0.0])
        # Brain centroid at (0, 0, 0) — x=50 maps to RAS (50,0,0),
        # positions[0]=0 → RAS = 50 + 0*(-1) = (50,0,0)
        # positions[2]=7.0 → RAS = 50 + 7*(-1) = (43,0,0) — closer to centroid
        brain_centroid = np.array([0.0, 0.0, 0.0])

        sorted_idx, oriented_dir = orient_deepest_first(
            positions, axis_origin, axis_direction, brain_centroid=brain_centroid
        )
        # Should reverse: positions[2] is deepest
        first_ras = axis_origin + positions[sorted_idx[0]] * axis_direction
        assert np.linalg.norm(first_ras - brain_centroid) < np.linalg.norm(
            (axis_origin + positions[sorted_idx[-1]] * axis_direction) - brain_centroid
        )

    def test_fallback_to_origin_when_no_centroid(self):
        """When brain_centroid is None, fall back to (0,0,0)."""
        positions = np.array([0.0, 3.5, 7.0])
        axis_origin = np.array([10.0, 0.0, 0.0])
        axis_direction = np.array([1.0, 0.0, 0.0])

        sorted_idx, _ = orient_deepest_first(
            positions, axis_origin, axis_direction, brain_centroid=None
        )
        # With origin (0,0,0): first contact at RAS (10,0,0), last at (17,0,0)
        # (10,0,0) is closer to origin → already correct, no reversal
        assert sorted_idx[0] == 0
