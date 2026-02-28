# tests/test_ransac_grouping.py
import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow")
)

import numpy as np
import pytest


def _make_electrode(start, direction, n_contacts, spacing=3.5):
    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    return np.array(
        [np.array(start) + i * spacing * direction for i in range(n_contacts)]
    )


class TestRansacGroupContacts:
    def test_two_parallel_electrodes(self):
        """Two well-separated parallel electrodes are found."""
        from SEEGFellowLib.electrode_detector import ransac_group_contacts

        e1 = _make_electrode([0, 0, 0], [1, 0, 0], 8)
        e2 = _make_electrode([0, 50, 0], [1, 0, 0], 6)
        pool = np.vstack([e1, e2])

        groups = ransac_group_contacts(pool, expected_spacing=3.5)
        assert len(groups) == 2
        sizes = sorted([len(g) for g in groups])
        assert sizes == [6, 8]

    def test_crossing_electrodes(self):
        """Two electrodes that cross in 3D are correctly separated."""
        from SEEGFellowLib.electrode_detector import ransac_group_contacts

        # Electrode along X, centered at origin
        e1 = _make_electrode([-14, 0, 0], [1, 0, 0], 8)
        # Electrode along Y, crossing near origin
        e2 = _make_electrode([0, -10.5, 0], [0, 1, 0], 6)
        pool = np.vstack([e1, e2])

        groups = ransac_group_contacts(
            pool, expected_spacing=3.5, distance_tolerance=2.0
        )
        assert len(groups) == 2
        sizes = sorted([len(g) for g in groups])
        assert sizes == [6, 8]

    def test_gapped_electrode(self):
        """An electrode with a gap is found as one group."""
        from SEEGFellowLib.electrode_detector import ransac_group_contacts

        # 6 contacts, 24mm gap, 6 contacts — all along X
        g1 = _make_electrode([0, 0, 0], [1, 0, 0], 6)
        g2 = _make_electrode([41.5, 0, 0], [1, 0, 0], 6)  # gap of 24mm
        pool = np.vstack([g1, g2])

        groups = ransac_group_contacts(pool, expected_spacing=3.5)
        assert len(groups) == 1
        assert len(groups[0]) == 12

    def test_rejects_noise(self):
        """Scattered noise points are not grouped into an electrode."""
        from SEEGFellowLib.electrode_detector import ransac_group_contacts

        np.random.seed(42)
        noise = np.random.uniform(-50, 50, (20, 3))
        groups = ransac_group_contacts(noise, expected_spacing=3.5, min_contacts=3)
        # Noise should produce no valid electrodes (or very few)
        total_assigned = sum(len(g) for g in groups)
        assert total_assigned < 5

    def test_too_few_contacts_returns_empty(self):
        """Fewer contacts than min_contacts yields no groups."""
        from SEEGFellowLib.electrode_detector import ransac_group_contacts

        e1 = _make_electrode([0, 0, 0], [1, 0, 0], 2)
        groups = ransac_group_contacts(e1, min_contacts=3)
        assert groups == []

    def test_preserves_original_coordinates(self):
        """Returned coordinates must be the exact LoG positions, not reprojected."""
        from SEEGFellowLib.electrode_detector import ransac_group_contacts

        np.random.seed(42)
        e1 = _make_electrode([0, 0, 0], [1, 0, 0], 8)
        # Add small perpendicular noise
        e1 += np.random.randn(*e1.shape) * 0.3
        original = e1.copy()
        pool = e1

        groups = ransac_group_contacts(pool, expected_spacing=3.5)
        assert len(groups) == 1
        # Each returned point must be bitwise identical to an input point
        returned = groups[0]
        for pt in returned:
            assert any(np.array_equal(pt, orig) for orig in original)

    def test_deterministic_with_seed(self):
        """Same seed produces same result."""
        from SEEGFellowLib.electrode_detector import ransac_group_contacts

        e1 = _make_electrode([0, 0, 0], [1, 0, 0], 8)
        e2 = _make_electrode([0, 50, 0], [0, 1, 0], 6)
        pool = np.vstack([e1, e2])

        g1 = ransac_group_contacts(pool, expected_spacing=3.5, random_seed=42)
        g2 = ransac_group_contacts(pool, expected_spacing=3.5, random_seed=42)
        assert len(g1) == len(g2)
        for a, b in zip(
            sorted(g1, key=lambda x: len(x)),
            sorted(g2, key=lambda x: len(x)),
        ):
            np.testing.assert_array_equal(a, b)
