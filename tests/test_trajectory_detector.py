# tests/test_trajectory_detector.py
import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow")
)

import numpy as np
from SEEGFellowLib.trajectory_detector import (
    estimate_trajectory,
    detect_contacts_from_intensity_profile,
)
from SEEGFellowLib.electrode_model import ElectrodeParams


class TestEstimateTrajectory:
    def test_pca_direction(self):
        """PCA on a line of points along x should give x-direction."""
        np.random.seed(42)
        n = 50
        points = np.column_stack(
            [
                np.linspace(0, 20, n),
                np.random.randn(n) * 0.3,
                np.random.randn(n) * 0.3,
            ]
        )
        direction = estimate_trajectory(points)
        assert abs(abs(direction[0]) - 1.0) < 0.1

    def test_direction_hint(self):
        """When given a direction hint, should use it."""
        np.random.seed(42)
        points = np.random.randn(20, 3) * 5  # random blob
        hint = np.array([0.0, 1.0, 0.0])
        direction = estimate_trajectory(points, direction_hint=hint)
        # Should be close to hint
        assert abs(np.dot(direction, hint)) > 0.9


class TestDetectContactsFromProfile:
    def _make_profile(self, n_contacts, spacing, contact_width=1.5, noise=50.0):
        """Create a synthetic intensity profile along an electrode."""
        length = (n_contacts + 1) * spacing
        positions = np.arange(0, length, 0.2)
        intensities = np.full_like(positions, 200.0)  # background
        # Add peaks at contact positions
        for i in range(n_contacts):
            center = (i + 0.5) * spacing
            mask = np.abs(positions - center) < contact_width / 2
            intensities[mask] = 3000.0
        intensities += np.random.randn(len(intensities)) * noise
        return positions, intensities

    def test_detects_correct_number(self):
        np.random.seed(42)
        positions, intensities = self._make_profile(8, spacing=3.5)
        contacts = detect_contacts_from_intensity_profile(
            positions, intensities, num_contacts=8, expected_spacing=3.5
        )
        assert len(contacts) == 8

    def test_contacts_are_sorted(self):
        np.random.seed(42)
        positions, intensities = self._make_profile(6, spacing=3.5)
        contacts = detect_contacts_from_intensity_profile(
            positions, intensities, num_contacts=6, expected_spacing=3.5
        )
        for i in range(len(contacts) - 1):
            assert contacts[i] < contacts[i + 1]
