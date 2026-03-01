import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow")
)

import numpy as np
import pytest

from SEEGFellowLib.contact_labeler import SYNTHSEG_LUT, label_contacts


class TestSynthSegLUT:
    def test_contains_hippocampus(self):
        assert SYNTHSEG_LUT[17] == "Left Hippocampus"
        assert SYNTHSEG_LUT[53] == "Right Hippocampus"

    def test_contains_cortical_parcels(self):
        assert SYNTHSEG_LUT[1028] == "Left Superior Frontal"
        assert SYNTHSEG_LUT[2028] == "Right Superior Frontal"

    def test_contains_white_matter(self):
        assert SYNTHSEG_LUT[2] == "Left Cerebral WM"
        assert SYNTHSEG_LUT[41] == "Right Cerebral WM"

    def test_background_not_in_lut(self):
        assert 0 not in SYNTHSEG_LUT


class TestLabelContacts:
    def _make_parcellation(self, shape=(30, 30, 30)):
        """Return a zeroed parcellation array and identity affine."""
        return np.zeros(shape, dtype=np.int32), np.eye(4)

    def test_single_contact_in_hippocampus(self):
        parc, affine = self._make_parcellation()
        # Place hippocampus label at voxel (15, 15, 15)
        # Slicer convention: array indexed as [K, J, I]
        # With identity affine, RAS = (I, J, K) → so RAS (15,15,15) maps to IJK (15,15,15) → KJI (15,15,15)
        parc[15, 15, 15] = 17  # Left Hippocampus
        contacts_ras = np.array([[15.0, 15.0, 15.0]])
        labels = label_contacts(contacts_ras, parc, affine)
        assert labels == ["Left Hippocampus"]

    def test_contact_in_white_matter_with_nearby_cortex(self):
        """WM contact should report 'WM near <nearest cortical label>'."""
        parc, affine = self._make_parcellation()
        parc[15, 15, 15] = 2  # Left Cerebral WM
        parc[15, 15, 17] = 1028  # Left Superior Frontal, 2 voxels away
        contacts_ras = np.array([[15.0, 15.0, 15.0]])
        labels = label_contacts(contacts_ras, parc, affine)
        assert labels == ["WM near Left Superior Frontal"]

    def test_contact_in_white_matter_no_nearby_cortex(self):
        """WM contact with no cortex nearby should report just 'Left Cerebral WM'."""
        parc, affine = self._make_parcellation()
        parc[15, 15, 15] = 2  # Left Cerebral WM
        contacts_ras = np.array([[15.0, 15.0, 15.0]])
        labels = label_contacts(contacts_ras, parc, affine, search_radius_mm=1.0)
        assert labels == ["Left Cerebral WM"]

    def test_contact_outside_brain(self):
        parc, affine = self._make_parcellation()
        contacts_ras = np.array([[15.0, 15.0, 15.0]])
        labels = label_contacts(contacts_ras, parc, affine)
        assert labels == ["Outside brain"]

    def test_contact_outside_volume_bounds(self):
        parc, affine = self._make_parcellation()
        contacts_ras = np.array([[100.0, 100.0, 100.0]])
        labels = label_contacts(contacts_ras, parc, affine)
        assert labels == ["Outside brain"]

    def test_multiple_contacts(self):
        parc, affine = self._make_parcellation()
        parc[10, 10, 10] = 17  # Left Hippocampus
        parc[20, 20, 20] = 53  # Right Hippocampus
        contacts_ras = np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
        labels = label_contacts(contacts_ras, parc, affine)
        assert labels == ["Left Hippocampus", "Right Hippocampus"]

    def test_non_identity_affine(self):
        """With 2mm voxels, RAS (20,20,20) maps to voxel (10,10,10)."""
        parc, _ = self._make_parcellation()
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        parc[10, 10, 10] = 18  # Left Amygdala
        contacts_ras = np.array([[20.0, 20.0, 20.0]])
        labels = label_contacts(contacts_ras, parc, affine)
        assert labels == ["Left Amygdala"]
