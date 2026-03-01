import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow")
)

from SEEGFellowLib.hemisphere_labels import (
    LEFT_HEMISPHERE_LABELS as _LEFT_HEMISPHERE_LABELS,
)
from SEEGFellowLib.hemisphere_labels import (
    RIGHT_HEMISPHERE_LABELS as _RIGHT_HEMISPHERE_LABELS,
)


class TestHemisphereLabels:
    def test_no_overlap(self):
        assert _LEFT_HEMISPHERE_LABELS.isdisjoint(_RIGHT_HEMISPHERE_LABELS)

    def test_left_contains_expected_subcortical(self):
        # Left Cerebral Cortex=3, Left WM=2, Left Hippocampus=17, Left Thalamus=10
        assert {2, 3, 10, 17}.issubset(_LEFT_HEMISPHERE_LABELS)

    def test_right_contains_expected_subcortical(self):
        # Right Cerebral Cortex=42, Right WM=41, Right Hippocampus=53, Right Thalamus=49
        assert {41, 42, 49, 53}.issubset(_RIGHT_HEMISPHERE_LABELS)

    def test_left_cortical_parcels(self):
        # A few DKT 1000-series parcels must be present
        assert {1002, 1024, 1035}.issubset(_LEFT_HEMISPHERE_LABELS)

    def test_right_cortical_parcels(self):
        # Corresponding 2000-series
        assert {2002, 2024, 2035}.issubset(_RIGHT_HEMISPHERE_LABELS)

    def test_midline_labels_excluded(self):
        # 3rd Ventricle=14, 4th Ventricle=15, Brain Stem=16, CSF=24
        midline = {14, 15, 16, 24}
        assert not midline.intersection(_LEFT_HEMISPHERE_LABELS)
        assert not midline.intersection(_RIGHT_HEMISPHERE_LABELS)

    def test_reasonable_sizes(self):
        # 14 subcortical + 31 DKT cortical = 45 per side
        assert len(_LEFT_HEMISPHERE_LABELS) == 45
        assert len(_RIGHT_HEMISPHERE_LABELS) == 45
