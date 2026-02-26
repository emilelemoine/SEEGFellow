# tests/test_metal_segmenter.py
import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow")
)

import numpy as np
from SEEGFellowLib.metal_segmenter import threshold_volume, cleanup_metal_mask


class TestThresholdVolume:
    def test_basic_threshold(self):
        vol = np.zeros((20, 20, 20), dtype=np.float32)
        vol[10, 10, 10] = 3000.0  # metal voxel
        vol[5, 5, 5] = 1000.0  # bone voxel (below threshold)
        mask = threshold_volume(vol, threshold=2500)
        assert mask[10, 10, 10] == 1
        assert mask[5, 5, 5] == 0

    def test_threshold_configurable(self):
        vol = np.zeros((10, 10, 10), dtype=np.float32)
        vol[5, 5, 5] = 2000.0
        assert threshold_volume(vol, threshold=1500)[5, 5, 5] == 1
        assert threshold_volume(vol, threshold=2500)[5, 5, 5] == 0


class TestCleanupMetalMask:
    def test_removes_small_noise(self):
        """Single isolated voxel should be removed by morphological opening."""
        mask = np.zeros((30, 30, 30), dtype=np.uint8)
        mask[15, 15, 15] = 1  # single noise voxel
        # Add a real cluster (line of contacts)
        for i in range(10, 20):
            mask[i, 15, 15] = 1
            mask[i, 16, 15] = 1
        cleaned = cleanup_metal_mask(mask)
        # The real cluster should survive
        assert cleaned[14, 15, 15] == 1 or cleaned[15, 15, 15] == 1
        # The isolated noise voxel far from the cluster should be gone
        # (it's actually within the cluster here, so let's use a truly isolated one)
        mask2 = np.zeros((30, 30, 30), dtype=np.uint8)
        mask2[2, 2, 2] = 1  # isolated noise
        for i in range(15, 25):
            mask2[i, 15, 15] = 1
            mask2[i, 16, 15] = 1
        cleaned2 = cleanup_metal_mask(mask2)
        assert cleaned2[2, 2, 2] == 0

    def test_removes_large_bone_blobs(self):
        """Large bulky components (bone) should be removed."""
        mask = np.zeros((50, 50, 50), dtype=np.uint8)
        # Electrode-like: thin elongated cluster
        for i in range(10, 30):
            mask[i, 25, 25] = 1
            mask[i, 26, 25] = 1
        # Bone-like: large bulky blob
        mask[35:45, 35:45, 35:45] = 1
        cleaned = cleanup_metal_mask(mask)
        # Electrode should survive
        assert np.any(cleaned[10:30, 25:27, 25] == 1)
        # Bone blob should be removed
        assert np.sum(cleaned[35:45, 35:45, 35:45]) == 0
