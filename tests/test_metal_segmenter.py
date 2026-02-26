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


from SEEGFellowLib.metal_segmenter import compute_head_mask


class TestComputeHeadMask:
    def test_excludes_air_background(self):
        """Voxels at -1000 HU (air) should be excluded from the head mask."""
        vol = np.full((20, 20, 20), -1000.0, dtype=np.float32)
        # A blob of soft tissue in the centre
        vol[8:12, 8:12, 8:12] = 40.0
        mask = compute_head_mask(vol)
        assert mask[10, 10, 10] == 1
        assert mask[0, 0, 0] == 0

    def test_fills_internal_air(self):
        """Air pockets inside the head (sinuses) should be filled."""
        vol = np.full((30, 30, 30), -1000.0, dtype=np.float32)
        # Head shell
        vol[5:25, 5:25, 5:25] = 40.0
        # Internal air pocket
        vol[12:18, 12:18, 12:18] = -900.0
        mask = compute_head_mask(vol)
        # Internal air should be inside the head mask
        assert mask[15, 15, 15] == 1


class TestCleanupMetalMaskImproved:
    def test_head_mask_excludes_external_metal(self):
        """Metal outside the head mask should be removed."""
        ct = np.full((40, 40, 40), -1000.0, dtype=np.float32)
        # Patient head in centre
        ct[10:30, 10:30, 10:30] = 40.0
        # Electrode inside head
        for i in range(12, 28):
            ct[i, 20, 20] = 3000.0
        # Metal object outside head (e.g. headframe)
        for i in range(0, 8):
            ct[i, 5, 5] = 3000.0

        from SEEGFellowLib.metal_segmenter import compute_head_mask

        head_mask = compute_head_mask(ct)
        metal = threshold_volume(ct, threshold=2500)
        cleaned = cleanup_metal_mask(metal, head_mask=head_mask)

        # Electrode inside head survives
        assert np.any(cleaned[12:28, 20, 20] == 1)
        # External metal removed
        assert np.sum(cleaned[0:8, 5, 5]) == 0

    def test_medium_bulky_component_rejected(self):
        """A medium-sized bulky component (30-500 voxels) should be rejected."""
        mask = np.zeros((50, 50, 50), dtype=np.uint8)
        # Medium bulky blob (~5x5x5 = 125 voxels, elongation ~1)
        mask[10:15, 10:15, 10:15] = 1
        # Elongated electrode (thin line)
        for i in range(20, 45):
            mask[i, 25, 25] = 1
        cleaned = cleanup_metal_mask(mask)
        # Bulky blob removed
        assert np.sum(cleaned[10:15, 10:15, 10:15]) == 0
        # Electrode kept
        assert np.any(cleaned[20:45, 25, 25] == 1)

    def test_small_fragment_kept(self):
        """Small fragments (< 30 voxels) survive — they may be electrode sub-clusters."""
        mask = np.zeros((30, 30, 30), dtype=np.uint8)
        # 10-voxel fragment (e.g. 2 contacts of a gapped electrode)
        for i in range(10, 15):
            mask[i, 15, 15] = 1
            mask[i, 16, 15] = 1
        cleaned = cleanup_metal_mask(mask)
        assert np.any(cleaned[10:15, 15:17, 15] == 1)


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
