# tests/test_metal_segmenter.py
import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow")
)

import numpy as np
from SEEGFellowLib.metal_segmenter import threshold_volume
from SEEGFellowLib.metal_segmenter import detect_contact_centers


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


class TestDetectContactCenters:
    def test_finds_isolated_blobs(self):
        """Three bright blobs at known positions should produce three centers."""
        ct = np.zeros((60, 60, 60), dtype=np.float32)
        metal_mask = np.zeros((60, 60, 60), dtype=np.uint8)
        # Place 3 bright spots (simulating contacts) at known IJK positions
        centers_ijk = [(15, 30, 30), (25, 30, 30), (35, 30, 30)]
        for ci, cj, ck in centers_ijk:
            # Gaussian-like blob: bright center, fading neighbours
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    for dk in range(-2, 3):
                        dist = (di**2 + dj**2 + dk**2) ** 0.5
                        ct[ci + di, cj + dj, ck + dk] = 3000.0 * max(0, 1 - dist / 3)
                        if dist <= 2:
                            metal_mask[ci + di, cj + dj, ck + dk] = 1

        detected = detect_contact_centers(ct, metal_mask, sigma=1.2)
        assert detected.shape[1] == 3
        assert len(detected) == 3
        # Each detected center should be within 2 voxels of a true center
        for true_center in centers_ijk:
            dists = np.linalg.norm(detected - np.array(true_center), axis=1)
            assert dists.min() < 2.0, f"No detection near {true_center}"

    def test_rejects_blobs_outside_metal_mask(self):
        """A bright spot outside the metal mask should not be detected."""
        ct = np.zeros((40, 40, 40), dtype=np.float32)
        metal_mask = np.zeros((40, 40, 40), dtype=np.uint8)
        # Blob inside metal mask
        ct[10, 20, 20] = 3000.0
        metal_mask[9:12, 19:22, 19:22] = 1
        # Blob outside metal mask (e.g. bone)
        ct[30, 20, 20] = 3000.0
        # metal_mask stays 0 at (30,20,20)

        detected = detect_contact_centers(ct, metal_mask, sigma=1.2)
        assert len(detected) == 1
        assert abs(detected[0, 0] - 10) < 2.0

    def test_empty_mask_returns_empty(self):
        """An all-zero metal mask should return no centers."""
        ct = np.full((20, 20, 20), 3000.0, dtype=np.float32)
        metal_mask = np.zeros((20, 20, 20), dtype=np.uint8)
        detected = detect_contact_centers(ct, metal_mask, sigma=1.2)
        assert len(detected) == 0


from unittest.mock import patch, MagicMock
from SEEGFellowLib.metal_segmenter import compute_brain_mask


class TestComputeBrainMask:
    def test_returns_binary_mask_from_deepbet(self, tmp_path):
        """compute_brain_mask should call deepbet and return a binary mask."""
        volume = np.random.rand(10, 20, 30).astype(np.float32)
        affine = np.eye(4)

        # Mock deepbet.run_bet to write a mask file filled with ones
        def fake_run_bet(
            input_paths, brain_paths, mask_paths, tiv_paths, threshold, n_dilate, no_gpu
        ):
            import nibabel as nib

            mask = np.ones((10, 20, 30), dtype=np.uint8)
            nib.save(nib.Nifti1Image(mask, affine), mask_paths[0])

        with patch("SEEGFellowLib.metal_segmenter.run_bet", fake_run_bet):
            result = compute_brain_mask(volume, affine)

        assert result.shape == (10, 20, 30)
        assert result.dtype == np.uint8
        assert np.all(result == 1)

    def test_empty_mask_raises(self, tmp_path):
        """compute_brain_mask should raise if deepbet produces an empty mask."""
        volume = np.random.rand(10, 20, 30).astype(np.float32)
        affine = np.eye(4)

        def fake_run_bet(
            input_paths, brain_paths, mask_paths, tiv_paths, threshold, n_dilate, no_gpu
        ):
            import nibabel as nib

            mask = np.zeros((10, 20, 30), dtype=np.uint8)
            nib.save(nib.Nifti1Image(mask, affine), mask_paths[0])

        with patch("SEEGFellowLib.metal_segmenter.run_bet", fake_run_bet):
            import pytest as pt

            with pt.raises(RuntimeError, match="empty"):
                compute_brain_mask(volume, affine)
