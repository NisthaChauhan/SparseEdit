"""Tests for leakage metric — Phase 7."""

from __future__ import annotations

import numpy as np
import pytest

from sparse_edit.metrics.leakage import compute_leakage, compute_leakage_map


class TestComputeLeakage:
    def test_no_change_anywhere(self) -> None:
        src = np.ones((64, 64, 3), dtype=np.float64) * 128.0
        mask = np.zeros((64, 64, 1), dtype=np.float64)
        mask[:32, :, :] = 1.0
        result = compute_leakage(src, src, mask)
        assert result["leakage_mse"] == 0.0
        assert result["leakage_psnr"] == float("inf")

    def test_change_only_inside_mask(self) -> None:
        src = np.zeros((64, 64, 3), dtype=np.float64)
        edited = src.copy()
        edited[:32, :, :] = 100.0
        mask = np.zeros((64, 64, 1), dtype=np.float64)
        mask[:32, :, :] = 1.0
        result = compute_leakage(src, edited, mask)
        assert result["leakage_mse"] == 0.0
        assert result["leakage_ratio"] == 0.0

    def test_change_only_outside_mask(self) -> None:
        src = np.zeros((64, 64, 3), dtype=np.float64)
        edited = src.copy()
        edited[32:, :, :] = 50.0
        mask = np.zeros((64, 64, 1), dtype=np.float64)
        mask[:32, :, :] = 1.0
        result = compute_leakage(src, edited, mask)
        assert result["leakage_mse"] > 0.0
        assert result["leakage_ratio"] == float("inf")

    def test_uniform_change(self) -> None:
        src = np.zeros((64, 64, 3), dtype=np.float64)
        edited = np.ones((64, 64, 3), dtype=np.float64) * 10.0
        mask = np.zeros((64, 64, 1), dtype=np.float64)
        mask[:32, :, :] = 1.0
        result = compute_leakage(src, edited, mask)
        assert abs(result["leakage_ratio"] - 1.0) < 1e-10

    def test_preserve_fraction(self) -> None:
        mask = np.zeros((100, 100, 1), dtype=np.float64)
        mask[:25, :, :] = 1.0
        src = np.zeros((100, 100, 3), dtype=np.float64)
        result = compute_leakage(src, src, mask)
        assert abs(result["preserve_fraction"] - 0.75) < 1e-10

    def test_2d_mask(self) -> None:
        src = np.zeros((64, 64, 3), dtype=np.float64)
        mask = np.zeros((64, 64), dtype=np.float64)
        mask[:32, :] = 1.0
        result = compute_leakage(src, src, mask)
        assert result["leakage_mse"] == 0.0

    def test_full_mask_raises(self) -> None:
        src = np.zeros((64, 64, 3), dtype=np.float64)
        mask = np.ones((64, 64, 1), dtype=np.float64)
        with pytest.raises(ValueError, match="no preserve region"):
            compute_leakage(src, src, mask)

    def test_empty_mask_raises(self) -> None:
        src = np.zeros((64, 64, 3), dtype=np.float64)
        mask = np.zeros((64, 64, 1), dtype=np.float64)
        with pytest.raises(ValueError, match="no edit region"):
            compute_leakage(src, src, mask)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_leakage(np.zeros((32, 32, 3)), np.zeros((64, 64, 3)), np.zeros((32, 32, 1)))

    def test_psnr_ordering(self) -> None:
        src = np.zeros((64, 64, 3), dtype=np.float64)
        mask = np.zeros((64, 64, 1), dtype=np.float64)
        mask[:32, :, :] = 1.0
        small_leak = src.copy()
        small_leak[32:, :, :] = 1.0
        big_leak = src.copy()
        big_leak[32:, :, :] = 50.0
        assert compute_leakage(src, small_leak, mask)["leakage_psnr"] > compute_leakage(src, big_leak, mask)["leakage_psnr"]


class TestComputeLeakageMap:
    def test_shape(self) -> None:
        src = np.zeros((64, 64, 3), dtype=np.float64)
        edited = np.ones((64, 64, 3), dtype=np.float64) * 10.0
        mask = np.zeros((64, 64, 1), dtype=np.float64)
        mask[:32, :, :] = 1.0
        assert compute_leakage_map(src, edited, mask).shape == (64, 64)

    def test_zero_inside_mask(self) -> None:
        src = np.zeros((64, 64, 3), dtype=np.float64)
        edited = np.ones((64, 64, 3), dtype=np.float64) * 10.0
        mask = np.zeros((64, 64, 1), dtype=np.float64)
        mask[:32, :, :] = 1.0
        heatmap = compute_leakage_map(src, edited, mask)
        assert np.all(heatmap[:32, :] == 0.0)

    def test_nonzero_outside_mask(self) -> None:
        src = np.zeros((64, 64, 3), dtype=np.float64)
        edited = np.zeros((64, 64, 3), dtype=np.float64)
        edited[32:, :, :] = 10.0
        mask = np.zeros((64, 64, 1), dtype=np.float64)
        mask[:32, :, :] = 1.0
        assert np.all(compute_leakage_map(src, edited, mask)[32:, :] > 0.0)
