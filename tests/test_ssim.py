"""Tests for SSIM metric — Phase 7."""

from __future__ import annotations

import numpy as np
import pytest

from sparse_edit.metrics.ssim import compute_ssim, compute_ssim_map


class TestComputeSsim:
    def test_identical_images(self) -> None:
        img = np.random.default_rng(42).integers(0, 256, (64, 64, 3)).astype(np.float64)
        assert abs(compute_ssim(img, img, data_range=255.0) - 1.0) < 1e-6

    def test_completely_different(self) -> None:
        a = np.zeros((64, 64, 3), dtype=np.float64)
        b = np.full((64, 64, 3), 255.0, dtype=np.float64)
        assert compute_ssim(a, b, data_range=255.0) < 0.1

    def test_valid_range(self) -> None:
        rng = np.random.default_rng(99)
        a = rng.random((64, 64, 3)) * 255.0
        b = rng.random((64, 64, 3)) * 255.0
        val = compute_ssim(a, b, data_range=255.0)
        assert -1.0 <= val <= 1.0

    def test_greyscale(self) -> None:
        a = np.random.default_rng(7).integers(0, 256, (64, 64)).astype(np.float64)
        assert abs(compute_ssim(a, a, data_range=255.0, channel_axis=None) - 1.0) < 1e-6

    def test_noise_reduces_ssim(self) -> None:
        rng = np.random.default_rng(42)
        base = rng.random((64, 64, 3)) * 255.0
        close = base + rng.normal(0, 1, base.shape)
        far = base + rng.normal(0, 30, base.shape)
        assert compute_ssim(base, close) > compute_ssim(base, far)

    def test_symmetry(self) -> None:
        rng = np.random.default_rng(123)
        a = rng.random((64, 64, 3)) * 255.0
        b = rng.random((64, 64, 3)) * 255.0
        assert abs(compute_ssim(a, b) - compute_ssim(b, a)) < 1e-10

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_ssim(np.zeros((32, 32, 3)), np.zeros((32, 64, 3)))

    def test_data_range_1(self) -> None:
        a = np.random.default_rng(0).random((64, 64, 3))
        assert abs(compute_ssim(a, a, data_range=1.0) - 1.0) < 1e-6


class TestComputeSsimMap:
    def test_map_shape(self) -> None:
        a = np.random.default_rng(0).random((64, 64)) * 255.0
        ssim_map = compute_ssim_map(a, a, data_range=255.0, kernel_size=11)
        assert ssim_map.shape == (54, 54)

    def test_identical_map_all_ones(self) -> None:
        a = np.random.default_rng(7).integers(0, 256, (64, 64)).astype(np.float64)
        assert np.allclose(compute_ssim_map(a, a, data_range=255.0), 1.0, atol=1e-6)

    def test_3d_raises(self) -> None:
        with pytest.raises(ValueError, match="single-channel"):
            compute_ssim_map(np.zeros((64, 64, 3)), np.zeros((64, 64, 3)))
