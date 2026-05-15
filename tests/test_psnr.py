"""Tests for PSNR metric — Phase 7."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sparse_edit.metrics.psnr import compute_psnr, compute_psnr_batch


class TestComputePsnr:
    def test_identical_images_returns_inf(self) -> None:
        img = np.random.randint(0, 256, (64, 64, 3)).astype(np.float64)
        assert compute_psnr(img, img) == float("inf")

    def test_known_value(self) -> None:
        img1 = np.zeros((32, 32), dtype=np.float64)
        img2 = np.ones((32, 32), dtype=np.float64)
        psnr = compute_psnr(img1, img2, data_range=1.0)
        assert abs(psnr - 0.0) < 1e-10

    def test_known_value_255(self) -> None:
        img1 = np.zeros((1, 1, 1), dtype=np.float64)
        img2 = np.full((1, 1, 1), 255.0, dtype=np.float64)
        psnr = compute_psnr(img1, img2, data_range=255.0)
        assert abs(psnr - 0.0) < 1e-10

    def test_higher_psnr_for_closer_images(self) -> None:
        rng = np.random.default_rng(42)
        base = rng.random((64, 64, 3)) * 255.0
        noise_small = rng.random((64, 64, 3)) * 1.0
        noise_large = rng.random((64, 64, 3)) * 50.0
        assert compute_psnr(base, base + noise_small) > compute_psnr(base, base + noise_large)

    def test_greyscale_input(self) -> None:
        img1 = np.zeros((32, 32), dtype=np.float64)
        img2 = np.ones((32, 32), dtype=np.float64) * 10.0
        psnr = compute_psnr(img1, img2, data_range=255.0)
        expected = 10.0 * math.log10(255.0**2 / 100.0)
        assert abs(psnr - expected) < 1e-8

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_psnr(np.zeros((32, 32)), np.zeros((32, 64)))

    def test_1d_input_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            compute_psnr(np.zeros(10), np.zeros(10))

    def test_symmetry(self) -> None:
        rng = np.random.default_rng(123)
        a = rng.random((64, 64, 3)) * 255.0
        b = rng.random((64, 64, 3)) * 255.0
        assert abs(compute_psnr(a, b) - compute_psnr(b, a)) < 1e-10


class TestComputePsnrBatch:
    def test_batch_matches_single(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.random((4, 32, 32, 3)) * 255.0
        b = rng.random((4, 32, 32, 3)) * 255.0
        batch_psnr = compute_psnr_batch(a, b, data_range=255.0)
        for i in range(4):
            assert abs(batch_psnr[i] - compute_psnr(a[i], b[i])) < 1e-8

    def test_identical_batch(self) -> None:
        a = np.random.default_rng(0).random((3, 16, 16, 3)) * 255.0
        assert all(np.isinf(compute_psnr_batch(a, a)))

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_psnr_batch(np.zeros((2, 16, 16, 3)), np.zeros((3, 16, 16, 3)))
