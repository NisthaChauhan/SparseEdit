"""Tests for LPIPS metric — Phase 8."""

from __future__ import annotations

import numpy as np
import pytest

from sparse_edit.metrics.lpips import (
    compute_lpips_from_features,
    _channel_normalise,
    VGG_STAGE_CHANNELS,
)


def _make_features(rng: np.random.Generator, spatial: int = 8) -> list[np.ndarray]:
    return [rng.random((spatial, spatial, ch)) for ch in VGG_STAGE_CHANNELS]


class TestChannelNormalise:
    def test_unit_norm(self) -> None:
        feat = np.random.default_rng(42).random((16, 16, 64))
        normed = _channel_normalise(feat)
        norms = np.sqrt(np.sum(normed ** 2, axis=-1))
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_zero_stable(self) -> None:
        assert not np.any(np.isnan(_channel_normalise(np.zeros((4, 4, 32)))))

    def test_batch_shape(self) -> None:
        feat = np.random.default_rng(0).random((2, 8, 8, 128))
        normed = _channel_normalise(feat)
        assert normed.shape == feat.shape


class TestComputeLpipsFromFeatures:
    def test_identical_zero(self) -> None:
        feats = _make_features(np.random.default_rng(42))
        assert abs(compute_lpips_from_features(feats, feats)) < 1e-10

    def test_different_positive(self) -> None:
        assert compute_lpips_from_features(
            _make_features(np.random.default_rng(42)),
            _make_features(np.random.default_rng(99)),
        ) > 0.0

    def test_symmetry(self) -> None:
        fx = _make_features(np.random.default_rng(1))
        fy = _make_features(np.random.default_rng(2))
        assert abs(compute_lpips_from_features(fx, fy) - compute_lpips_from_features(fy, fx)) < 1e-10

    def test_weights_scaling(self) -> None:
        fx = _make_features(np.random.default_rng(42))
        fy = _make_features(np.random.default_rng(7))
        ones = [np.ones(ch) for ch in VGG_STAGE_CHANNELS]
        twos = [np.full(ch, 2.0) for ch in VGG_STAGE_CHANNELS]
        d1 = compute_lpips_from_features(fx, fy, linear_weights=ones)
        d2 = compute_lpips_from_features(fx, fy, linear_weights=twos)
        assert abs(d2 - 2.0 * d1) < 1e-8

    def test_wrong_stage_count(self) -> None:
        feats = _make_features(np.random.default_rng(0))
        with pytest.raises(ValueError, match="5 feature stages"):
            compute_lpips_from_features(feats[:3], feats)

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            compute_lpips_from_features(
                _make_features(np.random.default_rng(0), spatial=8),
                _make_features(np.random.default_rng(0), spatial=16),
            )

    def test_ordering(self) -> None:
        base = _make_features(np.random.default_rng(42))
        similar = [f + np.random.default_rng(1).random(f.shape) * 0.01 for f in base]
        different = _make_features(np.random.default_rng(999))
        assert compute_lpips_from_features(base, similar) < compute_lpips_from_features(base, different)
