"""Tests for CLIP Score metric — Phase 7."""

from __future__ import annotations

import numpy as np
import pytest

from sparse_edit.metrics.clip_score import (
    compute_clip_score,
    compute_clip_score_batch,
    cosine_similarity,
)


class TestCosineSimilarity:
    def test_identical(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-10

    def test_orthogonal(self) -> None:
        assert abs(cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0]))) < 1e-10

    def test_opposite(self) -> None:
        assert abs(cosine_similarity(np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])) - (-1.0)) < 1e-10

    def test_zero_vector(self) -> None:
        assert cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0

    def test_scale_invariant(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(a, a * 100) - 1.0) < 1e-10


class TestComputeClipScore:
    def test_perfect_alignment(self) -> None:
        emb = np.random.default_rng(42).standard_normal(768)
        assert abs(compute_clip_score(emb, emb) - 100.0) < 1e-8

    def test_orthogonal(self) -> None:
        e1 = np.zeros(768, dtype=np.float64)
        e1[0] = 1.0
        e2 = np.zeros(768, dtype=np.float64)
        e2[1] = 1.0
        assert abs(compute_clip_score(e1, e2)) < 1e-10

    def test_opposite_clamped(self) -> None:
        emb = np.random.default_rng(42).standard_normal(768)
        assert compute_clip_score(emb, -emb) == 0.0

    def test_range(self) -> None:
        rng = np.random.default_rng(99)
        for _ in range(50):
            score = compute_clip_score(rng.standard_normal(768), rng.standard_normal(768))
            assert 0.0 <= score <= 100.0

    def test_ordering(self) -> None:
        rng = np.random.default_rng(42)
        base = rng.standard_normal(768)
        similar = base + rng.standard_normal(768) * 0.01
        different = rng.standard_normal(768)
        assert compute_clip_score(base, similar) > compute_clip_score(base, different)


class TestComputeClipScoreBatch:
    def test_matches_single(self) -> None:
        rng = np.random.default_rng(42)
        img_embs = rng.standard_normal((8, 768))
        txt_embs = rng.standard_normal((8, 768))
        batch = compute_clip_score_batch(img_embs, txt_embs)
        for i in range(8):
            assert abs(batch[i] - compute_clip_score(img_embs[i], txt_embs[i])) < 1e-6

    def test_shape(self) -> None:
        rng = np.random.default_rng(0)
        assert compute_clip_score_batch(rng.standard_normal((5, 768)), rng.standard_normal((5, 768))).shape == (5,)

    def test_all_identical(self) -> None:
        embs = np.random.default_rng(42).standard_normal((4, 768))
        assert np.allclose(compute_clip_score_batch(embs, embs), 100.0, atol=1e-6)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_clip_score_batch(np.zeros((3, 768)), np.zeros((4, 768)))

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            compute_clip_score_batch(np.zeros(768), np.zeros(768))
