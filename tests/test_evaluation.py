"""Tests for evaluation harness — Phase 8."""

from __future__ import annotations

import numpy as np
import pytest

from sparse_edit.metrics.evaluation import (
    EditEvaluation,
    evaluate_edit,
    evaluate_batch,
    aggregate_results,
)


class TestEditEvaluation:
    def test_to_dict(self) -> None:
        ev = EditEvaluation(psnr_whole=30.0, ssim_whole=0.95)
        d = ev.to_dict()
        assert d["psnr_whole"] == 30.0
        assert d["clip_score"] is None

    def test_summary(self) -> None:
        ev = EditEvaluation(psnr_whole=30.0, ssim_whole=0.95, leakage_mse=0.5, leakage_psnr=45.0, leakage_ratio=0.01)
        s = ev.summary()
        assert "PSNR=30.00dB" in s

    def test_summary_with_clip(self) -> None:
        assert "CLIP=85.00" in EditEvaluation(clip_score=85.0).summary()


class TestEvaluateEdit:
    def test_identical(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((64, 64, 3)) * 255.0
        mask = np.zeros((64, 64, 1))
        mask[:16, :, :] = 1.0
        ev = evaluate_edit(img, img, mask, data_range=255.0)
        assert ev.psnr_whole == float("inf")
        assert ev.leakage_mse == 0.0

    def test_inside_only(self) -> None:
        src = np.zeros((64, 64, 3))
        edited = src.copy()
        edited[:16, :, :] = 100.0
        mask = np.zeros((64, 64, 1))
        mask[:16, :, :] = 1.0
        ev = evaluate_edit(src, edited, mask, data_range=255.0)
        assert ev.leakage_mse == 0.0

    def test_with_clip(self) -> None:
        rng = np.random.default_rng(42)
        src = rng.random((64, 64, 3)) * 255.0
        mask = np.zeros((64, 64, 1))
        mask[:32, :, :] = 1.0
        emb = rng.standard_normal(768)
        ev = evaluate_edit(src, src + rng.normal(0, 5, src.shape), mask, image_embedding=emb, text_embedding=emb)
        assert ev.clip_score is not None
        assert abs(ev.clip_score - 100.0) < 1e-6

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            evaluate_edit(np.zeros((32, 32, 3)), np.zeros((64, 64, 3)), np.zeros((32, 32, 1)))


class TestEvaluateBatch:
    def test_count(self) -> None:
        rng = np.random.default_rng(0)
        B = 4
        src = rng.random((B, 32, 32, 3)) * 255.0
        edt = src + rng.normal(0, 5, src.shape)
        masks = np.zeros((B, 32, 32, 1))
        masks[:, :8, :, :] = 1.0
        assert len(evaluate_batch(src, edt, masks)) == B

    def test_with_embeddings(self) -> None:
        rng = np.random.default_rng(42)
        B = 3
        src = rng.random((B, 32, 32, 3)) * 255.0
        masks = np.zeros((B, 32, 32, 1))
        masks[:, :8, :, :] = 1.0
        embs = rng.standard_normal((B, 768))
        results = evaluate_batch(src, src, masks, image_embeddings=embs, text_embeddings=embs)
        for ev in results:
            assert abs(ev.clip_score - 100.0) < 1e-6


class TestAggregate:
    def test_mean(self) -> None:
        evs = [EditEvaluation(psnr_whole=30.0, ssim_whole=0.9), EditEvaluation(psnr_whole=40.0, ssim_whole=0.8)]
        agg = aggregate_results(evs)
        assert abs(agg["mean_psnr_whole"] - 35.0) < 1e-10

    def test_inf_filtered(self) -> None:
        evs = [EditEvaluation(psnr_whole=float("inf")), EditEvaluation(psnr_whole=30.0)]
        assert abs(aggregate_results(evs)["mean_psnr_whole"] - 30.0) < 1e-10

    def test_empty(self) -> None:
        assert aggregate_results([]) == {}

    def test_clip_agg(self) -> None:
        evs = [EditEvaluation(clip_score=80.0), EditEvaluation(clip_score=90.0), EditEvaluation()]
        assert abs(aggregate_results(evs)["mean_clip_score"] - 85.0) < 1e-10
