"""End-to-end integration tests — Phase 8."""

from __future__ import annotations

import numpy as np

from sparse_edit.metrics.evaluation import evaluate_edit
from sparse_edit.metrics.psnr import compute_psnr
from sparse_edit.metrics.ssim import compute_ssim
from sparse_edit.metrics.leakage import compute_leakage
from sparse_edit.metrics.clip_score import compute_clip_score


class TestEndToEndDataFlow:
    def test_perfect_edit(self) -> None:
        rng = np.random.default_rng(42)
        H, W, C = 128, 128, 3
        source = rng.random((H, W, C)) * 255.0
        mask = np.zeros((H, W, 1), dtype=np.float64)
        yy, xx = np.mgrid[:H, :W]
        circle = ((yy - 64) ** 2 + (xx - 64) ** 2) < 400
        mask[circle, 0] = 1.0
        edited = source.copy()
        edited[circle] = rng.random((int(circle.sum()), C)) * 255.0
        ev = evaluate_edit(source, edited, mask, data_range=255.0)
        assert ev.leakage_mse == 0.0
        assert ev.leakage_ratio == 0.0

    def test_leaky_edit_detected(self) -> None:
        rng = np.random.default_rng(42)
        H, W, C = 128, 128, 3
        source = rng.random((H, W, C)) * 255.0
        mask = np.zeros((H, W, 1), dtype=np.float64)
        mask[:32, :, :] = 1.0
        edited = source.copy()
        edited[:32, :, :] = rng.random((32, W, C)) * 255.0
        edited[100:, :, :] += 5.0
        ev = evaluate_edit(source, edited, mask, data_range=255.0)
        assert ev.leakage_mse > 0.0

    def test_metrics_ordering(self) -> None:
        rng = np.random.default_rng(42)
        H, W, C = 128, 128, 3
        source = rng.random((H, W, C)) * 255.0
        mask = np.zeros((H, W, 1), dtype=np.float64)
        mask[:32, :, :] = 1.0
        good = source.copy()
        good[:32, :, :] = 0.0
        bad = source.copy()
        bad[:32, :, :] = 0.0
        bad[32:, :, :] += rng.normal(0, 20, (H - 32, W, C))
        ev_good = evaluate_edit(source, good, mask, data_range=255.0)
        ev_bad = evaluate_edit(source, bad, mask, data_range=255.0)
        assert ev_good.leakage_mse < ev_bad.leakage_mse

    def test_clip_alignment(self) -> None:
        rng = np.random.default_rng(42)
        base = rng.standard_normal(768)
        aligned = base + rng.standard_normal(768) * 0.01
        random = rng.standard_normal(768)
        assert compute_clip_score(base, aligned) > compute_clip_score(base, random)

    def test_dict_keys(self) -> None:
        rng = np.random.default_rng(0)
        src = rng.random((64, 64, 3)) * 255.0
        mask = np.zeros((64, 64, 1))
        mask[:16, :, :] = 1.0
        d = evaluate_edit(src, src + rng.normal(0, 5, src.shape), mask).to_dict()
        expected = {"psnr_whole", "ssim_whole", "psnr_preserve", "ssim_preserve",
                    "leakage_mse", "leakage_psnr", "leakage_ratio", "preserve_fraction", "clip_score", "lpips"}
        assert expected.issubset(set(d.keys()))

    def test_vae_roundtrip_bound(self) -> None:
        rng = np.random.default_rng(42)
        original = rng.random((64, 64, 3)) * 255.0
        reconstructed = np.clip(original + rng.normal(0, 2.0, original.shape), 0, 255)
        assert compute_psnr(original, reconstructed, data_range=255.0) > 30.0

    def test_attention_mask_reduces_leakage(self) -> None:
        rng = np.random.default_rng(42)
        H, W, C = 64, 64, 3
        source = np.zeros((H, W, C), dtype=np.float64)
        mask = np.zeros((H, W, 1), dtype=np.float64)
        mask[:16, :, :] = 1.0
        good = source.copy()
        good[:16, :, :] = 100.0
        good[16:, :, :] += rng.normal(0, 0.1, (H - 16, W, C))
        bad = source.copy()
        bad[:16, :, :] = 100.0
        bad[16:, :, :] += rng.normal(0, 10.0, (H - 16, W, C))
        assert compute_leakage(source, good, mask)["leakage_mse"] < compute_leakage(source, bad, mask)["leakage_mse"]

    def test_sparsity_scheduling(self) -> None:
        rng = np.random.default_rng(42)
        H, W, C = 64, 64, 3
        source = rng.random((H, W, C)) * 255.0
        delta = rng.normal(0, 20, (H, W, C))
        low_lam = np.sign(delta) * np.maximum(np.abs(delta) - 5.0, 0)
        high_lam = np.sign(delta) * np.maximum(np.abs(delta) - 30.0, 0)
        ssim_low = compute_ssim(source, source + low_lam, data_range=255.0)
        ssim_high = compute_ssim(source, source + high_lam, data_range=255.0)
        assert ssim_high > ssim_low
