"""Unified evaluation harness for SparseEdit.

Orchestrates all metrics into a single call: PSNR, SSIM, Leakage,
CLIP Score, LPIPS.

Memory
------
    O(H * W * C). 512x512x3 float64: ~12 MB peak for all metrics.

Throughput
----------
    < 10 ms total for metrics 1-7 on 512x512x3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from sparse_edit.metrics.psnr import compute_psnr
from sparse_edit.metrics.ssim import compute_ssim
from sparse_edit.metrics.leakage import compute_leakage
from sparse_edit.metrics.clip_score import compute_clip_score


@dataclass
class EditEvaluation:
    """Container for all evaluation metrics of a single edit."""

    psnr_whole: float = 0.0
    ssim_whole: float = 0.0
    psnr_preserve: float = 0.0
    ssim_preserve: float = 0.0
    leakage_mse: float = 0.0
    leakage_psnr: float = 0.0
    leakage_ratio: float = 0.0
    preserve_fraction: float = 0.0
    clip_score: float | None = None
    lpips: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to flat dictionary."""
        return {
            "psnr_whole": self.psnr_whole,
            "ssim_whole": self.ssim_whole,
            "psnr_preserve": self.psnr_preserve,
            "ssim_preserve": self.ssim_preserve,
            "leakage_mse": self.leakage_mse,
            "leakage_psnr": self.leakage_psnr,
            "leakage_ratio": self.leakage_ratio,
            "preserve_fraction": self.preserve_fraction,
            "clip_score": self.clip_score,
            "lpips": self.lpips,
            **self.extra,
        }

    def summary(self) -> str:
        """Human-readable one-line summary."""
        parts = [
            f"PSNR={self.psnr_whole:.2f}dB",
            f"SSIM={self.ssim_whole:.4f}",
            f"Leak_MSE={self.leakage_mse:.4f}",
            f"Leak_PSNR={self.leakage_psnr:.2f}dB",
            f"Leak_ratio={self.leakage_ratio:.4f}",
        ]
        if self.clip_score is not None:
            parts.append(f"CLIP={self.clip_score:.2f}")
        if self.lpips is not None:
            parts.append(f"LPIPS={self.lpips:.4f}")
        return " | ".join(parts)


def _masked_image(
    image: npt.NDArray[np.float64],
    mask: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Zero out the edit region, keep preserve region."""
    m = mask.astype(np.float64)
    if m.ndim == 2 and image.ndim == 3:
        m = m[:, :, np.newaxis]
    return image * (1.0 - m)


def evaluate_edit(
    source: npt.NDArray[np.floating],
    edited: npt.NDArray[np.floating],
    mask: npt.NDArray[np.floating],
    data_range: float = 255.0,
    image_embedding: npt.NDArray[np.floating] | None = None,
    text_embedding: npt.NDArray[np.floating] | None = None,
    vgg_features_source: list[npt.NDArray[np.float64]] | None = None,
    vgg_features_edited: list[npt.NDArray[np.float64]] | None = None,
    lpips_weights: list[npt.NDArray[np.float64]] | None = None,
) -> EditEvaluation:
    """Run full evaluation suite on one source->edited pair.

    Parameters
    ----------
    source : ndarray, shape (H, W, C)
    edited : ndarray, shape (H, W, C)
    mask : ndarray, shape (H, W) or (H, W, 1)
    data_range : float
    image_embedding, text_embedding : ndarray or None
    vgg_features_source, vgg_features_edited : list or None
    lpips_weights : list or None

    Returns
    -------
    EditEvaluation
    """
    if source.shape != edited.shape:
        raise ValueError(
            f"Shape mismatch: source {source.shape} vs edited {edited.shape}"
        )

    src = source.astype(np.float64)
    edt = edited.astype(np.float64)
    m = mask.astype(np.float64)

    result = EditEvaluation()

    result.psnr_whole = compute_psnr(src, edt, data_range=data_range)
    result.ssim_whole = compute_ssim(src, edt, data_range=data_range)

    src_preserve = _masked_image(src, m)
    edt_preserve = _masked_image(edt, m)
    result.psnr_preserve = compute_psnr(src_preserve, edt_preserve, data_range=data_range)
    result.ssim_preserve = compute_ssim(src_preserve, edt_preserve, data_range=data_range)

    leak = compute_leakage(src, edt, m, data_range=data_range)
    result.leakage_mse = leak["leakage_mse"]
    result.leakage_psnr = leak["leakage_psnr"]
    result.leakage_ratio = leak["leakage_ratio"]
    result.preserve_fraction = leak["preserve_fraction"]

    if image_embedding is not None and text_embedding is not None:
        result.clip_score = compute_clip_score(image_embedding, text_embedding)

    if vgg_features_source is not None and vgg_features_edited is not None:
        from sparse_edit.metrics.lpips import compute_lpips_from_features
        result.lpips = compute_lpips_from_features(
            vgg_features_source, vgg_features_edited,
            linear_weights=lpips_weights,
        )

    return result


def evaluate_batch(
    sources: npt.NDArray[np.floating],
    editeds: npt.NDArray[np.floating],
    masks: npt.NDArray[np.floating],
    data_range: float = 255.0,
    image_embeddings: npt.NDArray[np.floating] | None = None,
    text_embeddings: npt.NDArray[np.floating] | None = None,
) -> list[EditEvaluation]:
    """Evaluate a batch of edits.

    Parameters
    ----------
    sources : ndarray, shape (B, H, W, C)
    editeds : ndarray, shape (B, H, W, C)
    masks : ndarray, shape (B, H, W) or (B, H, W, 1)
    data_range : float
    image_embeddings : ndarray, shape (B, D) or None
    text_embeddings : ndarray, shape (B, D) or None

    Returns
    -------
    list[EditEvaluation]

    Memory
    ------
        O(H * W * C) per iteration — processes one at a time.
    """
    batch_size = sources.shape[0]
    results: list[EditEvaluation] = []

    for i in range(batch_size):
        img_emb = image_embeddings[i] if image_embeddings is not None else None
        txt_emb = text_embeddings[i] if text_embeddings is not None else None
        ev = evaluate_edit(
            source=sources[i], edited=editeds[i], mask=masks[i],
            data_range=data_range,
            image_embedding=img_emb, text_embedding=txt_emb,
        )
        results.append(ev)

    return results


def aggregate_results(
    evaluations: list[EditEvaluation],
) -> dict[str, float]:
    """Compute mean metrics across evaluations.

    Parameters
    ----------
    evaluations : list[EditEvaluation]

    Returns
    -------
    dict[str, float]
    """
    if not evaluations:
        return {}

    keys = [
        "psnr_whole", "ssim_whole", "psnr_preserve", "ssim_preserve",
        "leakage_mse", "leakage_psnr", "leakage_ratio", "preserve_fraction",
    ]
    agg: dict[str, float] = {}

    for k in keys:
        vals = [getattr(e, k) for e in evaluations]
        finite_vals = [v for v in vals if np.isfinite(v)]
        agg[f"mean_{k}"] = float(np.mean(finite_vals)) if finite_vals else float("nan")

    clip_vals = [e.clip_score for e in evaluations if e.clip_score is not None]
    if clip_vals:
        agg["mean_clip_score"] = float(np.mean(clip_vals))

    lpips_vals = [e.lpips for e in evaluations if e.lpips is not None]
    if lpips_vals:
        agg["mean_lpips"] = float(np.mean(lpips_vals))

    agg["num_samples"] = float(len(evaluations))
    return agg
