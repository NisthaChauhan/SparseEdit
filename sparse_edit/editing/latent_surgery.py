"""Latent surgery: apply sparse proximal edit at each denoising step.

Combines attention mask modulation, sparsity scheduling, and
soft-thresholding into a single per-step operation.

Memory: O(B * H * W * C) for the latent tensor.
Throughput: < 0.5 ms per step at 64x64x4.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from sparse_edit.editing.sparse_optimizer import soft_threshold
from sparse_edit.editing.attention_mask import modulate_lambda


def sparse_latent_step(
    latent_source: npt.NDArray[np.floating],
    latent_denoised: npt.NDArray[np.floating],
    base_lambda: float,
    attention_mask: npt.NDArray[np.float64] | None = None,
    alpha: float = 0.9,
) -> npt.NDArray[np.float64]:
    """Apply one sparse editing step to the latent.

    Parameters
    ----------
    latent_source : ndarray, shape (B, H, W, C) — source latent
    latent_denoised : ndarray, shape (B, H, W, C) — denoised latent
    base_lambda : float — sparsity penalty
    attention_mask : ndarray, shape (H, W) or None
    alpha : float — mask modulation strength

    Returns
    -------
    ndarray, shape (B, H, W, C) — sparse-edited latent
    """
    src = latent_source.astype(np.float64)
    den = latent_denoised.astype(np.float64)
    delta = den - src

    if attention_mask is not None:
        lam_map = modulate_lambda(base_lambda, attention_mask, alpha)
        # Broadcast (H, W) -> (1, H, W, 1) for (B, H, W, C).
        lam_spatial = lam_map[np.newaxis, :, :, np.newaxis]
    else:
        lam_spatial = base_lambda

    sparse_delta = soft_threshold(delta, lam_spatial)
    return src + sparse_delta


# ── Backward-compat alias for sparse_edit.editing.pipeline ──
# pipeline.py imports latent_blend; the actual function that performs the
# sparse-blended latent update is sparse_latent_step.
latent_blend = sparse_latent_step
