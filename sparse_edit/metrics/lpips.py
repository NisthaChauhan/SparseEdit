"""Learned Perceptual Image Patch Similarity (LPIPS) metric.

VGG-16 feature extractor at 5 stages: conv1_2, conv2_2, conv3_3, conv4_3, conv5_3.
Channel-normalise, squared difference, optionally weighted by learned linear scalars.

    LPIPS(x, y) = sum_l  w_l * mean_hw( || phi_l(x)_hat - phi_l(y)_hat ||^2 )

Memory
------
    Precomputed features: ~5 MB for 224x224 input.
    Full VGG forward: ~260 MB peak.

Throughput
----------
    ~8 ms per image pair on M4 Pro.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt


VGG_STAGE_CHANNELS: tuple[int, ...] = (64, 128, 256, 512, 512)


def _channel_normalise(
    features: npt.NDArray[np.float64],
    eps: float = 1e-10,
) -> npt.NDArray[np.float64]:
    """Unit-normalise feature maps along the channel axis.

    Parameters
    ----------
    features : ndarray, shape (H, W, C) or (B, H, W, C)
    eps : float

    Returns
    -------
    ndarray, same shape
    """
    norm = np.sqrt(np.sum(features ** 2, axis=-1, keepdims=True) + eps)
    return features / norm


def compute_lpips_from_features(
    features_x: Sequence[npt.NDArray[np.float64]],
    features_y: Sequence[npt.NDArray[np.float64]],
    linear_weights: Sequence[npt.NDArray[np.float64]] | None = None,
) -> float:
    """Compute LPIPS distance from pre-extracted VGG feature maps.

    Parameters
    ----------
    features_x : sequence of 5 ndarrays
        VGG-16 features for image x at each stage. Shape (H_l, W_l, C_l).
    features_y : sequence of 5 ndarrays
        Corresponding features for image y.
    linear_weights : sequence of 5 ndarrays or None
        Learned per-channel weights. Shape (C_l,) each.
        If None, uses uniform weights.

    Returns
    -------
    float
        LPIPS distance. Lower = more similar.

    Memory
    ------
        O(sum_l H_l * W_l * C_l).
    """
    if len(features_x) != 5 or len(features_y) != 5:
        raise ValueError(
            f"Expected 5 feature stages, got {len(features_x)} and "
            f"{len(features_y)}"
        )

    total_dist: float = 0.0

    for stage_idx in range(5):
        fx = features_x[stage_idx].astype(np.float64)
        fy = features_y[stage_idx].astype(np.float64)

        if fx.shape != fy.shape:
            raise ValueError(
                f"Stage {stage_idx} shape mismatch: {fx.shape} vs {fy.shape}"
            )

        fx_hat = _channel_normalise(fx)
        fy_hat = _channel_normalise(fy)
        diff_sq = (fx_hat - fy_hat) ** 2

        if linear_weights is not None:
            w = linear_weights[stage_idx].astype(np.float64)
            diff_sq = diff_sq * w

        if diff_sq.ndim == 4:
            stage_dist = np.mean(
                np.sum(np.mean(diff_sq, axis=(1, 2)), axis=-1)
            )
        elif diff_sq.ndim == 3:
            stage_dist = float(np.sum(np.mean(diff_sq, axis=(0, 1))))
        else:
            raise ValueError(
                f"Unexpected feature ndim {diff_sq.ndim} at stage {stage_idx}"
            )

        total_dist += float(stage_dist)

    return total_dist


# ── Backward-compat alias for sparse_edit.editing.pipeline ──
compute_lpips = compute_lpips_from_features
