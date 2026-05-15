"""Proximal gradient sparse optimizer with soft-thresholding.

Implements the L1-penalised proximal operator:
    prox_λ(x) = sign(x) * max(|x| - λ, 0)

This enforces sparsity on the edit delta at each denoising timestep.

Memory: O(H * W * C) for the delta tensor.
Throughput: < 0.1 ms per call at 64x64x4.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def soft_threshold(
    x: npt.NDArray[np.floating],
    lam: float | npt.NDArray[np.floating],
) -> npt.NDArray[np.float64]:
    """Soft-thresholding (proximal operator of L1 norm).

    Parameters
    ----------
    x : ndarray
        Input tensor.
    lam : float or ndarray
        Non-negative threshold. If array, must be broadcastable to x.

    Returns
    -------
    ndarray
        sign(x) * max(|x| - λ, 0)

    Raises
    ------
    ValueError
        If lambda contains negative values.
    """
    lam_arr = np.asarray(lam, dtype=np.float64)
    if np.any(lam_arr < 0):
        raise ValueError("Lambda must be non-negative for proximal operator.")

    x_f = x.astype(np.float64)
    return np.sign(x_f) * np.maximum(np.abs(x_f) - lam_arr, 0.0)


def compute_sparsity_ratio(delta: npt.NDArray[np.floating]) -> float:
    """Fraction of zero elements in the edit delta.

    Parameters
    ----------
    delta : ndarray

    Returns
    -------
    float in [0, 1]. 1.0 = fully sparse (no edits).
    """
    return float(np.mean(delta == 0.0))


def proximal_gradient_step(
    latent_source: npt.NDArray[np.floating],
    latent_edited: npt.NDArray[np.floating],
    lam: float | npt.NDArray[np.floating],
) -> npt.NDArray[np.float64]:
    """Apply proximal gradient step to enforce sparse edit.

    Computes delta = edited - source, applies soft-thresholding,
    returns source + sparse_delta.

    Parameters
    ----------
    latent_source : ndarray — original latent
    latent_edited : ndarray — edited latent (from denoising)
    lam : float or ndarray — sparsity threshold

    Returns
    -------
    ndarray — source + soft_threshold(edited - source, lam)
    """
    delta = latent_edited.astype(np.float64) - latent_source.astype(np.float64)
    sparse_delta = soft_threshold(delta, lam)
    return latent_source.astype(np.float64) + sparse_delta
