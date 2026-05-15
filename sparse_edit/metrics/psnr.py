"""Peak Signal-to-Noise Ratio (PSNR) metric.

Measures pixel-level fidelity between two images.

Formula
-------
    MSE = mean((img1 - img2)^2)
    PSNR = 10 * log10(data_range^2 / MSE)      [dB]

When MSE = 0, PSNR = +inf (identical images).

Memory
------
    O(H * W * C). For 512x512x3 float64: ~6 MB peak.

Throughput
----------
    < 1 ms for 512x512x3 on modern CPU.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_psnr(
    img1: npt.NDArray[np.floating],
    img2: npt.NDArray[np.floating],
    data_range: float = 255.0,
) -> float:
    """Compute PSNR between two images.

    Parameters
    ----------
    img1 : ndarray, shape (H, W) or (H, W, C)
        Reference image.
    img2 : ndarray, shape (H, W) or (H, W, C)
        Test image.
    data_range : float, default 255.0
        Dynamic range of pixel values.

    Returns
    -------
    float
        PSNR in decibels. ``float('inf')`` when images are identical.

    Raises
    ------
    ValueError
        If shapes mismatch or fewer than 2 dimensions.
    """
    if img1.shape != img2.shape:
        raise ValueError(
            f"Shape mismatch: img1 {img1.shape} vs img2 {img2.shape}"
        )
    if img1.ndim < 2:
        raise ValueError(
            f"Images must have at least 2 dimensions, got {img1.ndim}"
        )

    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    mse: float = float(np.mean((a - b) ** 2))

    if mse == 0.0:
        return float("inf")

    return float(10.0 * np.log10(data_range**2 / mse))


def compute_psnr_batch(
    imgs1: npt.NDArray[np.floating],
    imgs2: npt.NDArray[np.floating],
    data_range: float = 255.0,
) -> npt.NDArray[np.float64]:
    """Compute PSNR for a batch of image pairs.

    Parameters
    ----------
    imgs1 : ndarray, shape (B, H, W) or (B, H, W, C)
    imgs2 : ndarray, shape (B, H, W) or (B, H, W, C)
    data_range : float, default 255.0

    Returns
    -------
    ndarray, shape (B,)
        Per-image PSNR values in dB.

    Memory
    ------
        O(B * H * W * C). For 16 x 512x512x3 float64: ~96 MB peak.
    """
    if imgs1.shape != imgs2.shape:
        raise ValueError(
            f"Shape mismatch: imgs1 {imgs1.shape} vs imgs2 {imgs2.shape}"
        )
    if imgs1.ndim < 3:
        raise ValueError(
            f"Batch images must have >= 3 dimensions, got {imgs1.ndim}"
        )

    a = imgs1.astype(np.float64)
    b = imgs2.astype(np.float64)
    reduce_axes = tuple(range(1, a.ndim))
    mse = np.mean((a - b) ** 2, axis=reduce_axes)

    psnr = np.where(
        mse == 0.0,
        np.inf,
        10.0 * np.log10(data_range**2 / mse),
    )
    return psnr.astype(np.float64)
