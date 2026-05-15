"""Attention mask extraction and modulation for SparseEdit.

Extracts spatial masks from cross-attention maps for specific tokens,
then modulates the sparsity penalty lambda spatially.

Memory: O(H * W) per mask.
Throughput: < 1 ms for mask extraction + interpolation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def extract_token_mask(
    attention_map: npt.NDArray[np.floating],
    token_index: int,
    spatial_size: tuple[int, int],
) -> npt.NDArray[np.float64]:
    """Extract a spatial attention mask for a specific token.

    Parameters
    ----------
    attention_map : ndarray, shape (n_heads, H*W, seq_len)
        Cross-attention weights from one layer.
    token_index : int
        Token position in the prompt to extract.
    spatial_size : (H, W)
        Original spatial dimensions.

    Returns
    -------
    ndarray, shape (H, W), range [0, 1]
    """
    # Average over heads, select token, reshape.
    mask = np.mean(attention_map[:, :, token_index], axis=0)
    H, W = spatial_size
    mask = mask.reshape(H, W).astype(np.float64)

    # Normalise to [0, 1].
    m_min, m_max = mask.min(), mask.max()
    if m_max - m_min > 1e-8:
        mask = (mask - m_min) / (m_max - m_min)
    else:
        mask = np.zeros_like(mask)

    return mask


def aggregate_token_masks(
    attention_maps: list[npt.NDArray[np.floating]],
    token_indices: list[int],
    target_size: tuple[int, int],
) -> npt.NDArray[np.float64]:
    """Aggregate masks across layers and tokens.

    Parameters
    ----------
    attention_maps : list of attention map arrays
    token_indices : list of token positions to aggregate
    target_size : (H, W) for output mask

    Returns
    -------
    ndarray, shape (H, W), range [0, 1]
    """
    from PIL import Image

    masks: list[npt.NDArray[np.float64]] = []
    for attn_map in attention_maps:
        n_heads = attn_map.shape[0]
        hw = attn_map.shape[1]
        side = int(np.sqrt(hw))
        for tok_idx in token_indices:
            m = extract_token_mask(attn_map, tok_idx, (side, side))
            # Resize to target.
            m_pil = Image.fromarray((m * 255).astype(np.uint8), mode="L")
            m_resized = m_pil.resize((target_size[1], target_size[0]), Image.BILINEAR)
            masks.append(np.array(m_resized, dtype=np.float64) / 255.0)

    if not masks:
        return np.zeros(target_size, dtype=np.float64)

    combined = np.mean(np.stack(masks, axis=0), axis=0)
    # Re-normalise.
    c_min, c_max = combined.min(), combined.max()
    if c_max - c_min > 1e-8:
        combined = (combined - c_min) / (c_max - c_min)

    return combined


def modulate_lambda(
    base_lambda: float,
    attention_mask: npt.NDArray[np.float64],
    alpha: float = 0.9,
) -> npt.NDArray[np.float64]:
    """Modulate sparsity lambda using the attention mask.

    High attention = edit region -> low lambda (allow change).
    Low attention = preserve region -> high lambda (suppress change).

    lambda_map = base_lambda * (1 - alpha * attention_mask)

    Parameters
    ----------
    base_lambda : float
    attention_mask : ndarray, shape (H, W), range [0, 1]
    alpha : float in [0, 1]

    Returns
    -------
    ndarray, shape (H, W), non-negative
    """
    lam_map = base_lambda * (1.0 - alpha * attention_mask)
    return np.maximum(lam_map, 0.0)


# ── Backward-compat alias for sparse_edit.editing.pipeline ──
# pipeline.py imports build_attention_mask; the canonical builder is
# aggregate_token_masks (composes per-token masks into a single spatial mask).
build_attention_mask = aggregate_token_masks
