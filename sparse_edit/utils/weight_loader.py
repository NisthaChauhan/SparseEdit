"""Weight loading utilities for safetensors checkpoints.

Handles key remapping from HuggingFace diffusers format to SparseEdit
module paths. MLX Conv2d uses NHWC weights; PyTorch uses NCHW.

Memory: O(total_params). ~1.7 GB for U-Net fp16.
Throughput: ~2 s to load and remap full SD 1.5 checkpoint on M4 Pro.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


def load_safetensors(path: str | Path) -> dict[str, mx.array]:
    """Load a safetensors file into a dict of MLX arrays.

    Parameters
    ----------
    path : str or Path
        Path to .safetensors file.

    Returns
    -------
    dict[str, mx.array]
    """
    weights = mx.load(str(path))
    return weights


def remap_keys(
    weights: dict[str, mx.array],
    key_map: dict[str, str],
) -> dict[str, mx.array]:
    """Remap weight dictionary keys using a mapping.

    Parameters
    ----------
    weights : dict[str, mx.array]
        Original weights with source keys.
    key_map : dict[str, str]
        Mapping from source key -> target key.

    Returns
    -------
    dict[str, mx.array]
        Remapped weights. Unmapped keys are dropped.
    """
    remapped: dict[str, mx.array] = {}
    for src_key, tgt_key in key_map.items():
        if src_key in weights:
            remapped[tgt_key] = weights[src_key]
    return remapped


def transpose_conv_weights(
    weight: mx.array,
) -> mx.array:
    """Transpose Conv2d weights from PyTorch NCHW to MLX NHWC.

    PyTorch shape: (out_ch, in_ch, kH, kW)
    MLX shape:     (out_ch, kH, kW, in_ch)

    Parameters
    ----------
    weight : mx.array, shape (O, I, kH, kW)

    Returns
    -------
    mx.array, shape (O, kH, kW, I)
    """
    return mx.transpose(weight, axes=(0, 2, 3, 1))


def load_and_remap(
    path: str | Path,
    key_map: dict[str, str],
    conv_keys: set[str] | None = None,
) -> dict[str, mx.array]:
    """Load safetensors, remap keys, and transpose conv weights.

    Parameters
    ----------
    path : str or Path
    key_map : dict[str, str]
    conv_keys : set[str] or None
        Set of target keys that are conv weights needing NCHW->NHWC transpose.

    Returns
    -------
    dict[str, mx.array]
    """
    raw = load_safetensors(path)
    remapped = remap_keys(raw, key_map)

    if conv_keys:
        for k in conv_keys:
            if k in remapped and remapped[k].ndim == 4:
                remapped[k] = transpose_conv_weights(remapped[k])

    return remapped
