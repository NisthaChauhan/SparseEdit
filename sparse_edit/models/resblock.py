"""Residual block for U-Net and VAE.

GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU -> Conv + timestep projection.
Optional channel-change shortcut via 1x1 conv.

Memory: O(B * H * W * C). ~5 MB per block at 64x64x320 fp16.
Throughput: ~0.5 ms per block on M4 Pro.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class ResBlock(nn.Module):
    """Residual block with optional timestep conditioning.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    time_embed_dim : int or None
        If provided, adds a projection for timestep embedding.
    norm_groups : int
        Number of groups for GroupNorm.
    norm_eps : float
        Epsilon for GroupNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int | None = 1280,
        norm_groups: int = 32,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(norm_groups, in_channels, eps=norm_eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(norm_groups, out_channels, eps=norm_eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if time_embed_dim is not None:
            self.time_proj = nn.Linear(time_embed_dim, out_channels)
        else:
            self.time_proj = None

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

    def __call__(
        self,
        x: mx.array,
        time_emb: mx.array | None = None,
    ) -> mx.array:
        """Forward pass.

        Parameters
        ----------
        x : mx.array, shape (B, H, W, C_in)  — NHWC format
        time_emb : mx.array, shape (B, time_embed_dim) or None

        Returns
        -------
        mx.array, shape (B, H, W, C_out)
        """
        residual = x

        h = self.norm1(x)
        h = nn.silu(h)
        h = self.conv1(h)

        if self.time_proj is not None and time_emb is not None:
            t = nn.silu(time_emb)
            t = self.time_proj(t)
            # Reshape (B, C) -> (B, 1, 1, C) for broadcasting.
            h = h + t[:, None, None, :]

        h = self.norm2(h)
        h = nn.silu(h)
        h = self.conv2(h)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        return h + residual
