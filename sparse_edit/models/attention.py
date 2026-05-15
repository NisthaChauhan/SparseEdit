"""Self-attention and cross-attention modules for SparseEdit.

Implements SpatialTransformer (GroupNorm -> 1x1 Conv -> Transformer -> 1x1 Conv)
with BasicTransformerBlock (self-attn -> cross-attn -> GeGLU FF).

SD 1.5 uses attention_head_dim=8, so num_heads = channels // 8.
Cross-attention dimension = 768 (CLIP ViT-L/14 hidden size).

Memory: O(B * (H*W)^2 * n_heads) for attention maps. ~200 MB peak at 64x64.
Throughput: ~2 ms per SpatialTransformer on M4 Pro.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class GeGLU(nn.Module):
    """Gated Linear Unit with GELU activation.

    Parameters
    ----------
    dim_in : int
    dim_out : int
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def __call__(self, x: mx.array) -> mx.array:
        x_proj = self.proj(x)
        x, gate = mx.split(x_proj, 2, axis=-1)
        return x * nn.gelu(gate)


class FeedForward(nn.Module):
    """Feed-forward network: GeGLU -> Linear.

    Parameters
    ----------
    dim : int
    mult : int
        Hidden dimension multiplier.
    """

    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        inner_dim = dim * mult
        self.geglu = GeGLU(dim, inner_dim)
        self.linear = nn.Linear(inner_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(self.geglu(x))


class CrossAttention(nn.Module):
    """Multi-head cross-attention (falls back to self-attention if no context).

    Parameters
    ----------
    query_dim : int
    context_dim : int or None
        If None, self-attention.
    n_heads : int
    d_head : int
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        n_heads: int = 8,
        d_head: int = 40,
    ) -> None:
        super().__init__()
        inner_dim = n_heads * d_head
        context_dim = context_dim or query_dim

        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def __call__(
        self,
        x: mx.array,
        context: mx.array | None = None,
    ) -> mx.array:
        """Forward pass.

        Parameters
        ----------
        x : mx.array, shape (B, seq_len, query_dim)
        context : mx.array, shape (B, ctx_len, context_dim) or None

        Returns
        -------
        mx.array, shape (B, seq_len, query_dim)
        """
        if context is None:
            context = x

        B, S, _ = x.shape

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape to (B, seq, n_heads, d_head) then transpose to (B, n_heads, seq, d_head).
        q = q.reshape(B, S, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        attn_weights = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn_weights = mx.softmax(attn_weights, axis=-1)

        out = attn_weights @ v  # (B, n_heads, S, d_head)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    """Pre-norm transformer block: self-attn -> cross-attn -> FF.

    Parameters
    ----------
    dim : int
    n_heads : int
    d_head : int
    context_dim : int
        Cross-attention context dimension (768 for CLIP).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        context_dim: int = 768,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, context_dim=None, n_heads=n_heads, d_head=d_head)

        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, context_dim=context_dim, n_heads=n_heads, d_head=d_head)

        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def __call__(
        self,
        x: mx.array,
        context: mx.array | None = None,
    ) -> mx.array:
        """Forward pass.

        Parameters
        ----------
        x : mx.array, shape (B, seq, dim)
        context : mx.array, shape (B, ctx_len, context_dim) or None

        Returns
        -------
        mx.array, shape (B, seq, dim)
        """
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """Spatial transformer for U-Net: applies attention to spatial feature maps.

    GroupNorm -> 1x1 Conv -> reshape to seq -> Transformer -> reshape back -> 1x1 Conv.

    Parameters
    ----------
    channels : int
    n_heads : int
    d_head : int
    context_dim : int
    n_layers : int
    """

    def __init__(
        self,
        channels: int,
        n_heads: int,
        d_head: int,
        context_dim: int = 768,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)

        self.transformer_blocks = [
            BasicTransformerBlock(channels, n_heads, d_head, context_dim)
            for _ in range(n_layers)
        ]

        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def __call__(
        self,
        x: mx.array,
        context: mx.array | None = None,
    ) -> mx.array:
        """Forward pass.

        Parameters
        ----------
        x : mx.array, shape (B, H, W, C)  — NHWC format
        context : mx.array, shape (B, seq, context_dim) or None

        Returns
        -------
        mx.array, shape (B, H, W, C)
        """
        residual = x
        B, H, W, C = x.shape

        x = self.norm(x)
        x = self.proj_in(x)

        # Reshape (B, H, W, C) -> (B, H*W, C).
        x = x.reshape(B, H * W, C)

        for block in self.transformer_blocks:
            x = block(x, context=context)

        # Reshape back (B, H*W, C) -> (B, H, W, C).
        x = x.reshape(B, H, W, C)
        x = self.proj_out(x)

        return x + residual
