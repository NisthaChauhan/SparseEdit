"""CLIP ViT-L/14 text encoder for Stable Diffusion 1.5.

Architecture: 12 transformer layers, hidden_size=768, 12 heads,
intermediate_size=3072, QuickGELU, causal self-attention, max 77 tokens,
vocab_size=49408.

Outputs: last_hidden_state (B, 77, 768) and pooled_output (B, 768).

Memory: ~246 MB fp16.
Throughput: ~5 ms per forward on M4 Pro.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def quick_gelu(x: mx.array) -> mx.array:
    """QuickGELU activation: x * sigmoid(1.702 * x)."""
    return x * mx.sigmoid(1.702 * x)


class CLIPMLP(nn.Module):
    """CLIP MLP: Linear -> QuickGELU -> Linear."""

    def __init__(self, hidden_size: int = 768, intermediate_size: int = 3072) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(quick_gelu(self.fc1(x)))


class CLIPAttention(nn.Module):
    """Multi-head self-attention for CLIP text encoder."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x: mx.array, causal_mask: mx.array | None = None) -> mx.array:
        B, S, _ = x.shape
        q = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if causal_mask is not None:
            attn = attn + causal_mask

        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.out_proj(out)


class CLIPEncoderLayer(nn.Module):
    """Single CLIP transformer layer: pre-norm attention + pre-norm MLP."""

    def __init__(self, hidden_size: int = 768, num_heads: int = 12, intermediate_size: int = 3072) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.self_attn = CLIPAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.mlp = CLIPMLP(hidden_size, intermediate_size)

    def __call__(self, x: mx.array, causal_mask: mx.array | None = None) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x), causal_mask=causal_mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class CLIPTextEncoder(nn.Module):
    """Full CLIP ViT-L/14 text encoder.

    Parameters
    ----------
    vocab_size : int
    max_length : int
    hidden_size : int
    num_layers : int
    num_heads : int
    intermediate_size : int
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        max_length: int = 77,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.hidden_size = hidden_size

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)

        self.layers = [
            CLIPEncoderLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ]

        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)

    def _build_causal_mask(self, seq_len: int) -> mx.array:
        mask = mx.full((seq_len, seq_len), -1e9)
        mask = mx.triu(mask, k=1)
        return mask

    def __call__(self, input_ids: mx.array) -> tuple[mx.array, mx.array]:
        """Forward pass.

        Parameters
        ----------
        input_ids : mx.array, shape (B, seq_len), dtype int32

        Returns
        -------
        tuple of:
            last_hidden_state : mx.array, shape (B, seq_len, 768)
            pooled_output : mx.array, shape (B, 768)
        """
        B, S = input_ids.shape
        x = self.token_embedding(input_ids) + self.position_embedding(mx.arange(S))
        causal_mask = self._build_causal_mask(S)

        for layer in self.layers:
            x = layer(x, causal_mask=causal_mask)

        last_hidden_state = self.final_layer_norm(x)

        # Pooled output: take the embedding at the EOT token (argmax of input_ids).
        eot_indices = mx.argmax(input_ids, axis=-1)
        pooled_output = last_hidden_state[mx.arange(B), eot_indices]

        return last_hidden_state, pooled_output


# ── Backward-compat tokenize method for sparse_edit.editing.pipeline ──
# pipeline.py calls self.text_encoder.tokenize(prompt) and expects an
# mx.array of shape [1, 77] of int32 token IDs (CLIP standard length).
# CLIPTextEncoder is the transformer model only — it does not own the
# tokenizer. We bolt on a tokenize() method that lazily loads the
# standard OpenAI CLIP-ViT-L/14 tokenizer from HuggingFace.
import numpy as _np_te


def _clip_tokenize(self, prompt: str) -> mx.array:
    """Tokenize a prompt string into shape [1, 77] mx.array of int32 IDs."""
    if not hasattr(self, "_hf_tokenizer"):
        from transformers import CLIPTokenizer
        self._hf_tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
    enc = self._hf_tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="np",
    )
    # enc["input_ids"] is shape (1, 77), dtype int64. Convert to int32 mx.array.
    ids_np = _np_te.asarray(enc["input_ids"], dtype=_np_te.int32)
    return mx.array(ids_np)


# Bolt the method onto CLIPTextEncoder.
CLIPTextEncoder.tokenize = _clip_tokenize
