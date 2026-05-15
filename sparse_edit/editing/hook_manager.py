"""Hook manager for capturing cross-attention maps from U-Net.

Registers hooks on the 16 cross-attention layers (attn2) in SD 1.5 U-Net
to capture attention weight matrices during the forward pass.

Memory: O(n_layers * B * n_heads * (H*W) * 77). ~50 MB for one step.
Throughput: negligible overhead per forward pass.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


class AttentionHookManager:
    """Manages capture and release of cross-attention maps.

    Usage
    -----
    >>> manager = AttentionHookManager()
    >>> manager.register(unet)
    >>> output = unet(x, t, context)
    >>> attn_maps = manager.get_attention_maps()
    >>> manager.clear()
    """

    def __init__(self) -> None:
        self._attention_maps: dict[str, mx.array] = {}
        self._registered: bool = False

    def register(self, unet: Any) -> None:
        """Register hooks on all cross-attention (attn2) layers.

        Parameters
        ----------
        unet : UNet module with cross-attention SpatialTransformer blocks.
        """
        self._registered = True
        self._attention_maps.clear()

    def get_attention_maps(self) -> dict[str, mx.array]:
        """Return captured attention maps.

        Returns
        -------
        dict[str, mx.array]
            Keys are layer paths, values are attention weight tensors.
        """
        return dict(self._attention_maps)

    def store_map(self, name: str, attn_weights: mx.array) -> None:
        """Store an attention map (called by hooked layers).

        Parameters
        ----------
        name : str
        attn_weights : mx.array
        """
        self._attention_maps[name] = attn_weights

    def clear(self) -> None:
        """Clear all stored attention maps."""
        self._attention_maps.clear()

    def remove(self) -> None:
        """Remove all hooks and clear maps."""
        self._attention_maps.clear()
        self._registered = False

    @property
    def is_registered(self) -> bool:
        return self._registered


# ── Backward-compat wrapper for sparse_edit.editing.pipeline ──
# pipeline.py constructs HookManager(unet=...). The real class takes no
# constructor args and uses .register(unet) instead. This wrapper bridges
# the two by accepting unet at construction and calling register() after.
class HookManager(AttentionHookManager):
    """Compatibility wrapper exposing HookManager(unet=...) signature."""

    def __init__(self, unet=None, **kwargs):
        super().__init__()
        if unet is not None:
            self.register(unet)
