"""Variational Autoencoder (VAE) for Stable Diffusion 1.5.

Encoder: RGB (512x512x3) -> latent (64x64x4) via 4 downsampling stages.
Decoder: latent (64x64x4) -> RGB (512x512x3) via 4 upsampling stages.
Scaling factor: 0.18215.

Block channels: [128, 256, 512, 512]. Layers per block: 2 (encoder), 3 (decoder).
Mid-block: 2 ResBlocks + single-head self-attention at 512 channels.

Memory: ~160 MB fp16.
Throughput: ~15 ms encode, ~20 ms decode on M4 Pro at 512x512.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from sparse_edit.models.resblock import ResBlock


SCALING_FACTOR: float = 0.18215


class Downsample(nn.Module):
    """Stride-2 convolution for spatial downsampling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        # Asymmetric padding: (0,1,0,1) to match diffusers.
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbour 2x upsample followed by 3x3 conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
        return self.conv(x)


class VAEMidBlock(nn.Module):
    """Mid-block: ResBlock -> Self-Attention -> ResBlock."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.resnet_1 = ResBlock(channels, channels, time_embed_dim=None)
        self.resnet_2 = ResBlock(channels, channels, time_embed_dim=None)

        # Single-head self-attention.
        self.attn_norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.attn_q = nn.Linear(channels, channels)
        self.attn_k = nn.Linear(channels, channels)
        self.attn_v = nn.Linear(channels, channels)
        self.attn_out = nn.Linear(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnet_1(x)

        # Self-attention.
        residual = x
        B, H, W, C = x.shape
        h = self.attn_norm(x)
        h = h.reshape(B, H * W, C)

        q = self.attn_q(h)
        k = self.attn_k(h)
        v = self.attn_v(h)

        scale = C ** -0.5
        attn = (q @ k.transpose(0, 2, 1)) * scale
        attn = mx.softmax(attn, axis=-1)
        h = (attn @ v)
        h = self.attn_out(h)
        h = h.reshape(B, H, W, C)
        x = h + residual

        x = self.resnet_2(x)
        return x


class VAEEncoder(nn.Module):
    """VAE encoder: image -> latent mean and logvar."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.down_blocks: list[list[ResBlock]] = []
        self.downsamplers: list[Downsample | None] = []

        ch_in = block_out_channels[0]
        for i, ch_out in enumerate(block_out_channels):
            resnets = []
            for j in range(layers_per_block):
                resnets.append(ResBlock(ch_in if j == 0 else ch_out, ch_out, time_embed_dim=None))
                ch_in = ch_out
            self.down_blocks.append(resnets)

            if i < len(block_out_channels) - 1:
                self.downsamplers.append(Downsample(ch_out))
            else:
                self.downsamplers.append(None)

        self.mid_block = VAEMidBlock(block_out_channels[-1])

        self.conv_norm_out = nn.GroupNorm(32, block_out_channels[-1], eps=1e-6)
        self.conv_out = nn.Conv2d(block_out_channels[-1], latent_channels * 2, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)

        for resnets, ds in zip(self.down_blocks, self.downsamplers):
            for r in resnets:
                x = r(x)
            if ds is not None:
                x = ds(x)

        x = self.mid_block(x)
        x = self.conv_norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)
        return x


class VAEDecoder(nn.Module):
    """VAE decoder: latent -> image."""

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 4,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
    ) -> None:
        super().__init__()
        reversed_channels = list(reversed(block_out_channels))

        self.conv_in = nn.Conv2d(latent_channels, reversed_channels[0], kernel_size=3, padding=1)
        self.mid_block = VAEMidBlock(reversed_channels[0])

        self.up_blocks: list[list[ResBlock]] = []
        self.upsamplers: list[Upsample | None] = []

        ch_in = reversed_channels[0]
        for i, ch_out in enumerate(reversed_channels):
            resnets = []
            for j in range(layers_per_block + 1):
                resnets.append(ResBlock(ch_in if j == 0 else ch_out, ch_out, time_embed_dim=None))
                ch_in = ch_out
            self.up_blocks.append(resnets)

            if i < len(reversed_channels) - 1:
                self.upsamplers.append(Upsample(ch_out))
            else:
                self.upsamplers.append(None)

        self.conv_norm_out = nn.GroupNorm(32, reversed_channels[-1], eps=1e-6)
        self.conv_out = nn.Conv2d(reversed_channels[-1], out_channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        x = self.mid_block(x)

        for resnets, us in zip(self.up_blocks, self.upsamplers):
            for r in resnets:
                x = r(x)
            if us is not None:
                x = us(x)

        x = self.conv_norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)
        return x


class AutoencoderKL(nn.Module):
    """Full VAE: encoder + decoder + quant/post_quant 1x1 convs."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = VAEEncoder()
        self.decoder = VAEDecoder()
        self.quant_conv = nn.Conv2d(8, 8, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(4, 4, kernel_size=1)

    def encode(self, x: mx.array) -> mx.array:
        """Encode image to latent, sample from diagonal Gaussian.

        Parameters
        ----------
        x : mx.array, shape (B, H, W, 3), range [-1, 1]

        Returns
        -------
        mx.array, shape (B, H//8, W//8, 4), scaled by SCALING_FACTOR
        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = mx.split(h, 2, axis=-1)
        # Deterministic encoding: just use the mean.
        return mean * SCALING_FACTOR

    def decode(self, z: mx.array) -> mx.array:
        """Decode latent to image.

        Parameters
        ----------
        z : mx.array, shape (B, H//8, W//8, 4), scaled

        Returns
        -------
        mx.array, shape (B, H, W, 3), range [-1, 1]
        """
        z = z / SCALING_FACTOR
        z = self.post_quant_conv(z)
        return self.decoder(z)
