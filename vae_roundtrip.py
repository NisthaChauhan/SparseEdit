import sys
sys.path.insert(0, "/Users/aayush/Projects/sparse_edit")

"""
vae_roundtrip.py
================
Load SD 1.5 VAE weights into AutoencoderKL, encode a real image
to latent space, decode it back, and measure reconstruction quality.

Expected results:
    PSNR > 25 dB (good reconstruction)
    SSIM > 0.85  (structure preserved)
"""

import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from sparse_edit.models.vae import AutoencoderKL, SCALING_FACTOR
from sparse_edit.metrics.psnr import compute_psnr
from sparse_edit.metrics.ssim import compute_ssim


WEIGHTS_DIR = Path("weights/sd-1.5")
INPUT_IMAGE = "input.png"
OUTPUT_IMAGE = "vae_roundtrip_output.png"
USE_FP16 = False
DTYPE = mx.float16 if USE_FP16 else mx.float32


# ─────────────────────────────────────────────────────
#  Weight mapping: HuggingFace -> our AutoencoderKL
# ─────────────────────────────────────────────────────

def map_vae_weights(hf_weights: dict[str, mx.array]) -> list[tuple[str, mx.array]]:
    """Map HuggingFace VAE weight keys to our AutoencoderKL structure."""
    mapped = []

    for hf_key, value in hf_weights.items():
        new_key = hf_key
        val = value.astype(DTYPE)

        # ── quant_conv / post_quant_conv ──
        if hf_key.startswith("quant_conv.") or hf_key.startswith("post_quant_conv."):
            if "weight" in hf_key and val.ndim == 4:
                val = val.transpose(0, 2, 3, 1)
            mapped.append((new_key, val))
            continue

        # ── Mid block resnets ──
        if "mid_block.resnets.0." in new_key:
            new_key = new_key.replace("mid_block.resnets.0.", "mid_block.resnet_1.")
        elif "mid_block.resnets.1." in new_key:
            new_key = new_key.replace("mid_block.resnets.1.", "mid_block.resnet_2.")

        # ── Mid block attention ──
        # HF SD1.5 VAE uses: mid_block.attentions.0.query/key/value/proj_attn
        # Our model uses:     mid_block.attn_q/attn_k/attn_v/attn_out
        elif "mid_block.attentions.0." in new_key:
            new_key = new_key.replace("mid_block.attentions.0.group_norm.", "mid_block.attn_norm.")
            new_key = new_key.replace("mid_block.attentions.0.query.", "mid_block.attn_q.")
            new_key = new_key.replace("mid_block.attentions.0.key.", "mid_block.attn_k.")
            new_key = new_key.replace("mid_block.attentions.0.value.", "mid_block.attn_v.")
            new_key = new_key.replace("mid_block.attentions.0.proj_attn.", "mid_block.attn_out.")
            new_key = new_key.replace("mid_block.attentions.0.to_q.", "mid_block.attn_q.")
            new_key = new_key.replace("mid_block.attentions.0.to_k.", "mid_block.attn_k.")
            new_key = new_key.replace("mid_block.attentions.0.to_v.", "mid_block.attn_v.")
            new_key = new_key.replace("mid_block.attentions.0.to_out.0.", "mid_block.attn_out.")

        # ── Down/Up block resnet mapping ──
        for prefix in ["encoder.down_blocks", "decoder.up_blocks"]:
            for block_i in range(4):
                for res_j in range(4):
                    hf_pat = f"{prefix}.{block_i}.resnets.{res_j}."
                    our_pat = f"{prefix}.{block_i}.{res_j}."
                    if hf_pat in new_key:
                        new_key = new_key.replace(hf_pat, our_pat)

        # ── Downsamplers / Upsamplers ──
        for block_i in range(4):
            hf_ds = f"encoder.down_blocks.{block_i}.downsamplers.0.conv."
            our_ds = f"encoder.downsamplers.{block_i}.conv."
            if hf_ds in new_key:
                new_key = new_key.replace(hf_ds, our_ds)

            hf_us = f"decoder.up_blocks.{block_i}.upsamplers.0.conv."
            our_us = f"decoder.upsamplers.{block_i}.conv."
            if hf_us in new_key:
                new_key = new_key.replace(hf_us, our_us)

        # ── conv_shortcut -> shortcut ──
        new_key = new_key.replace(".conv_shortcut.", ".shortcut.")

        # ── Transpose conv weights: PyTorch (O,I,kH,kW) -> MLX (O,kH,kW,I) ──
        if val.ndim == 4:
            val = val.transpose(0, 2, 3, 1)

        mapped.append((new_key, val))

    return mapped



# ─────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────

def main():
    print()
    print("=" * 60)
    print("  VAE Round-Trip Test (Real Weights)")
    print("=" * 60)

    # ── 1. Load image ──
    print("\n[1/5] Loading image...")
    img = Image.open(INPUT_IMAGE).convert("RGBA")
# Create white background
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    background.paste(img, mask=img.split()[3])
    img = background.convert("RGB").resize((512, 512))

    img_np = np.array(img).astype(np.float32)
    img_normalized = img_np / 127.5 - 1.0  # [-1, 1]
    img_mx = mx.array(img_normalized[np.newaxis]).astype(DTYPE)  # [1, 512, 512, 3]
    print(f"  Image shape: {img_mx.shape}, dtype: {img_mx.dtype}")

    # ── 2. Create model ──
    print("\n[2/5] Creating AutoencoderKL...")
    vae = AutoencoderKL()
    print(f"  Model created.")

    # ── 3. Load weights ──
    print("\n[3/5] Loading and mapping weights...")
    t0 = time.time()

    vae_path = WEIGHTS_DIR / "vae" / "diffusion_pytorch_model.safetensors"
    raw_weights = mx.load(str(vae_path))
    print(f"  Loaded {len(raw_weights)} raw tensors from {vae_path.name}")

    mapped = map_vae_weights(raw_weights)
    print(f"  Mapped to {len(mapped)} weight pairs")

    # Show some mapped keys for debugging
    print(f"  Sample mapped keys:")
    for k, v in mapped[:8]:
        print(f"    {k}: {v.shape}")

    # Load into model (strict=False to skip mismatched keys gracefully)
    try:
        vae.load_weights(mapped, strict=True)
        print(f"  Weights loaded (strict=True) in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  Strict loading failed: {e}")
        print(f"  Trying non-strict loading...")
        vae.load_weights(mapped, strict=False)
        print(f"  Weights loaded (strict=False) in {time.time()-t0:.1f}s")

    # ── 4. Encode -> Decode ──
    print("\n[4/5] Running VAE encode -> decode...")
    t0 = time.time()

    # Encode
    latent = vae.encode(img_mx)  # [1, 64, 64, 4]
    mx.eval(latent)
    print(f"  Latent shape: {latent.shape}, dtype: {latent.dtype}")
    print(f"  Latent range: [{np.array(latent).min():.3f}, {np.array(latent).max():.3f}]")

    # Decode
    reconstructed = vae.decode(latent)  # [1, 512, 512, 3]
    mx.eval(reconstructed)
    print(f"  Reconstructed shape: {reconstructed.shape}")
    print(f"  Encode + Decode time: {time.time()-t0:.2f}s")

    # Convert back to uint8
    recon_np = np.array(reconstructed[0]).astype(np.float32)
    recon_np = np.clip(recon_np, -1.0, 1.0)  # clamp to valid range
    recon_np = np.clip((recon_np + 1.0) * 127.5, 0, 255).astype(np.uint8)


    # Save
    Image.fromarray(recon_np).save(OUTPUT_IMAGE)
    print(f"  Saved reconstruction to: {OUTPUT_IMAGE}")

    # ── 5. Metrics ──
    print("\n[5/5] Computing reconstruction metrics...")
    src_f = img_np  # original [0-255] float32
    rec_f = recon_np.astype(np.float32)

    psnr = compute_psnr(src_f, rec_f, data_range=255.0)
    ssim = compute_ssim(src_f, rec_f, data_range=255.0)

    print(f"  PSNR: {psnr:.2f} dB  (good if > 25)")
    print(f"  SSIM: {ssim:.4f}    (good if > 0.85)")

    # Pixel-level comparison
    diff = np.abs(src_f - rec_f)
    print(f"  Mean pixel error: {diff.mean():.2f}")
    print(f"  Max pixel error:  {diff.max():.2f}")

    if psnr > 25:
        print("\n  [PASS] VAE reconstruction quality is good!")
    elif psnr > 20:
        print("\n  [WARN] VAE reconstruction is acceptable but not great.")
    else:
        print("\n  [FAIL] VAE reconstruction quality is poor — check weight mapping.")

    print()
    print("=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
