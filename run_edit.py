"""
run_edit.py — Run a single sparse edit.
"""

import mlx.core as mx
import numpy as np
from PIL import Image

from sparse_edit.models.text_encoder import CLIPTextEncoder
from sparse_edit.models.vae import VAEEncoder, VAEDecoder
from sparse_edit.models.unet import UNet
from sparse_edit.utils.weight_loader import load_weights
from sparse_edit.utils.image_utils import load_image, save_image
from sparse_edit.editing.pipeline import (
    SparseEditPipeline,
    PipelineConfig,
)


def main():
    # ── 1. Configuration ──────────────────────────────────────
    WEIGHTS_DIR = "./weights/sd-2.1"
    SOURCE_IMAGE = "./examples/cat.png"        # your input image
    SOURCE_PROMPT = "a photo of a cat"
    TARGET_PROMPT = "a photo of a dog"
    OUTPUT_PATH = "./output/edited.png"

    config = PipelineConfig(
        num_inference_steps=50,
        guidance_scale=7.5,
        scheduler_type="ddim",
        lambda_max=0.05,
        lambda_min=0.005,
        attention_threshold=0.3,
        image_size=(512, 512),
        dtype=mx.float16,
        seed=42,
        run_metrics=True,
    )

    # ── 2. Load models ────────────────────────────────────────
    print("Loading text encoder...")
    text_encoder = CLIPTextEncoder()
    load_weights(text_encoder, f"{WEIGHTS_DIR}/text_encoder")

    print("Loading VAE...")
    vae_encoder = VAEEncoder()
    vae_decoder = VAEDecoder()
    load_weights(vae_encoder, f"{WEIGHTS_DIR}/vae", prefix="encoder")
    load_weights(vae_decoder, f"{WEIGHTS_DIR}/vae", prefix="decoder")

    print("Loading U-Net...")
    unet = UNet()
    load_weights(unet, f"{WEIGHTS_DIR}/unet")

    # ── 3. Build pipeline ─────────────────────────────────────
    pipe = SparseEditPipeline(
        text_encoder=text_encoder,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        unet=unet,
        config=config,
    )
    print(pipe)

    # ── 4. Load source image ──────────────────────────────────
    source_img = load_image(SOURCE_IMAGE, size=config.image_size)
    print(f"Source image shape: {source_img.shape}")

    # ── 5. Run the edit ───────────────────────────────────────
    print(f"Editing: '{SOURCE_PROMPT}' → '{TARGET_PROMPT}'")
    result = pipe.edit(
        source_image=source_img,
        source_prompt=SOURCE_PROMPT,
        target_prompt=TARGET_PROMPT,
    )

    # ── 6. Save output ────────────────────────────────────────
    save_image(result.edited_image, OUTPUT_PATH)
    print(f"Saved edited image to {OUTPUT_PATH}")

    # ── 7. Print metrics ──────────────────────────────────────
    if result.metrics:
        print("\n── Metrics ─────────────────────────")
        for name, value in result.metrics.items():
            print(f"  {name:20s}: {value:.4f}")

    # ── 8. Save attention mask ────────────────────────────────
    if result.attention_mask is not None:
        mask_uint8 = (result.attention_mask * 255).astype(np.uint8)
        Image.fromarray(mask_uint8, mode="L").save(
            OUTPUT_PATH.replace(".png", "_mask.png")
        )
        print("Saved attention mask.")


if __name__ == "__main__":
    main()
