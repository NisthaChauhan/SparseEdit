"""
ui_backend.py
-------------
Real prompt-driven image editor backed by SDXL Turbo (MLX).

Drop-in replacement for the previous SparseEdit adapter. Same EditRequest
and EditResult shapes, so app.py needs no changes.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable

import numpy as np
from PIL import Image

import mlx.core as mx

# MLX GPU streams are thread-local; Gradio dispatches on a worker thread
# where no GPU stream exists. CPU mode avoids that constraint.
mx.set_default_device(mx.cpu)

# Make the local mlx-examples SD package importable.
_MLX_SD_PATH = "/Users/aayush/Projects/mlx-examples/stable_diffusion"
if _MLX_SD_PATH not in sys.path:
    sys.path.insert(0, _MLX_SD_PATH)


# --------------------------------------------------------------------- #
# Public dataclasses (unchanged shape — app.py depends on these)        #
# --------------------------------------------------------------------- #
@dataclass
class EditRequest:
    source_image: Image.Image
    prompt: str
    source_prompt: str = ""
    num_steps: int = 4
    guidance_scale: float = 0.0
    lambda_sparsity: float = 0.05
    tau: float = 0.02
    eta: float = 0.1
    seed: int = 0
    height: int = 512
    width: int = 512
    strength: float = 0.55


@dataclass
class EditResult:
    edited_image: Image.Image
    metrics: Dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0
    sparsity_ratio: float = 0.0
    backend: str = "sdxl-turbo-mlx"


# --------------------------------------------------------------------- #
# PIL <-> MLX helpers                                                   #
# --------------------------------------------------------------------- #
def pil_to_mx(img: Image.Image, height: int, width: int) -> mx.array:
    img = img.convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    arr = arr[None, ...]
    return mx.array(arr, dtype=mx.float32)


def mx_to_pil(arr: mx.array) -> Image.Image:
    a = np.array(arr)
    if a.ndim == 4:
        a = a[0]
    a = ((a + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(a)


# --------------------------------------------------------------------- #
# Lazy SDXL Turbo loader                                                #
# --------------------------------------------------------------------- #
_PIPELINE: Optional[Any] = None
_BACKEND: str = "uninitialized"


def _load_pipeline() -> None:
    global _PIPELINE, _BACKEND
    if _PIPELINE is not None:
        return
    print("[ui_backend] Loading SDXL Turbo (one-time op)...")
    t0 = time.time()
    try:
        from stable_diffusion import StableDiffusionXL
        sd = StableDiffusionXL(model="stabilityai/sdxl-turbo", float16=True)
        sd.ensure_models_are_loaded()
        _PIPELINE = sd
        _BACKEND = "sdxl-turbo-mlx"
        print(f"[ui_backend] SDXL Turbo ready ({time.time()-t0:.1f}s).")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ui_backend] SDXL Turbo load FAILED: {e}")
        _PIPELINE = None
        _BACKEND = "mock"


def warm_up() -> None:
    _load_pipeline()


def get_pipeline() -> Optional[Any]:
    _load_pipeline()
    return _PIPELINE


def get_backend_name() -> str:
    return _BACKEND


# --------------------------------------------------------------------- #
# Progress-callback adapter (tolerates multiple cb signatures)          #
# --------------------------------------------------------------------- #
def _safe_cb(cb, step, total, msg):
    if cb is None:
        return
    try:
        cb(int(step), int(total), str(msg)); return
    except TypeError:
        pass
    try:
        cb(int(step), int(total)); return
    except TypeError:
        pass
    try:
        frac = float(step) / max(float(total), 1.0)
        cb(frac, str(msg))
    except Exception:
        pass


# --------------------------------------------------------------------- #
# Metrics                                                               #
# --------------------------------------------------------------------- #
def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-10:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    return float(num / den) if den > 0 else 0.0


def _diff_sparsity(a: np.ndarray, b: np.ndarray, threshold: int = 8) -> float:
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=-1)
    return float(np.mean(diff <= threshold))


# --------------------------------------------------------------------- #
# Core edit function                                                    #
# --------------------------------------------------------------------- #
def edit_image(req: EditRequest,
               progress_cb: Optional[Callable] = None) -> EditResult:
    mx.set_default_device(mx.cpu)
    n_steps_hint = max(1, min(int(getattr(req, "num_steps", 4)), 4))

    _safe_cb(progress_cb, 0, n_steps_hint, "Loading model...")
    _load_pipeline()

    if _PIPELINE is None:
        _safe_cb(progress_cb, 1, 1, "Mock fallback (model unavailable).")
        src = req.source_image.convert("RGB").resize(
            (req.width, req.height), Image.LANCZOS
        )
        return EditResult(
            edited_image=src,
            metrics={"PSNR_bg": 99.0, "SSIM_bg": 1.0, "Sparsity": 1.0,
                     "Leakage": 0.0, "CLIP_score": 0.0},
            elapsed_s=0.0,
            sparsity_ratio=1.0,
            backend="mock",
        )

    sd = _PIPELINE
    t_start = time.time()

    H = max(256, (req.height // 64) * 64)
    W = max(256, (req.width // 64) * 64)

    _safe_cb(progress_cb, 0, n_steps_hint, "Encoding source image...")

    src_pil = req.source_image.convert("RGB").resize((W, H), Image.LANCZOS)
    src_np = np.asarray(src_pil, dtype=np.uint8)
    src_arr = mx.array(
        np.asarray(src_pil, dtype=np.float32) / 255.0,
        dtype=mx.float32,
    )

    n_steps = max(1, min(int(req.num_steps), 4))
    cfg = max(0.0, float(req.guidance_scale))
    strength = float(np.clip(getattr(req, "strength", 0.55), 0.05, 0.95))

    _safe_cb(progress_cb, 0, n_steps,
             f"Denoising ({n_steps} steps, strength={strength:.2f})...")

    latents_iter = sd.generate_latents_from_image(
        image=src_arr,
        text=req.prompt,
        n_images=1,
        strength=strength,
        num_steps=n_steps,
        cfg_weight=cfg,
        negative_text="",
        seed=int(req.seed) if req.seed else None,
    )

    last_latent = None
    total = n_steps
    for i, x_t in enumerate(latents_iter):
        mx.eval(x_t)
        last_latent = x_t
        _safe_cb(progress_cb, i + 1, total, f"Denoising step {i+1}/{total}")

    _safe_cb(progress_cb, n_steps, n_steps, "Decoding latent...")
    decoded = sd.decode(last_latent)
    mx.eval(decoded)

    out_np = np.array(decoded)
    if out_np.ndim == 4:
        out_np = out_np[0]
    out_np = (out_np * 255.0).clip(0, 255).astype(np.uint8)
    edited_pil = Image.fromarray(out_np)

    _safe_cb(progress_cb, n_steps, n_steps, "Computing metrics...")

    psnr_v = _psnr(src_np, out_np)
    ssim_v = _ssim_simple(src_np.mean(-1), out_np.mean(-1))
    spars = _diff_sparsity(src_np, out_np)

    elapsed = time.time() - t_start
    _safe_cb(progress_cb, n_steps, n_steps, f"Done in {elapsed:.1f}s")

    return EditResult(
        edited_image=edited_pil,
        metrics={
            "PSNR_bg": round(psnr_v, 2),
            "SSIM_bg": round(ssim_v, 4),
            "Leakage": round(1.0 - spars, 4),
            "CLIP_score": 0.0,
            "Sparsity": round(spars, 4),
        },
        elapsed_s=round(elapsed, 2),
        sparsity_ratio=spars,
        backend=_BACKEND,
    )
