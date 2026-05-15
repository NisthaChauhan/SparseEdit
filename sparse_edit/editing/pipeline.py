"""
sparse_edit/editing/pipeline.py
================================

End-to-end sparse image editing pipeline for Apple Silicon (MLX-only).

This module ties together every component built across Phases 1–8:

    Phase 1 – Foundation blocks (ResBlock, Attention, TimestepEmbedding)
    Phase 2 – CLIP Text Encoder
    Phase 3 – VAE (encode / decode)
    Phase 4 – U-Net (eps-prediction with cross-attention maps)
    Phase 5 – Schedulers (DDIM, DDIM Inversion, Rectified Flow)
    Phase 6 – Sparse Editing (sparse optimizer, attention masks, hooks,
              sparsity scheduler, latent surgery)
    Phase 7 – Metrics (PSNR, SSIM, Leakage, CLIP Score, LPIPS)
    Phase 8 – Evaluation harness & CLI

Pipeline overview
-----------------
1.  Encode the source image to latent z₀ via the VAE encoder.
2.  Encode the source and target text prompts via the CLIP text encoder.
3.  Invert z₀ → z_T using DDIM Inversion (or Rectified-Flow forward ODE).
4.  Denoise z_T → z₀_edited over T steps, at each step:
        a. Predict noise ε via the U-Net (with cross-attention on target prompt).
        b. Extract cross-attention maps via hook_manager.
        c. Build a soft spatial mask from the attention maps.
        d. Compute the sparsity coefficient λ(t) via cosine-annealed schedule.
        e. Apply the proximal (soft-thresholding) operator on the latent delta.
        f. Blend edited and original latents using the attention mask
           (latent surgery) to localise the edit.
5.  Decode z₀_edited → edited image via the VAE decoder.
6.  (Optional) Evaluate with PSNR, SSIM, Leakage, CLIP Score, LPIPS.

Constraints
-----------
*   MLX backend only – no PyTorch, no diffusers.
*   All tensors are ``mx.array`` in NHWC layout (Apple convention).
*   Full type annotations, ruff-compliant formatting.
*   Proximal operator enforces λ ≥ 0 (safeguarded).

References
----------
*   Song et al., DDIM (ICLR 2021)
*   Rout et al., RF-Inversion (ICLR 2025)
*   CompVis Stable Diffusion (LDM, CVPR 2022)
*   MLX examples – ml-explore/mlx-examples
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import numpy.typing as npt

# ── internal imports (Phases 1-8) ──────────────────────────────────────
from sparse_edit.models.text_encoder import CLIPTextEncoder
from sparse_edit.models.unet import UNet
from sparse_edit.models.vae import VAEDecoder, VAEEncoder

from sparse_edit.schedulers.ddim import DDIMScheduler
from sparse_edit.schedulers.ddim_inversion import DDIMInversionScheduler
from sparse_edit.schedulers.rectified_flow import RectifiedFlowScheduler

from sparse_edit.editing.sparse_optimizer import soft_threshold
from sparse_edit.editing.hook_manager import HookManager
from sparse_edit.editing.attention_mask import build_attention_mask
from sparse_edit.editing.sparsity_scheduler import cosine_anneal_lambda
from sparse_edit.editing.latent_surgery import latent_blend

from sparse_edit.metrics.psnr import compute_psnr
from sparse_edit.metrics.ssim import compute_ssim
from sparse_edit.metrics.leakage import compute_leakage
from sparse_edit.metrics.clip_score import compute_clip_score
from sparse_edit.metrics.lpips import compute_lpips

from sparse_edit.utils.image_utils import (
    load_image,
    save_image,
    preprocess_image,
    postprocess_image,
)

# ── public API ─────────────────────────────────────────────────────────
__all__ = [
    "PipelineConfig",
    "EditResult",
    "SparseEditPipeline",
]


# ───────────────────────────────────────────────────────────────────────
#  Configuration
# ───────────────────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    """Immutable configuration for :class:`SparseEditPipeline`.

    Parameters
    ----------
    num_inference_steps : int
        Number of denoising steps (default 50).
    guidance_scale : float
        Classifier-free guidance scale (default 7.5).
    scheduler_type : str
        ``"ddim"`` or ``"rectified_flow"`` (default ``"ddim"``).
    lambda_max : float
        Maximum sparsity coefficient for the proximal operator.
    lambda_min : float
        Minimum sparsity coefficient at the final timestep.
    attention_threshold : float
        Soft threshold applied to cross-attention maps to build the
        spatial mask.  Range [0, 1] (default 0.3).
    mask_blur_sigma : float
        Gaussian blur σ applied to the attention mask for smooth
        boundaries (default 1.0).
    blend_strength : float
        Blending factor in latent surgery.  1.0 = full edit,
        0.0 = original image (default 1.0).
    vae_scale_factor : float
        Scaling factor for VAE latents (default 0.18215 for SD 1.x).
    image_size : tuple[int, int]
        Target (H, W) for the input image (default (512, 512)).
    dtype : mx.Dtype
        Computation dtype (default ``mx.float16``).
    seed : int | None
        Reproducibility seed (default ``None``).
    eta : float
        DDIM stochasticity parameter η ∈ [0, 1] (default 0.0 for
        deterministic DDIM).
    rf_gamma : float
        Controller guidance γ for rectified-flow inversion
        (default 0.5).  See Rout et al., ICLR 2025.
    rf_eta : float
        Controller guidance η for rectified-flow reverse
        (default 0.5).
    collect_intermediates : bool
        If True, store latent snapshots at every step (default False).
    run_metrics : bool
        If True, compute PSNR / SSIM / Leakage after editing
        (default False).
    """

    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    scheduler_type: Literal["ddim", "rectified_flow"] = "ddim"
    lambda_max: float = 0.05
    lambda_min: float = 0.005
    attention_threshold: float = 0.3
    mask_blur_sigma: float = 1.0
    blend_strength: float = 1.0
    vae_scale_factor: float = 0.18215
    image_size: tuple[int, int] = (512, 512)
    dtype: Any = mx.float16  # mx.Dtype not subscriptable at runtime
    seed: int | None = None
    eta: float = 0.0
    rf_gamma: float = 0.5
    rf_eta: float = 0.5
    collect_intermediates: bool = False
    run_metrics: bool = False


# ───────────────────────────────────────────────────────────────────────
#  Result container
# ───────────────────────────────────────────────────────────────────────
@dataclass
class EditResult:
    """Container returned by :meth:`SparseEditPipeline.edit`.

    Attributes
    ----------
    edited_image : npt.NDArray[np.uint8]
        Edited image as HWC uint8 NumPy array.
    source_image : npt.NDArray[np.uint8]
        Original source image (HWC uint8).
    attention_mask : npt.NDArray[np.float32] | None
        Final soft attention mask [H, W] in [0, 1], or None.
    latent_trajectory : list[mx.array]
        Latent snapshots if ``collect_intermediates=True``, else empty.
    metrics : dict[str, float]
        Metric results if ``run_metrics=True``, else empty dict.
    """

    edited_image: npt.NDArray[np.uint8]
    source_image: npt.NDArray[np.uint8]
    attention_mask: npt.NDArray[np.float32] | None = None
    latent_trajectory: list[mx.array] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


# ───────────────────────────────────────────────────────────────────────
#  Pipeline
# ───────────────────────────────────────────────────────────────────────
class SparseEditPipeline:
    """End-to-end sparse latent editing pipeline (MLX-only).

    Parameters
    ----------
    text_encoder : CLIPTextEncoder
        Phase 2 text encoder.
    vae_encoder : VAEEncoder
        Phase 3 encoder (image → latent).
    vae_decoder : VAEDecoder
        Phase 3 decoder (latent → image).
    unet : UNet
        Phase 4 U-Net with cross-attention hooks.
    config : PipelineConfig
        Pipeline hyper-parameters.

    Example
    -------
    >>> pipe = SparseEditPipeline(text_enc, vae_enc, vae_dec, unet)
    >>> result = pipe.edit(
    ...     source_image=img,
    ...     source_prompt="a photo of a cat",
    ...     target_prompt="a photo of a dog",
    ... )
    >>> save_image(result.edited_image, "edited.png")
    """

    def __init__(
        self,
        text_encoder: CLIPTextEncoder,
        vae_encoder: VAEEncoder,
        vae_decoder: VAEDecoder,
        unet: UNet,
        config: PipelineConfig | None = None,
    ) -> None:
        self.text_encoder = text_encoder
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.unet = unet
        self.config = config or PipelineConfig()

        # ── build scheduler ────────────────────────────────────────
        self._forward_scheduler: DDIMInversionScheduler | RectifiedFlowScheduler
        self._reverse_scheduler: DDIMScheduler | RectifiedFlowScheduler

        if self.config.scheduler_type == "ddim":
            self._forward_scheduler = DDIMInversionScheduler(
                num_train_timesteps=1000,
                num_inference_steps=self.config.num_inference_steps,
            )
            self._reverse_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                num_inference_steps=self.config.num_inference_steps,
                eta=self.config.eta,
            )
        elif self.config.scheduler_type == "rectified_flow":
            self._forward_scheduler = RectifiedFlowScheduler(
                num_inference_steps=self.config.num_inference_steps,
            )
            self._reverse_scheduler = RectifiedFlowScheduler(
                num_inference_steps=self.config.num_inference_steps,
            )
        else:
            raise ValueError(
                f"Unknown scheduler_type: {self.config.scheduler_type!r}. "
                "Choose 'ddim' or 'rectified_flow'."
            )

        # ── hook manager for cross-attention extraction ────────────
        self._hook_manager = HookManager(unet=self.unet)

        # ── optional RNG key ───────────────────────────────────────
        if self.config.seed is not None:
            mx.random.seed(self.config.seed)

    # ───────────────────────────────────────────────────────────────
    #  Public API
    # ───────────────────────────────────────────────────────────────
    def edit(
        self,
        source_image: npt.NDArray[np.uint8] | mx.array,
        source_prompt: str,
        target_prompt: str,
        mask: npt.NDArray[np.float32] | mx.array | None = None,
    ) -> EditResult:
        """Run a sparse edit from *source_prompt* → *target_prompt*.

        Parameters
        ----------
        source_image : ndarray | mx.array
            Source image as HWC uint8 (NumPy) or NHWC float (MLX).
        source_prompt : str
            Text prompt describing the **source** image.
        target_prompt : str
            Text prompt describing the **desired** edit.
        mask : ndarray | mx.array | None
            Optional explicit binary/soft mask [H, W] in [0, 1].
            If provided, overrides the automatic attention-derived mask.

        Returns
        -------
        EditResult
            Container with the edited image, mask, optional latent
            trajectory, and optional quality metrics.
        """
        cfg = self.config

        # ── 1. Preprocess source image ────────────────────────────
        source_np = self._ensure_numpy_uint8(source_image)
        x_pixel = preprocess_image(
            source_np,
            target_size=cfg.image_size,
            dtype=cfg.dtype,
        )  # mx.array [1, H, W, 3] in [-1, 1]

        # ── 2. Encode image → latent z₀ ──────────────────────────
        z_0 = self._encode_image(x_pixel)  # [1, H//8, W//8, C]
        mx.eval(z_0)

        # ── 3. Encode text prompts ────────────────────────────────
        cond_source = self._encode_text(source_prompt)   # [1, S, D]
        cond_target = self._encode_text(target_prompt)   # [1, S, D]
        cond_uncond = self._encode_text("")               # [1, S, D]
        mx.eval(cond_source, cond_target, cond_uncond)

        # ── 4. Inversion: z₀ → z_T ───────────────────────────────
        z_T = self._invert(z_0, cond_source, cond_uncond)
        mx.eval(z_T)

        # ── 5. Sparse denoising: z_T → z₀_edited ─────────────────
        z_edited, attn_mask_mx, trajectory = self._denoise_sparse(
            z_T=z_T,
            z_source=z_0,
            cond_target=cond_target,
            cond_uncond=cond_uncond,
            explicit_mask=self._to_mx(mask) if mask is not None else None,
        )
        mx.eval(z_edited)

        # ── 6. Decode latent → image ─────────────────────────────
        edited_pixel = self._decode_latent(z_edited)  # [1, H, W, 3]
        mx.eval(edited_pixel)

        edited_np = postprocess_image(edited_pixel)  # HWC uint8

        # ── 7. Build attention mask (NumPy for output) ────────────
        attn_mask_np: npt.NDArray[np.float32] | None = None
        if attn_mask_mx is not None:
            attn_mask_np = np.array(attn_mask_mx[0, :, :, 0], dtype=np.float32)

        # ── 8. Optional metrics ───────────────────────────────────
        metrics: dict[str, float] = {}
        if cfg.run_metrics:
            metrics = self._compute_metrics(
                source_np=source_np,
                edited_np=edited_np,
                mask_np=attn_mask_np,
                target_prompt=target_prompt,
            )

        return EditResult(
            edited_image=edited_np,
            source_image=source_np,
            attention_mask=attn_mask_np,
            latent_trajectory=trajectory,
            metrics=metrics,
        )

    # ───────────────────────────────────────────────────────────────
    #  Internal: Encoding
    # ───────────────────────────────────────────────────────────────
    def _encode_image(self, x: mx.array) -> mx.array:
        """Encode pixel-space image to VAE latent.

        Parameters
        ----------
        x : mx.array
            [1, H, W, 3] float in [-1, 1].

        Returns
        -------
        mx.array
            [1, H//8, W//8, C] scaled latent.
        """
        posterior = self.vae_encoder(x)  # returns (mean, logvar) concatenated
        # The encoder outputs 2*latent_channels (mean and log-variance).
        # For deterministic editing, take only the mean (first half of channels).
        if posterior.shape[-1] % 2 == 0:
            mean, _logvar = mx.split(posterior, 2, axis=-1)
        else:
            mean = posterior  # already split upstream
        z = mean * self.config.vae_scale_factor
        return z

    def _decode_latent(self, z: mx.array) -> mx.array:
        """Decode VAE latent back to pixel space.

        Parameters
        ----------
        z : mx.array
            [1, H//8, W//8, C] scaled latent.

        Returns
        -------
        mx.array
            [1, H, W, 3] float in [-1, 1].
        """
        z_unscaled = z / self.config.vae_scale_factor
        x = self.vae_decoder(z_unscaled)
        return x

    def _encode_text(self, prompt: str) -> mx.array:
        """Encode a text prompt via the CLIP text encoder.

        Parameters
        ----------
        prompt : str
            Natural-language text prompt.

        Returns
        -------
        mx.array
            [1, seq_len, hidden_dim] text embeddings.
        """
        tokens = self.text_encoder.tokenize(prompt)  # mx.array [1, S]
        embeddings = self.text_encoder(tokens)        # may be tuple
        # CLIPTextEncoder returns (last_hidden_state, pooled_output).
        # Cross-attention wants the per-token hidden states (first element).
        if isinstance(embeddings, tuple):
            embeddings = embeddings[0]
        return embeddings

    # ───────────────────────────────────────────────────────────────
    #  Internal: DDIM Inversion
    # ───────────────────────────────────────────────────────────────
    def _invert(
        self,
        z_0: mx.array,
        cond_text: mx.array,
        cond_uncond: mx.array,
    ) -> mx.array:
        """Invert z₀ → z_T via the forward scheduler.

        For DDIM: deterministic DDIM inversion (Song et al., 2021a).
        For Rectified Flow: controlled forward ODE (Rout et al., 2025).

        Parameters
        ----------
        z_0 : mx.array
            Source latent [1, H', W', C].
        cond_text : mx.array
            Source-prompt embeddings [1, S, D].
        cond_uncond : mx.array
            Unconditional (null) embeddings [1, S, D].

        Returns
        -------
        mx.array
            Inverted noise latent z_T of the same shape.
        """
        cfg = self.config
        timesteps = self._forward_scheduler.get_timesteps()  # ascending

        z_t = z_0
        for i, t in enumerate(timesteps):
            t_mx = mx.array(t, dtype=cfg.dtype)

            # ── predict noise with CFG ────────────────────────────
            eps_uncond = self.unet(z_t, t_mx, cond_uncond)
            eps_cond = self.unet(z_t, t_mx, cond_text)
            eps = eps_uncond + cfg.guidance_scale * (eps_cond - eps_uncond)

            # ── forward step (add noise) ──────────────────────────
            if cfg.scheduler_type == "ddim":
                z_t = self._forward_scheduler.step(
                    model_output=eps,
                    timestep=t,
                    sample=z_t,
                )
            else:
                # Rectified flow: controlled forward ODE
                # dY = [u_t(Y) + γ (u_t(Y|y₁) - u_t(Y))] dt
                z_t = self._forward_scheduler.step_forward(
                    model_output=eps,
                    timestep=t,
                    sample=z_t,
                    gamma=cfg.rf_gamma,
                )

            # Evaluate lazily every step to manage memory
            mx.eval(z_t)

        return z_t

    # ───────────────────────────────────────────────────────────────
    #  Internal: Sparse Denoising Loop
    # ───────────────────────────────────────────────────────────────
    def _denoise_sparse(
        self,
        z_T: mx.array,
        z_source: mx.array,
        cond_target: mx.array,
        cond_uncond: mx.array,
        explicit_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None, list[mx.array]]:
        """Denoise z_T → z₀_edited with sparse proximal updates.

        At each timestep the pipeline:
        1. Predicts noise ε with classifier-free guidance.
        2. Takes a standard DDIM/RF reverse step → z_prev_standard.
        3. Computes Δ = z_prev_standard − z_source_at_t.
        4. Applies soft-thresholding: Δ_sparse = S_λ(Δ).
        5. Reconstructs: z_prev = z_source_at_t + mask ⊙ Δ_sparse.

        Parameters
        ----------
        z_T : mx.array
            Starting noise latent [1, H', W', C].
        z_source : mx.array
            Source clean latent z₀ [1, H', W', C].
        cond_target : mx.array
            Target-prompt embeddings [1, S, D].
        cond_uncond : mx.array
            Null-prompt embeddings [1, S, D].
        explicit_mask : mx.array | None
            Optional user-provided mask [1, H', W', 1] in [0, 1].

        Returns
        -------
        z_edited : mx.array
            Denoised, edited latent [1, H', W', C].
        attn_mask : mx.array | None
            Spatial attention mask (last step) or None.
        trajectory : list[mx.array]
            Latent snapshots (if ``collect_intermediates``).
        """
        cfg = self.config
        timesteps = self._reverse_scheduler.get_timesteps()  # descending
        num_steps = len(timesteps)

        z_t = z_T
        trajectory: list[mx.array] = []
        attn_mask: mx.array | None = None

        # Install cross-attention hooks
        self._hook_manager.register_hooks()

        try:
            for step_idx, t in enumerate(timesteps):
                t_mx = mx.array(t, dtype=cfg.dtype)

                # ── (a) Predict noise with classifier-free guidance ──
                eps_uncond = self.unet(z_t, t_mx, cond_uncond)
                eps_cond = self.unet(z_t, t_mx, cond_target)
                eps = eps_uncond + cfg.guidance_scale * (
                    eps_cond - eps_uncond
                )

                # ── (b) Extract cross-attention maps ─────────────────
                cross_attn_maps = self._hook_manager.get_attention_maps()
                # cross_attn_maps: list of [B, heads, H'W', S]

                # ── (c) Build spatial attention mask ─────────────────
                if explicit_mask is not None:
                    spatial_mask = explicit_mask
                elif cross_attn_maps:
                    spatial_mask = build_attention_mask(
                        attention_maps=cross_attn_maps,
                        latent_shape=z_t.shape,
                        threshold=cfg.attention_threshold,
                        blur_sigma=cfg.mask_blur_sigma,
                    )  # [1, H', W', 1]
                else:
                    # fallback: edit everywhere
                    spatial_mask = mx.ones_like(z_t[:, :, :, :1])

                attn_mask = spatial_mask

                # ── (d) Standard reverse step ────────────────────────
                if cfg.scheduler_type == "ddim":
                    z_prev_standard = self._reverse_scheduler.step(
                        model_output=eps,
                        timestep=t,
                        sample=z_t,
                    )
                else:
                    # Rectified flow: controlled reverse ODE
                    z_prev_standard = self._reverse_scheduler.step_reverse(
                        model_output=eps,
                        timestep=t,
                        sample=z_t,
                        reference=z_source,
                        eta=cfg.rf_eta,
                    )

                # ── (e) Compute source latent at current noise level ─
                z_source_t = self._noise_source_to_level(
                    z_source=z_source,
                    timestep=t,
                    step_idx=step_idx,
                )

                # ── (f) Sparse proximal update ───────────────────────
                # Progress fraction for cosine-annealed λ
                progress = step_idx / max(num_steps - 1, 1)
                lam = cosine_anneal_lambda(
                    progress=progress,
                    lambda_max=cfg.lambda_max,
                    lambda_min=cfg.lambda_min,
                )
                # Safeguard: λ ≥ 0
                lam = max(lam, 0.0)

                delta = z_prev_standard - z_source_t
                delta_sparse = soft_threshold(delta, lam)

                # ── (g) Latent surgery: blend via mask ───────────────
                z_t = latent_blend(
                    z_source=z_source_t,
                    z_edited=z_source_t + delta_sparse,
                    mask=spatial_mask,
                    strength=cfg.blend_strength,
                )

                # Evaluate to free graph memory
                mx.eval(z_t)

                # ── (h) Optionally collect intermediates ─────────────
                if cfg.collect_intermediates:
                    trajectory.append(z_t)

                # Clear hooks for next iteration
                self._hook_manager.clear_maps()

        finally:
            self._hook_manager.remove_hooks()

        return z_t, attn_mask, trajectory

    # ───────────────────────────────────────────────────────────────
    #  Internal: Noise level matching
    # ───────────────────────────────────────────────────────────────
    def _noise_source_to_level(
        self,
        z_source: mx.array,
        timestep: int | float,
        step_idx: int,
    ) -> mx.array:
        """Add the appropriate amount of noise to z_source to match
        the noise level at ``timestep``.

        For DDIM this uses the forward diffusion formula:
            z_t = √ᾱ_t · z₀ + √(1 − ᾱ_t) · ε
        with ε = 0 (deterministic reference – no stochastic noise added).

        For rectified flow this is the linear interpolation:
            z_t = (1 − t) · z₀ + t · z_T

        Parameters
        ----------
        z_source : mx.array
            Clean source latent z₀.
        timestep : int | float
            Current scheduler timestep.
        step_idx : int
            Integer step index (used for rectified flow normalisation).

        Returns
        -------
        mx.array
            Noised source latent at the given noise level.
        """
        if self.config.scheduler_type == "ddim":
            # Retrieve αbar_t from the reverse scheduler
            alpha_bar_t = self._reverse_scheduler.get_alpha_bar(timestep)
            sqrt_alpha = mx.sqrt(mx.array(alpha_bar_t, dtype=self.config.dtype))
            # Deterministic reference: noise component is zero
            z_source_t = sqrt_alpha * z_source
            return z_source_t
        else:
            # Rectified flow: linear interpolation fraction
            total = self.config.num_inference_steps
            t_frac = mx.array(
                1.0 - step_idx / max(total - 1, 1),
                dtype=self.config.dtype,
            )
            z_source_t = (1.0 - t_frac) * z_source
            return z_source_t

    # ───────────────────────────────────────────────────────────────
    #  Internal: Metrics
    # ───────────────────────────────────────────────────────────────
    def _compute_metrics(
        self,
        source_np: npt.NDArray[np.uint8],
        edited_np: npt.NDArray[np.uint8],
        mask_np: npt.NDArray[np.float32] | None,
        target_prompt: str,
    ) -> dict[str, float]:
        """Compute quality metrics between source and edited images.

        Parameters
        ----------
        source_np, edited_np : ndarray
            HWC uint8 images.
        mask_np : ndarray | None
            Spatial mask [H, W] in [0, 1].
        target_prompt : str
            Target text for CLIP score.

        Returns
        -------
        dict
            Metric name → value.
        """
        src_f = source_np.astype(np.float32)
        edt_f = edited_np.astype(np.float32)

        results: dict[str, float] = {}

        # PSNR (full image)
        results["psnr"] = float(compute_psnr(src_f, edt_f, data_range=255.0))

        # SSIM (full image)
        results["ssim"] = float(compute_ssim(src_f, edt_f, data_range=255.0))

        # Leakage (outside-mask MSE)
        if mask_np is not None:
            leakage = compute_leakage(
                source=src_f,
                edited=edt_f,
                mask=mask_np,
            )
            results.update(
                {f"leakage_{k}": float(v) for k, v in leakage.items()}
            )

        # CLIP score is only available in pre-computed mode by default.
        # If embeddings are not pre-computed, we skip gracefully.
        try:
            results["clip_score"] = float(
                compute_clip_score(
                    image=edited_np,
                    text=target_prompt,
                )
            )
        except (NotImplementedError, TypeError):
            pass

        # LPIPS
        try:
            results["lpips"] = float(
                compute_lpips(source_np, edited_np)
            )
        except (NotImplementedError, TypeError):
            pass

        return results

    # ───────────────────────────────────────────────────────────────
    #  Utilities
    # ───────────────────────────────────────────────────────────────
    @staticmethod
    def _ensure_numpy_uint8(
        img: npt.NDArray[np.uint8] | mx.array,
    ) -> npt.NDArray[np.uint8]:
        """Convert any image input to HWC uint8 NumPy."""
        if isinstance(img, mx.array):
            arr = np.array(img)
            if arr.ndim == 4:
                arr = arr[0]  # remove batch dim
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 127.5 + 127.5, 0, 255).astype(np.uint8)
            return arr
        arr = np.asarray(img)
        if arr.ndim == 4:
            arr = arr[0]
        return arr.astype(np.uint8)

    @staticmethod
    def _to_mx(arr: npt.NDArray | mx.array | None) -> mx.array | None:
        """Convert NumPy array to mx.array if needed."""
        if arr is None:
            return None
        if isinstance(arr, mx.array):
            return arr
        a = mx.array(np.asarray(arr, dtype=np.float32))
        # Ensure shape is [1, H, W, 1] for broadcasting
        if a.ndim == 2:
            a = a[None, :, :, None]
        elif a.ndim == 3:
            a = a[None, :, :, :]
        return a

    def __repr__(self) -> str:
        return (
            f"SparseEditPipeline("
            f"scheduler={self.config.scheduler_type!r}, "
            f"steps={self.config.num_inference_steps}, "
            f"guidance={self.config.guidance_scale}, "
            f"λ=[{self.config.lambda_min}, {self.config.lambda_max}]"
            f")"
        )
