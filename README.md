
```
░██████╗██████╗  █████╗ ██████╗ ███████╗███████╗██████╗ ██╗████████╗
██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗██║╚══██╔══╝
╚█████╗ ██████╔╝███████║██████╔╝███████╗█████╗  ██║  ██║██║   ██║
 ╚═══██╗██╔═══╝ ██╔══██║██╔══██╗╚════██║██╔══╝  ██║  ██║██║   ██║
██████╔╝██║     ██║  ██║██║  ██║███████║███████╗██████╔╝██║   ██║
╚═════╝ ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═════╝ ╚═╝   ╚═╝
```
> *"What the field provides is a latent paintbrush. What users need is a latent scalpel."*

**Sparse Latent Surgery for Zero-Leakage Diffusion Image Editing**  
NJIT · Deep Learning SP26 · Chauhan & Desai

---

## ⸸ The Problem

Ask a diffusion model to *change the hat to red* — and it obeys. It also corrupts the background, drifts the skin tones, and after six edits, steals the subject's face entirely. This is **latent leakage**, and it is structural: the denoiser modifies every latent dimension at every step, unconstrained.

---

## ⸸ The Cure

SparseEdit injects an **L1 proximal gradient step** into the denoising loop at every timestep, forcing over **85% of latent coordinates to exactly zero**:

```
δ* = sign(δ̃) ⊙ max(|δ̃| − ητ, 0)
```

No new model. No user-drawn mask. One tunable knob: **λ**.  
Edit regions are discovered automatically via cross-attention.

---

## ⸸ Stack

| Component | Source | Size |
|---|---|---|
| SDXL Turbo | Stability AI | ~5.1 GB |
| SD3 16-ch VAE | Stability AI | ~335 MB |
| CLIP ViT-L/14 + OpenCLIP ViT-bigG | OpenAI / LAION | ~1.6 GB |

Runs on **Apple Silicon (MLX)**. ~17–20s per image. Resolution: 512×512.

---

## ⸸ Invocation

```bash
# Gradio altar
python app.py

# CLI ritual
python cli_edit.py \
  --source image.png \
  --src_prompt "a cat wearing a blue hat" \
  --tgt_prompt "a cat wearing a red hat, studio lighting" \
  --lambda 0.05 --tau 0.02 --eta 0.1 --steps 4
```

---

## ⸸ Targets

| Metric | Target |
|---|---|
| Background PSNR | > 40 dB |
| Background SSIM | > 0.98 |
| Leakage Score | < 0.01 |
| Sparsity Ratio | > 0.85 |

Evaluated on **PIE-Bench**. Full quantitative results in progress.

---

*The latent space is vast and unconstrained. SparseEdit is the hand that holds the scalpel still.*
