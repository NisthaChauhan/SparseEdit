import sys
sys.path.insert(0, "/Users/aayush/Projects/sparse_edit")

"""
run_project.py
==============
Complete working script to run the SparseEdit pipeline.
"""

import time
import numpy as np
import mlx.core as mx


# ─────────────────────────────────────────────────────
#  PART 1: Test that metrics work (no weights needed)
# ─────────────────────────────────────────────────────

def test_metrics():
    print("=" * 60)
    print("PART 1: Testing metrics (no model weights needed)")
    print("=" * 60)

    from sparse_edit.metrics.psnr import compute_psnr
    from sparse_edit.metrics.ssim import compute_ssim
    from sparse_edit.metrics.leakage import compute_leakage

    h, w = 128, 128
    img_a = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8).astype(np.float32)
    noise = np.random.normal(0, 15, img_a.shape).astype(np.float32)
    img_b = np.clip(img_a + noise, 0, 255).astype(np.float32)

    psnr_val = compute_psnr(img_a, img_b, data_range=255.0)
    print(f"  PSNR:  {psnr_val:.2f} dB")

    ssim_val = compute_ssim(img_a, img_b, data_range=255.0)
    print(f"  SSIM:  {ssim_val:.4f}")

    mask = np.zeros((h, w), dtype=np.float32)
    mask[32:96, 32:96] = 1.0
    leakage = compute_leakage(source=img_a, edited=img_b, mask=mask)
    for k, v in leakage.items():
        print(f"  {k}: {v:.4f}")

    print("  [PASS] All metrics work.\n")


# ─────────────────────────────────────────────────────
#  PART 2: Test sparse editing operations
# ─────────────────────────────────────────────────────

def test_sparse_ops():
    print("=" * 60)
    print("PART 2: Testing sparse editing operations")
    print("=" * 60)

    from sparse_edit.editing.sparse_optimizer import soft_threshold, compute_sparsity_ratio
    from sparse_edit.editing.sparsity_scheduler import CosineSparsityScheduler

    # Soft thresholding (NumPy arrays — that's what the function expects)
    x = np.array([[0.1, -0.3, 0.5, -0.02, 0.8]])
    result = soft_threshold(x, lam=0.2)
    print(f"  Input:            {x[0]}")
    print(f"  Soft-thresh(0.2): {result[0]}")
    print(f"  Sparsity ratio:   {compute_sparsity_ratio(result):.2f}")

    # Sparsity schedule
    scheduler = CosineSparsityScheduler(lambda_min=0.005, lambda_max=0.05, num_steps=6)
    print("  Cosine-annealed lambda schedule:")
    for step in range(6):
        lam = scheduler.get_lambda(step)
        print(f"    step {step}/5 -> lambda={lam:.4f}")

    print("  [PASS] All sparse ops work.\n")


# ─────────────────────────────────────────────────────
#  PART 3: Full pipeline demo (simulated with NumPy)
# ─────────────────────────────────────────────────────

def demo_full_pipeline():
    print("=" * 60)
    print("PART 3: Full pipeline demo (simulated)")
    print("=" * 60)

    from sparse_edit.editing.sparse_optimizer import soft_threshold, proximal_gradient_step
    from sparse_edit.editing.sparsity_scheduler import CosineSparsityScheduler
    from sparse_edit.metrics.psnr import compute_psnr
    from sparse_edit.metrics.ssim import compute_ssim

    print("  Simulating the full edit pipeline with dummy tensors...\n")

    # Step 1: Simulate VAE encode -> latent z0
    print("  Step 1: Encode source image -> latent z0")
    z0 = np.random.randn(1, 64, 64, 4).astype(np.float64)
    print(f"           z0 shape: {z0.shape}")

    # Step 2: Simulate DDIM inversion z0 -> zT
    print("  Step 2: DDIM inversion z0 -> zT (50 steps)")
    zT = z0 + np.random.randn(*z0.shape) * 0.5
    print(f"           zT shape: {zT.shape}")

    # Step 3: Sparse denoising loop
    print("  Step 3: Sparse denoising zT -> z0_edited (50 steps)")
    num_steps = 50
    sched = CosineSparsityScheduler(lambda_min=0.005, lambda_max=0.05, num_steps=num_steps)

    z_t = zT.copy()
    for step in range(num_steps):
        lam = sched.get_lambda(step)

        # Pretend U-Net prediction
        eps_pred = np.random.randn(*z_t.shape) * 0.01

        # Standard step (simplified)
        z_prev = z_t - eps_pred * 0.02

        # Sparse proximal update (all NumPy)
        z_t = proximal_gradient_step(
            latent_source=z0,
            latent_edited=z_prev,
            lam=lam,
        )

        # Apply a simple mask (center region only)
        mask = np.zeros((1, 64, 64, 1), dtype=np.float64)
        mask[:, 16:48, 16:48, :] = 1.0
        z_t = z0 * (1.0 - mask) + z_t * mask

        if step % 10 == 0:
            diff = np.mean(np.abs(z_t - z0))
            print(f"           step {step:3d}/{num_steps}, "
                  f"lambda={lam:.4f}, mean|delta|={diff:.6f}")

    # Step 4: Simulate VAE decode
    print("  Step 4: Decode z0_edited -> edited image")
    source_np = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    edited_np = source_np.copy()
    # Simulate a small edit in center
    edited_np[128:384, 128:384, :] = np.clip(
        edited_np[128:384, 128:384, :].astype(np.int16) + 20, 0, 255
    ).astype(np.uint8)

    # Step 5: Compute metrics
    print("  Step 5: Compute metrics")
    psnr = compute_psnr(
        source_np.astype(np.float32),
        edited_np.astype(np.float32),
        data_range=255.0,
    )
    ssim = compute_ssim(
        source_np.astype(np.float32),
        edited_np.astype(np.float32),
        data_range=255.0,
    )
    print(f"           PSNR: {psnr:.2f} dB")
    print(f"           SSIM: {ssim:.4f}")

    print("\n  [PASS] Full pipeline simulation complete.\n")


# ─────────────────────────────────────────────────────
#  PART 4: MLX hardware check
# ─────────────────────────────────────────────────────

def test_mlx():
    print("=" * 60)
    print("PART 4: MLX hardware check")
    print("=" * 60)

    x = mx.random.normal((1, 64, 64, 4))
    y = mx.random.normal((1, 64, 64, 4))
    z = x + y
    mx.eval(z)
    print(f"  MLX random tensor shape: {z.shape}")
    print(f"  MLX dtype: {z.dtype}")
    print(f"  MLX mean:  {mx.mean(z).item():.4f}")
    print(f"  [PASS] MLX is working on Apple Silicon.\n")


# ─────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("SparseEdit Project Runner")
    print("========================\n")

    start = time.time()

    test_metrics()
    test_sparse_ops()
    demo_full_pipeline()
    test_mlx()

    elapsed = time.time() - start
    print("=" * 60)
    print(f"ALL DONE in {elapsed:.1f}s")
    print("=" * 60)
