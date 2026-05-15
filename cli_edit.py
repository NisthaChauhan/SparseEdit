"""
cli_edit.py
-----------
Headless prompt-driven editor.

Example:
    python cli_edit.py --image input_kitchen.png \
                       --prompt "replace cabinets with oak wood" \
                       --out output.png --steps 30
"""

from __future__ import annotations

import argparse
import sys

from PIL import Image

from ui_backend import EditRequest, edit_image


def main() -> int:
    ap = argparse.ArgumentParser(description="SparseEdit CLI")
    ap.add_argument("--image", required=True, help="Path to source image")
    ap.add_argument("--prompt", required=True, help="Edit prompt")
    ap.add_argument("--source-prompt", default="", help="Optional source caption")
    ap.add_argument("--out", default="output.png", help="Output path")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=1024)
    args = ap.parse_args()

    src = Image.open(args.image).convert("RGB")
    req = EditRequest(
        source_image=src,
        prompt=args.prompt,
        source_prompt=args.source_prompt,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        lambda_sparsity=args.lam,
        tau=args.tau,
        eta=args.eta,
        seed=args.seed,
        height=args.res,
        width=args.res,
    )

    def _cb(step, total):
        bar = "#" * int(40 * step / max(total, 1))
        sys.stdout.write(f"\r[{bar:<40}] {step}/{total}")
        sys.stdout.flush()

    result = edit_image(req, progress_cb=_cb)
    print()
    result.edited_image.save(args.out)
    print(f"Saved -> {args.out}  ({result.elapsed_s:.1f}s)")
    for k, v in result.metrics.items():
        print(f"  {k:>12}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
