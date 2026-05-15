"""CLI script for running SparseEdit evaluation on a batch of edits.

Usage
-----
    python -m sparse_edit.cli.evaluate \\
        --source-dir ./data/sources/ \\
        --edited-dir ./data/edited/ \\
        --mask-dir ./data/masks/ \\
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_image(path: str | Path) -> np.ndarray:
    """Load an image as float64 array."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float64)


def _load_mask(path: str | Path) -> np.ndarray:
    """Load a mask as float64 array normalised to [0, 1]."""
    from PIL import Image
    mask = Image.open(path).convert("L")
    arr = np.array(mask, dtype=np.float64) / 255.0
    return arr[:, :, np.newaxis]


def main(argv: list[str] | None = None) -> None:
    """Entry point for evaluation CLI."""
    parser = argparse.ArgumentParser(description="SparseEdit evaluation harness")
    parser.add_argument("--source-dir", type=str, required=True)
    parser.add_argument("--edited-dir", type=str, required=True)
    parser.add_argument("--mask-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--data-range", type=float, default=255.0)
    parser.add_argument(
        "--extensions", type=str, default=".png,.jpg,.jpeg,.bmp",
        help="Comma-separated image file extensions.",
    )
    args = parser.parse_args(argv)

    from sparse_edit.metrics.evaluation import evaluate_edit, aggregate_results

    extensions = tuple(args.extensions.split(","))
    source_dir = Path(args.source_dir)
    edited_dir = Path(args.edited_dir)
    mask_dir = Path(args.mask_dir)

    source_files = sorted(
        f for f in source_dir.iterdir()
        if f.suffix.lower() in extensions
    )

    if not source_files:
        print(f"No images found in {source_dir} with extensions {extensions}")
        sys.exit(1)

    results = []
    per_image: list[dict] = []

    for src_path in source_files:
        edt_path = edited_dir / src_path.name
        mask_path = mask_dir / src_path.with_suffix(".png").name

        if not edt_path.exists():
            print(f"  SKIP (no edited): {src_path.name}")
            continue
        if not mask_path.exists():
            print(f"  SKIP (no mask): {src_path.name}")
            continue

        source = _load_image(src_path)
        edited = _load_image(edt_path)
        mask = _load_mask(mask_path)

        if mask.shape[:2] != source.shape[:2]:
            from PIL import Image
            mask_pil = Image.fromarray(
                (mask[:, :, 0] * 255).astype(np.uint8)
            ).resize((source.shape[1], source.shape[0]), Image.NEAREST)
            mask = np.array(mask_pil, dtype=np.float64)[:, :, np.newaxis] / 255.0

        ev = evaluate_edit(source, edited, mask, data_range=args.data_range)
        per_image.append({"filename": src_path.name, **ev.to_dict()})
        results.append(ev)
        print(f"  {src_path.name}: {ev.summary()}")

    agg = aggregate_results(results)

    output = {"aggregate": agg, "per_image": per_image}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {args.output}")
    print(f"Aggregate ({len(results)} images):")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
