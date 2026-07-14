"""
Test script: compare original vs preprocessed image.

Usage:
    python tests/compare_preprocessing.py path/to/image.jpg [options]

Options:
    --output-dir       Directory to save results (default: comparison_results/)
    --max-dim          Max image dimension (default: 2000)
    --apply-threshold  Apply binary thresholding
"""
import sys
import os
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.region_preprocess import process_image


def compare_original_vs_preprocessed(
    image_path: str,
    output_dir: str = "comparison_results",
    **kwargs,
) -> dict:
    """Load image, preprocess, and save side-by-side comparison.

    Returns:
        dict with paths: {"original", "preprocessed", "side_by_side"}
    """
    # ── 1. Load original ───────────────────────────────────────────────────
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    logger.info(f"Original size: {original.shape[1]}x{original.shape[0]}")

    # ── 2. Preprocess ──────────────────────────────────────────────────────
    preprocessed = process_image(
        original,
        image_path=image_path,
        **kwargs,
    )

    # Convert back to BGR for saving with OpenCV
    if len(preprocessed.shape) == 2:
        preprocessed_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
    elif preprocessed.shape[2] == 4:
        preprocessed_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_RGBA2BGR)
    else:
        preprocessed_bgr = preprocessed

    logger.info(f"Preprocessed size: {preprocessed.shape[1]}x{preprocessed.shape[0]}")

    # ── 3. Create side-by-side ─────────────────────────────────────────────
    # Resize preprocessed to match original height for stacking
    orig_h, orig_w = original.shape[:2]
    prep_h, prep_w = preprocessed.shape[:2]

    scale = orig_h / prep_h
    new_prep_w = int(prep_w * scale)
    preprocessed_resized = cv2.resize(preprocessed_bgr, (new_prep_w, orig_h))

    # Ensure same width for side-by-side
    max_width = max(orig_w, new_prep_w)
    if orig_w < max_width:
        pad = np.zeros((orig_h, max_width - orig_w, 3), dtype=np.uint8)
        original_padded = cv2.hconcat([original, pad])
    else:
        original_padded = original

    if new_prep_w < max_width:
        pad = np.zeros((orig_h, max_width - new_prep_w, 3), dtype=np.uint8)
        preprocessed_padded = cv2.hconcat([preprocessed_resized, pad])
    else:
        preprocessed_padded = preprocessed_resized

    side_by_side = cv2.hconcat([
        original_padded,
        np.ones((orig_h, 10, 3), dtype=np.uint8) * 128,  # grey divider
        preprocessed_padded,
    ])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(side_by_side, "ORIGINAL", (20, 40), font, 1.2, (0, 255, 0), 3)
    label_x = max_width + 10 + 20
    cv2.putText(side_by_side, "PREPROCESSED", (label_x, 40), font, 1.2, (0, 255, 0), 3)

    # ── 4. Save results ────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(image_path).stem

    orig_out = os.path.join(output_dir, f"{stem}_original.jpg")
    prep_out = os.path.join(output_dir, f"{stem}_preprocessed.jpg")
    sbs_out = os.path.join(output_dir, f"{stem}_comparison.jpg")

    cv2.imwrite(orig_out, original)
    cv2.imwrite(prep_out, preprocessed_bgr)
    cv2.imwrite(sbs_out, side_by_side)

    logger.info(f"Saved comparison: {sbs_out}")

    return {
        "original": orig_out,
        "preprocessed": prep_out,
        "side_by_side": sbs_out,
        "original_size": f"{orig_w}x{orig_h}",
        "preprocessed_size": f"{preprocessed.shape[1]}x{preprocessed.shape[0]}",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare original vs preprocessed image"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output-dir", default="comparison_results",
                        help="Output directory")
    parser.add_argument("--max-dim", type=int, default=2000,
                        help="Max image dimension")
    parser.add_argument("--apply-threshold", action="store_true",
                        help="Apply binary thresholding")
    args = parser.parse_args()

    result = compare_original_vs_preprocessed(
        args.image,
        output_dir=args.output_dir,
        max_image_dim=args.max_dim,
        apply_threshold=args.apply_threshold,
    )

    print("\nComparison complete!")
    print(f"   Original size:      {result['original_size']}")
    print(f"   Preprocessed size:  {result['preprocessed_size']}")
    print(f"   Side-by-side:       {result['side_by_side']}")